"""
Recurring escalation scheduler.

Runs every ``ESCALATION_CHECK_INTERVAL`` seconds (default 5 min) via
APScheduler.  For each **active** incident it:

1. Counts the pipeline's current runs and compares to the stored snapshot.
2. If **no new run** was added  →  sends an escalation mail to L1 + L2 + L3.
3. If a **new run** is detected:
   a. RUNNING / QUEUED  →  skip, wait for it to finish.
   b. SUCCEEDED         →  resolve the incident.
   c. FAILED            →  start a new escalation cycle (L1 mail first).
"""
from __future__ import annotations

import logging
from datetime import datetime

# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.agent_models import (
    Incident,
    IncidentEvent,
    IncidentEventType,
    IncidentStatus,
)
from app.models.pipeline import Pipeline, PipelineRun, RunStatus
from app.models import User
from app.models.user import UserRole
from app.services import email_service
from app.services.incident_service import _incident_to_dict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _log_event(
    db: Session,
    incident_id: int,
    event_type: str,
    *,
    escalation_level: str | None = None,
    recipients: list[dict] | None = None,
    related_run_id: int | None = None,
    details: str | None = None,
) -> IncidentEvent:
    """Insert a row into ``incident_events`` and return it."""
    evt = IncidentEvent(
        incident_id=incident_id,
        event_type=event_type,
        escalation_level=escalation_level,
        recipients=recipients or [],
        related_run_id=related_run_id,
        details=details,
    )
    db.add(evt)
    db.commit()
    db.refresh(evt)
    return evt


def _pick_l1_users(db: Session) -> list[User]:
    """All active DataOps Engineers (L1)."""
    return (
        db.query(User)
        .filter(User.is_active == True, User.role == UserRole.dataops_engineer)  # noqa: E712
        .order_by(User.id.asc())
        .all()
    )


def _pick_l2_l3_users(db: Session) -> list[User]:
    """Data Platform Leads (L2) + Data Engineers (L3)."""
    return (
        db.query(User)
        .filter(
            User.is_active == True,  # noqa: E712
            User.role.in_([UserRole.data_platform_lead, UserRole.data_engineer]),
        )
        .order_by(User.id.asc())
        .all()
    )


def _send_l1_mail(db: Session, inc: Incident, run: PipelineRun | None) -> None:
    """Send initial alert to L1 (DataOps/SRE Engineers)."""
    l1_users = _pick_l1_users(db)
    if not l1_users:
        logger.warning("No L1 users found for escalation mail")
        return

    inc_dict = _incident_to_dict(inc)
    llm_block = {
        "summary": inc.agent_thought,
        "root_cause": inc.root_cause,
        "suggested_fix": inc.proposed_action,
        "confidence": inc.confidence_score,
    }

    recipients_meta = []
    for u in l1_users:
        if u.email:
            email_service.send_incident_alert(u.email, inc_dict, llm_block)
            recipients_meta.append({"email": u.email, "role": u.role.value if hasattr(u.role, "value") else str(u.role)})

    if recipients_meta:
        inc.initial_email_sent_at = datetime.utcnow()
        inc.initial_email_recipient = ", ".join([r["email"] for r in recipients_meta])
        inc.initial_email_role = ", ".join(list(set([r["role"] for r in recipients_meta])))
        db.commit()

        _log_event(
            db, inc.id,
            IncidentEventType.INITIAL_MAIL_SENT.value,
            escalation_level="L1",
            recipients=recipients_meta,
            related_run_id=run.id if run else None,
            details=f"Initial alert sent to {len(recipients_meta)} L1 engineer(s)",
        )


def _send_escalation_mail(db: Session, inc: Incident) -> None:
    """Send escalation mail to L1 + L2 + L3."""
    l1_users = _pick_l1_users(db)
    l2_l3_users = _pick_l2_l3_users(db)
    all_users = l1_users + l2_l3_users

    if not all_users:
        logger.warning("No users found for escalation mail")
        return

    inc_dict = _incident_to_dict(inc)
    llm_block = {
        "summary": inc.agent_thought,
        "root_cause": inc.root_cause,
        "suggested_fix": inc.proposed_action,
        "confidence": inc.confidence_score,
    }

    to_emails = [u.email for u in all_users if u.email]
    primary_notified = inc.initial_email_recipient

    if to_emails:
        email_service.send_incident_escalation(
            to_emails, inc_dict, llm_block,
            primary_notified=primary_notified,
        )

    recipients_meta = [
        {"email": u.email, "role": u.role.value if hasattr(u.role, "value") else str(u.role)}
        for u in all_users if u.email
    ]

    inc.escalation_count = (inc.escalation_count or 0) + 1
    inc.last_escalation_at = datetime.utcnow()
    inc.escalation_email_sent_at = datetime.utcnow()
    inc.escalation_email_recipients = recipients_meta
    inc.status = IncidentStatus.ESCALATED
    db.commit()

    _log_event(
        db, inc.id,
        IncidentEventType.ESCALATION_MAIL_SENT.value,
        escalation_level="L1_L2_L3",
        recipients=recipients_meta,
        details=f"Escalation #{inc.escalation_count} sent to {len(recipients_meta)} engineer(s)",
    )


# ──────────────────────────────────────────────────────────────────────
# Main scheduler entry point
# ──────────────────────────────────────────────────────────────────────

async def check_active_incidents() -> None:
    """Called by APScheduler every ESCALATION_CHECK_INTERVAL seconds."""
    db: Session = SessionLocal()
    try:
        active_incidents = (
            db.query(Incident)
            .filter(Incident.is_active == True)  # noqa: E712
            .filter(Incident.pipeline_id.isnot(None))
            .all()
        )

        if not active_incidents:
            return

        logger.info("Escalation scheduler: checking %d active incidents", len(active_incidents))

        for inc in active_incidents:
            try:
                _process_one_incident(db, inc)
            except Exception:
                logger.exception("Error processing incident #%s", inc.id)
                db.rollback()

    except Exception:
        logger.exception("Escalation scheduler failed")
    finally:
        db.close()


def _process_one_incident(db: Session, inc: Incident) -> None:
    """Check a single active incident for new runs on its pipeline."""

    # Count current runs on the pipeline
    current_run_count = (
        db.query(PipelineRun)
        .filter(PipelineRun.pipeline_id == inc.pipeline_id)
        .count()
    )

    stored_count = inc.last_known_run_count or 0

    if current_run_count <= stored_count:
        # SLA target is 10800 seconds (3 hours)
        sla_seconds = 10800
        
        # Calculate precise elapsed time from creation or last escalation
        base_time = inc.last_escalation_at or inc.detected_at
        elapsed_seconds = (datetime.utcnow() - base_time).total_seconds()
        
        if elapsed_seconds < sla_seconds:
            # SLA window has not elapsed yet -> skip escalation check for this tick
            db.commit()
            return

        # ── SLA elapsed and no new runs → send escalation ──
        logger.info(
            "Incident #%s: SLA elapsed (no new runs for %ds), sending escalation",
            inc.id, int(elapsed_seconds),
        )

        _log_event(
            db, inc.id,
            IncidentEventType.ESCALATION_CHECK.value,
            details=f"SLA elapsed (no new runs detected after {int(elapsed_seconds)} seconds)",
        )

        _send_escalation_mail(db, inc)
        return

    # ── New run(s) found ──
    logger.info(
        "Incident #%s: new run detected on pipeline_id=%s (was %d, now %d)",
        inc.id, inc.pipeline_id, stored_count, current_run_count,
    )

    # Get the latest run for this pipeline
    latest_run: PipelineRun | None = (
        db.query(PipelineRun)
        .filter(PipelineRun.pipeline_id == inc.pipeline_id)
        .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
        .first()
    )

    if not latest_run:
        return

    _log_event(
        db, inc.id,
        IncidentEventType.RERUN_DETECTED.value,
        related_run_id=latest_run.id,
        details=f"New run detected: {latest_run.external_run_id} (status: {latest_run.status.value if latest_run.status else 'UNKNOWN'})",
    )

    # Update the stored count
    inc.last_known_run_count = current_run_count

    if latest_run.status in (RunStatus.RUNNING, RunStatus.QUEUED):
        # Still running → skip, check next cycle
        logger.info("Incident #%s: latest run still in progress, waiting", inc.id)
        db.commit()
        return

    if latest_run.status == RunStatus.SUCCEEDED:
        # ── Pipeline fixed! Resolve the incident ──
        logger.info("Incident #%s: latest run SUCCEEDED — resolving", inc.id)

        inc.is_active = False
        inc.status = IncidentStatus.REMEDIATED
        inc.resolved_at = datetime.utcnow()
        db.commit()

        _log_event(
            db, inc.id,
            IncidentEventType.RESOLVED.value,
            related_run_id=latest_run.id,
            details=f"Pipeline resolved — run {latest_run.external_run_id} succeeded",
        )

        # Broadcast the update via WebSocket
        try:
            from app.websockets.manager import manager
            import asyncio
            asyncio.ensure_future(
                manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
            )
        except Exception:
            pass

        return

    if latest_run.status == RunStatus.FAILED:
        # ── Rerun also failed → close old incident (new run creates new incident) ──
        logger.info("Incident #%s: latest run FAILED again — closing old incident", inc.id)

        _log_event(
            db, inc.id,
            IncidentEventType.RERUN_FAILED.value,
            related_run_id=latest_run.id,
            details=f"Rerun failed: {latest_run.external_run_id} — {latest_run.error_message or 'unknown error'}",
        )

        inc.is_active = False
        inc.status = IncidentStatus.FAILED
        db.commit()

        # Broadcast the update via WebSocket
        try:
            from app.websockets.manager import manager
            import asyncio
            asyncio.ensure_future(
                manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
            )
        except Exception:
            pass

        return

    # Other statuses (CANCELLED, UNKNOWN) — update count and wait
    db.commit()
