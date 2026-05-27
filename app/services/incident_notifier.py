"""
Tiny dispatch layer between `incident_service` and `email_service`.

Responsibilities
----------------
1. Pick the right initial recipient (first DataOps Engineer).
2. Fire the initial alert email (non-blocking — runs in a thread so SMTP
   latency never stalls the incident loop). Persist the send time, the
   recipient email and the recipient role on the incident row so the
   Incident Timeline page can render them.
3. Schedule a follow-up coroutine that fires after
   `INCIDENT_ESCALATION_MINUTES` minutes; if the incident is still
   AWAITING_APPROVAL at that point, it mails everyone HIGHER than the
   initial recipient on the role ladder, then persists when it fired and
   who got it.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from app.core.database import SessionLocal
from app.models.agent_models import Incident, IncidentStatus
from app.services import email_service

logger = logging.getLogger(__name__)


def _escalation_seconds() -> int:
    """Default 15 minutes. Override with INCIDENT_ESCALATION_MINUTES env var.
    Use INCIDENT_ESCALATION_SECONDS for fast demos."""
    secs = os.getenv("INCIDENT_ESCALATION_SECONDS")
    if secs:
        try:
            return max(5, int(secs))
        except ValueError:
            pass
    mins = os.getenv("INCIDENT_ESCALATION_MINUTES", "15")
    try:
        return max(60, int(mins) * 60)
    except ValueError:
        return 15 * 60


def _llm_block_from_incident(inc: Incident, raw_result: dict[str, Any] | None) -> dict[str, Any]:
    """Build the dict the email templates expect. Prefer the live LLM result
    that was just produced; fall back to whatever is stored on the incident."""
    r = raw_result or {}
    return {
        "summary":       r.get("summary")       or inc.agent_thought or "",
        "root_cause":    r.get("root_cause")    or inc.root_cause    or "",
        "suggested_fix": r.get("suggested_fix") or "\n".join(inc.remediation_plan or []),
        "confidence":    r.get("confidence")    or inc.confidence_score,
        "used_context":  r.get("used_context"),
    }


def _incident_payload(inc: Incident) -> dict[str, Any]:
    """Serialise just enough fields for the email body."""
    return {
        "id":              inc.id,
        "pipeline_name":   inc.pipeline_name,
        "risk_tier":       inc.risk_tier,
        "status":          inc.status.value if hasattr(inc.status, "value") else str(inc.status),
        "detected_at":     inc.detected_at.isoformat() if inc.detected_at else "",
        "failed_node":     inc.failed_node,
        "error_log":       inc.error_log,
    }


def _role_string(user) -> str | None:
    """Return the User.role as a plain string regardless of whether the column
    yields the enum member or already the string value."""
    if user is None:
        return None
    role = getattr(user, "role", None)
    if role is None:
        return None
    return role.value if hasattr(role, "value") else str(role)


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

async def notify_incident_diagnosed(
    incident_id: int,
    raw_result: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    """
    Send the initial mail to the first DataOps Engineer.

    Returns
    -------
    (recipient_email, recipient_role) — both may be None if no recipient was
    available or SMTP is not configured. The role string is needed by
    `schedule_incident_escalation` to escalate UP the hierarchy.
    """
    db = SessionLocal()
    try:
        inc = db.query(Incident).filter(Incident.id == incident_id).first()
        if not inc:
            return None, None

        recipient = email_service.pick_initial_recipient(db)
        if not recipient or not recipient.email:
            logger.info("No initial recipient available for incident %s", incident_id)
            return None, None

        recipient_email = recipient.email
        recipient_role  = _role_string(recipient)

        payload = _incident_payload(inc)
        llm     = _llm_block_from_incident(inc, raw_result)

        # SMTP can be slow — push to a worker thread so the agent loop
        # stays snappy.
        sent_ok = await asyncio.to_thread(
            email_service.send_incident_alert, recipient_email, payload, llm,
        )

        # Persist the send regardless of SMTP success/failure: if the row
        # says "sent at T1 to ..." that's the truth of what was attempted,
        # and the audit log will record the failure separately. We DO
        # however prefer to only record the timestamp when SMTP returned
        # True, so the timeline doesn't lie about deliveries.
        now = datetime.utcnow()
        if sent_ok:
            inc.initial_email_sent_at   = now
            inc.initial_email_recipient = recipient_email
            inc.initial_email_role      = recipient_role
            db.commit()

        return recipient_email, recipient_role
    except Exception:
        logger.exception("notify_incident_diagnosed failed for incident %s", incident_id)
        return None, None
    finally:
        db.close()


def schedule_incident_escalation(
    incident_id: int,
    primary_email: str | None,
    primary_role:  str | None,
    raw_result:    dict[str, Any] | None,
) -> None:
    """
    Fire-and-forget: schedule a coroutine that, after the escalation delay,
    checks whether the incident is still unactioned and, if so, mails
    everyone HIGHER on the role ladder than `primary_role`.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop — cannot schedule escalation for %s", incident_id)
        return

    delay_s = _escalation_seconds()
    logger.info("Escalation for incident %s scheduled in %ds (primary_role=%s)",
                incident_id, delay_s, primary_role)

    loop.create_task(
        _escalation_after_delay(incident_id, primary_email, primary_role, raw_result, delay_s)
    )


async def _escalation_after_delay(
    incident_id:   int,
    primary_email: str | None,
    primary_role:  str | None,
    raw_result:    dict[str, Any] | None,
    delay_s:       int,
) -> None:
    try:
        await asyncio.sleep(delay_s)
    except asyncio.CancelledError:
        return

    if email_service.was_escalated(incident_id):
        return

    db = SessionLocal()
    try:
        inc = db.query(Incident).filter(Incident.id == incident_id).first()
        if not inc:
            return

        # If the engineer already acted, no need to escalate.
        if inc.status != IncidentStatus.AWAITING_APPROVAL:
            logger.info("Incident %s no longer awaiting approval (now %s); skip escalation",
                        incident_id, inc.status)
            return

        # Find the originally-notified user so we can exclude them from the
        # escalation list (they were already paged).
        original = None
        if primary_email:
            from app.models import User                                            # noqa: PLC0415
            original = db.query(User).filter(User.email == primary_email).first()

        recipients = email_service.pick_escalation_recipients(
            db,
            exclude_user_id=original.id if original else None,
            primary_role=primary_role,
        )
        if not recipients:
            logger.info("No escalation recipients above role %r for incident %s",
                        primary_role, incident_id)
            return

        to_emails = [u.email for u in recipients]
        recipient_records = [
            {"email": u.email, "role": _role_string(u) or "Unknown"}
            for u in recipients
        ]

        payload = _incident_payload(inc)
        llm     = _llm_block_from_incident(inc, raw_result)

        ok = await asyncio.to_thread(
            email_service.send_incident_escalation,
            to_emails, payload, llm, primary_email, primary_email,
        )
        if ok:
            email_service.mark_escalated(incident_id)

            inc.escalation_email_sent_at    = datetime.utcnow()
            inc.escalation_email_recipients = recipient_records

            # Audit the escalation on the incident timeline too.
            tl = list(inc.timeline or [])
            tl.append({
                "ts":     datetime.utcnow().isoformat(),
                "stage":  "Escalation Email Sent",
                "agent":  "orchestrator",
                "detail": (
                    f"No response within SLA — escalation mail sent to "
                    f"{len(to_emails)} senior member(s): "
                    f"{', '.join(r['role'] for r in recipient_records)}"
                ),
            })
            inc.timeline = tl
            db.commit()
    except Exception:
        logger.exception("Escalation check failed for incident %s", incident_id)
    finally:
        db.close()
