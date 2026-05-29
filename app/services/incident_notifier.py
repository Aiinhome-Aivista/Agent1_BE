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
    Send the initial mail to ALL active DataOps Engineers (L1).

    Returns
    -------
    (recipient_emails, recipient_role) — both may be None if no recipient was
    available or SMTP is not configured.
    """
    db = SessionLocal()
    try:
        inc = db.query(Incident).filter(Incident.id == incident_id).first()
        if not inc:
            return None, None

        from app.models import User
        from app.models.user import UserRole
        # Fetch all active L1 engineers
        recipients = (
            db.query(User)
            .filter(User.is_active == True, User.role == UserRole.dataops_engineer)
            .all()
        )
        role_str = "DataOps Engineer"

        # Fallback to admins if no L1 is active
        if not recipients:
            recipients = (
                db.query(User)
                .filter(User.is_active == True, User.is_admin == True)
                .all()
            )
            role_str = "Admin"

        if not recipients:
            logger.info("No initial recipient available for incident %s", incident_id)
            return None, None

        emails = [r.email for r in recipients if r.email]
        if not emails:
            return None, None

        payload = _incident_payload(inc)
        llm     = _llm_block_from_incident(inc, raw_result)

        # SMTP can be slow — push to a worker thread so the agent loop stays snappy.
        sent_ok = True
        for email in emails:
            ok = await asyncio.to_thread(
                email_service.send_incident_alert, email, payload, llm,
            )
            if not ok:
                sent_ok = False

        now = datetime.utcnow()
        recipient_emails_str = ", ".join(emails)
        
        # Persist the send to the incident table
        inc.initial_email_sent_at   = now
        inc.initial_email_recipient = recipient_emails_str
        inc.initial_email_role      = role_str
        db.commit()

        return recipient_emails_str, role_str
    except Exception:
        logger.exception("notify_incident_diagnosed failed for incident %s", incident_id)
        return None, None
    finally:
        db.close()



# ────────────────────────────────────────────────────────────────────
# OLD one-shot escalation (REMOVED)
# ────────────────────────────────────────────────────────────────────
# schedule_incident_escalation() and _escalation_after_delay() have been
# removed.  Escalation is now handled by the recurring scheduler in
# ``app.services.escalation_scheduler`` which checks active incidents
# every ESCALATION_CHECK_INTERVAL seconds.

