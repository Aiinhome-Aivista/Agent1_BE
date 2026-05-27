"""
Email notifications.

Existing behaviour (`send_pipeline_error_email`) is kept unchanged for
backward compatibility with `sync_service.py` and `connectors.py`.

New for the incident loop
-------------------------
- send_incident_alert            : initial mail to ONE recipient — the
                                   first DataOps Engineer — with the LLM's
                                   diagnosis and a deep-link to the
                                   incident page.
- send_incident_escalation       : mail to the rest of the team when the
                                   first DataOps Engineer hasn't acknowledged
                                   the incident in time.
- pick_initial_recipient         : returns the User to notify first.
- pick_escalation_recipients     : returns the Users to escalate to.
- mark_escalated / was_escalated : tiny in-memory guard so we don't
                                   double-escalate the same incident.
"""
from __future__ import annotations

import logging
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Iterable

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.models import User
from app.models.user import UserRole

load_dotenv()
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# SMTP plumbing — shared by every send_* function below
# ────────────────────────────────────────────────────────────────────

def _smtp_config() -> tuple[str, int, str, str]:
    return (
        os.getenv("SMTP_SERVER", ""),
        int(os.getenv("SMTP_PORT", "587")),
        os.getenv("SMTP_EMAIL", ""),
        os.getenv("SMTP_PASSWORD", ""),
    )


def _frontend_base() -> str:
    return os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")


def _send(
    to_emails: list[str],
    subject: str,
    body_text: str,
    body_html: str | None = None,
    cc_emails: list[str] | None = None,
) -> bool:
    """Single low-level send. Returns True on success, False on failure.
    Never raises — callers are inside the incident loop and a flaky SMTP
    should not break the diagnosis pipeline."""
    server_host, port, sender, password = _smtp_config()
    if not (server_host and sender and password):
        logger.warning("SMTP not configured (SMTP_SERVER / SMTP_EMAIL / SMTP_PASSWORD). Skipping email.")
        return False

    recipients = [e for e in (to_emails or []) if e]
    if not recipients:
        logger.info("No recipients for email %r; nothing to do.", subject)
        return False
    cc_clean = [e for e in (cc_emails or []) if e and e not in recipients]

    msg = MIMEMultipart("alternative")
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = subject
    if cc_clean:
        msg["Cc"]  = ", ".join(cc_clean)

    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(server_host, port) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(sender, password)
            s.sendmail(sender, recipients + cc_clean, msg.as_string())
        logger.info("Email sent: %r → %s%s",
                    subject, recipients, f" (cc {cc_clean})" if cc_clean else "")
        return True
    except Exception:
        logger.exception("Failed to send email %r", subject)
        return False


# ────────────────────────────────────────────────────────────────────
# Legacy entry point — unchanged signature
# ────────────────────────────────────────────────────────────────────

def send_pipeline_error_email(
    to_email: str,
    pipeline_name: str,
    connector_name: str,
    error_message: str,
) -> None:
    subject = f"Pipeline Failure Alert - {pipeline_name}"
    body = (
        "Dear User,\n\n"
        "We detected a failure during one of your pipeline executions.\n\n"
        "Pipeline Details:\n"
        "---------------------------------\n"
        f"Pipeline Name : {pipeline_name}\n"
        f"Connector Name: {connector_name}\n"
        "Status         : FAILED\n\n"
        "Error Details:\n"
        "---------------------------------\n"
        f"{error_message}\n\n"
        "Regards,\n"
        "AI Pipeline Monitoring System\n"
    )
    _send([to_email], subject, body)


# ────────────────────────────────────────────────────────────────────
# Recipient selection
# ────────────────────────────────────────────────────────────────────

def pick_initial_recipient(db: Session) -> User | None:
    """
    First active DataOps Engineer (by user id). Falls back to any admin
    if no DataOps Engineer exists, so a fresh install doesn't go silent.
    """
    user = (
        db.query(User)
          .filter(User.is_active == True,                              # noqa: E712
                  User.role     == UserRole.dataops_engineer)
          .order_by(User.id.asc())
          .first()
    )
    if user:
        return user

    return (
        db.query(User)
          .filter(User.is_active == True, User.is_admin == True)      # noqa: E712
          .order_by(User.id.asc())
          .first()
    )


def pick_escalation_recipients(
    db: Session,
    exclude_user_id: int | None,
    primary_role: str | None = None,
) -> list[User]:
    """Escalate UP the seniority ladder.

    The role hierarchy (lowest → highest) is:
        DataOps Engineer  →  Data Engineer  →  Data Platform Lead  →  Risk Officer

    Given the role of the originally-notified user, we mail every active user
    whose role sits STRICTLY HIGHER on the ladder than `primary_role`.
    Business Data Consumer is always skipped (they shouldn't be paged for
    production failures).

    If `primary_role` is missing or unknown we fall back to "everyone above
    DataOps Engineer" — i.e. the previous behaviour minus juniors.
    """
    # Lowest index = least senior; rightmost = most senior.
    hierarchy = [
        UserRole.dataops_engineer,
        UserRole.data_engineer,
        UserRole.data_platform_lead,
        UserRole.risk_officer,
    ]
    role_to_index = {r.value: i for i, r in enumerate(hierarchy)}

    base_index = role_to_index.get(primary_role or "", -1)

    q = db.query(User).filter(User.is_active == True)                  # noqa: E712
    if exclude_user_id is not None:
        q = q.filter(User.id != exclude_user_id)

    out: list[User] = []
    for u in q.all():
        if not u.email:
            continue
        if u.role == UserRole.business_data_consumer:
            continue

        role_str = u.role.value if hasattr(u.role, "value") else str(u.role)
        idx      = role_to_index.get(role_str, -1)

        # Strictly higher than the primary recipient's role
        if idx > base_index:
            out.append(u)

    # Stable sort by seniority so the email recipient list reads naturally
    out.sort(key=lambda u: role_to_index.get(
        u.role.value if hasattr(u.role, "value") else str(u.role), 0
    ))
    return out


# ────────────────────────────────────────────────────────────────────
# Body builders
# ────────────────────────────────────────────────────────────────────

def _incident_url(incident_id: int | str) -> str:
    return f"{_frontend_base()}/app/incidents/{incident_id}"


def _trim(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _build_incident_text(
    incident: dict[str, Any],
    llm: dict[str, Any] | None,
    *,
    is_escalation: bool,
    primary_notified: str | None = None,
) -> tuple[str, str]:
    """Returns (subject, plain-text body)."""
    inc_id    = incident.get("id", "?")
    pipe_name = incident.get("pipeline_name") or "(unknown pipeline)"
    risk      = incident.get("risk_tier") or "Medium"
    status    = incident.get("status") or "Detected"
    detected  = incident.get("detected_at") or ""
    err_log   = _trim(incident.get("error_log"), 2000)
    failed_node = incident.get("failed_node") or "—"

    llm = llm or {}
    summary       = _trim(llm.get("summary"),       400)
    root_cause    = _trim(llm.get("root_cause"),   1200)
    suggested_fix = _trim(llm.get("suggested_fix"),2500)
    confidence    = llm.get("confidence")
    used_context  = llm.get("used_context")

    if is_escalation:
        subject = f"[ESCALATION] Unactioned incident #{inc_id} — {pipe_name}"
    else:
        subject = f"[Incident #{inc_id}] {risk} risk — {pipe_name}"

    header_lines = []
    if is_escalation:
        header_lines.append(
            f"This incident was first notified to {primary_notified or 'the on-call DataOps Engineer'} "
            f"and has not been actioned within the SLA window.\n"
        )

    body = (
        "\n".join(header_lines) +
        "Incident Details\n"
        "---------------------------------\n"
        f"ID            : {inc_id}\n"
        f"Pipeline      : {pipe_name}\n"
        f"Failed Node   : {failed_node}\n"
        f"Risk Tier     : {risk}\n"
        f"Status        : {status}\n"
        f"Detected At   : {detected}\n\n"
        "Mistral Diagnosis\n"
        "---------------------------------\n"
        f"Summary       : {summary or '(not yet produced)'}\n"
        f"Root Cause    : {root_cause or '(none)'}\n"
        f"Confidence    : {confidence if confidence is not None else 'n/a'}\n"
        f"Used Context  : {used_context if used_context is not None else 'n/a'}\n\n"
        "Suggested Fix\n"
        "---------------------------------\n"
        f"{suggested_fix or '(none)'}\n\n"
        "Error Log (truncated)\n"
        "---------------------------------\n"
        f"{err_log or '(no log captured)'}\n\n"
        "Open the incident\n"
        "---------------------------------\n"
        f"{_incident_url(inc_id)}\n\n"
        "Regards,\n"
        "AI Pipeline Monitoring System\n"
    )
    return subject, body


def _build_incident_html(
    incident: dict[str, Any],
    llm: dict[str, Any] | None,
    *,
    is_escalation: bool,
    primary_notified: str | None = None,
) -> str:
    inc_id    = incident.get("id", "?")
    pipe_name = incident.get("pipeline_name") or "(unknown pipeline)"
    risk      = incident.get("risk_tier") or "Medium"
    status    = incident.get("status") or "Detected"
    detected  = incident.get("detected_at") or ""
    err_log   = _trim(incident.get("error_log"), 2000)
    failed_node = incident.get("failed_node") or "—"

    llm = llm or {}
    summary       = _trim(llm.get("summary"),       400)
    root_cause    = _trim(llm.get("root_cause"),   1200)
    suggested_fix = _trim(llm.get("suggested_fix"),2500)
    confidence    = llm.get("confidence")
    used_context  = llm.get("used_context")

    banner_bg, banner_text = (
        ("#fef2f2", "🚨 Escalation — incident not actioned in SLA window")
        if is_escalation else
        ("#eff6ff", "🔔 New incident detected — your action is requested")
    )

    primary_note = (
        f"<p style='color:#6b7280;font-size:13px;margin:0 0 12px 0;'>Originally notified: "
        f"<b>{primary_notified}</b></p>"
        if is_escalation and primary_notified else ""
    )

    risk_badge_bg = {"High": "#fee2e2", "Medium": "#fef3c7", "Low": "#dcfce7"}.get(risk, "#e5e7eb")
    risk_badge_fg = {"High": "#b91c1c", "Medium": "#92400e", "Low": "#166534"}.get(risk, "#374151")

    return f"""
<html>
  <body style="font-family: -apple-system, Segoe UI, Roboto, sans-serif; color:#111827; background:#f9fafb; padding:20px;">
    <div style="max-width:680px; margin:0 auto; background:white; border:1px solid #e5e7eb; border-radius:12px; overflow:hidden;">

      <div style="background:{banner_bg}; padding:14px 20px; border-bottom:1px solid #e5e7eb;">
        <div style="font-size:13px; font-weight:700; color:#111827;">{banner_text}</div>
      </div>

      <div style="padding:24px;">
        {primary_note}
        <h2 style="margin:0 0 4px 0; font-size:20px;">{pipe_name}</h2>
        <div style="color:#6b7280; font-size:12px; margin-bottom:18px;">
          Incident <b>#{inc_id}</b> &middot; {status} &middot; detected {detected}
        </div>

        <table style="width:100%; border-collapse:collapse; font-size:13px; margin-bottom:18px;">
          <tr>
            <td style="padding:6px 0; color:#6b7280; width:140px;">Risk tier</td>
            <td><span style="background:{risk_badge_bg}; color:{risk_badge_fg}; font-weight:700; padding:2px 8px; border-radius:6px; font-size:11px;">{risk}</span></td>
          </tr>
          <tr><td style="padding:6px 0; color:#6b7280;">Failed node</td><td>{failed_node}</td></tr>
          <tr><td style="padding:6px 0; color:#6b7280;">Confidence</td><td>{confidence if confidence is not None else 'n/a'}</td></tr>
          <tr><td style="padding:6px 0; color:#6b7280;">RAG context used</td><td>{used_context if used_context is not None else 'n/a'}</td></tr>
        </table>

        <h3 style="font-size:14px; margin:20px 0 6px 0;">Summary</h3>
        <div style="background:#f9fafb; border:1px solid #f3f4f6; border-radius:8px; padding:10px 12px; font-size:13px;">{summary or '<i>not yet produced</i>'}</div>

        <h3 style="font-size:14px; margin:20px 0 6px 0;">Root cause</h3>
        <div style="background:#f9fafb; border:1px solid #f3f4f6; border-radius:8px; padding:10px 12px; font-size:13px;">{root_cause or '<i>none</i>'}</div>

        <h3 style="font-size:14px; margin:20px 0 6px 0;">Suggested fix</h3>
        <div style="background:#f0fdf4; border:1px solid #dcfce7; border-radius:8px; padding:10px 12px; font-size:13px; white-space:pre-wrap;">{suggested_fix or '<i>none</i>'}</div>

        <h3 style="font-size:14px; margin:20px 0 6px 0;">Error log (truncated)</h3>
        <pre style="background:#111827; color:#e5e7eb; padding:12px; border-radius:8px; font-size:12px; overflow:auto; white-space:pre-wrap;">{err_log or '(no log captured)'}</pre>

        <div style="margin-top:24px; text-align:center;">
          <a href="{_incident_url(inc_id)}" style="background:#111827; color:white; text-decoration:none; padding:10px 18px; border-radius:8px; font-weight:700; font-size:13px;">Open incident</a>
        </div>
      </div>

      <div style="padding:14px 20px; border-top:1px solid #e5e7eb; background:#f9fafb; font-size:11px; color:#6b7280;">
        AI Pipeline Monitoring System &middot; automated alert
      </div>
    </div>
  </body>
</html>
"""


# ────────────────────────────────────────────────────────────────────
# Public API used by incident_service
# ────────────────────────────────────────────────────────────────────

def send_incident_alert(
    to_email: str,
    incident: dict[str, Any],
    llm: dict[str, Any] | None,
) -> bool:
    subject, body = _build_incident_text(incident, llm, is_escalation=False)
    html = _build_incident_html(incident, llm, is_escalation=False)
    return _send([to_email], subject, body, body_html=html)


def send_incident_escalation(
    to_emails: list[str],
    incident: dict[str, Any],
    llm: dict[str, Any] | None,
    cc_email: str | None = None,
    primary_notified: str | None = None,
) -> bool:
    subject, body = _build_incident_text(
        incident, llm, is_escalation=True, primary_notified=primary_notified,
    )
    html = _build_incident_html(
        incident, llm, is_escalation=True, primary_notified=primary_notified,
    )
    return _send(to_emails, subject, body, body_html=html, cc_emails=[cc_email] if cc_email else None)


# ────────────────────────────────────────────────────────────────────
# De-dup guard so a misfired scheduler can't escalate the same
# incident twice in one process lifetime.
# ────────────────────────────────────────────────────────────────────

_escalated_ids: set[int | str] = set()
_escalated_lock = threading.Lock()


def was_escalated(incident_id: int | str) -> bool:
    with _escalated_lock:
        return incident_id in _escalated_ids


def mark_escalated(incident_id: int | str) -> None:
    with _escalated_lock:
        _escalated_ids.add(incident_id)
