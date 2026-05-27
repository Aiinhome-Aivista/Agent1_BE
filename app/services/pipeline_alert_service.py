"""
pipeline_alert_service.py

Background scheduler that:
  1. Polls every 60 seconds for pipelines whose last_run_status = FAILED.
  2. For each newly-failed pipeline, starts an alert_cycle():
       a. Send initial email to ALL DataOps Engineers  (delay=0, from config table)
       b. Wait `escalation_delay` minutes              (from config table)
       c. Check whether a NEW run appeared since step (a)
          → If yes  : new run found, stop this cycle (pipeline is retrying itself)
          → If no   : send escalation email to Data Engineer + Data Platform Lead
       d. Wait 5 minutes (RECHECK_MINUTES, also configurable).
       e. If last_run_status is still FAILED → go back to step (a) for next cycle.
          Otherwise stop.

All timings are loaded from the alert_config table at the START of each cycle,
so updating a row in the DB takes effect on the next cycle with no restart needed.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.pipeline import Pipeline, PipelineRun, RunStatus
from app.models.user import User, UserRole
from app.models.alert_models import AlertConfig, EmailLog
from app.services import email_service
from app.services.mistral_service import mistral_service

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 60      # how often the scheduler scans for failed pipelines
RECHECK_MINUTES       = 5       # how long to wait before re-checking if still failed
# (overridden per-cycle by the config table — this is just the startup default)

# tracks which pipelines are currently inside an alert_cycle so we don't
# spawn a second coroutine for the same pipeline while one is already running.
_active_cycles: set[int] = set()


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_config(db: Session) -> dict[str, int]:
    """
    Load delay_minutes for every role from the alert_config table.
    Returns e.g. {"DataOps Engineer": 0, "Data Engineer": 2, "Data Platform Lead": 2}
    """
    rows = db.query(AlertConfig).all()
    return {row.role: row.delay_minutes for row in rows}


def _get_dataops_engineers(db: Session) -> list[User]:
    return (
        db.query(User)
          .filter(User.is_active == True,
                  User.role == UserRole.dataops_engineer)
          .order_by(User.id)
          .all()
    )


def _get_escalation_recipients(db: Session) -> list[User]:
    """Data Engineers + Data Platform Leads."""
    return (
        db.query(User)
          .filter(User.is_active == True,
                  User.role.in_([UserRole.data_engineer,
                                 UserRole.data_platform_lead]))
          .order_by(User.role, User.id)
          .all()
    )


def _latest_run_started_at(db, pipeline_id):
    """Return the started_at of the most recent PipelineRun for this pipeline."""
    run = (
        db.query(PipelineRun)
          .filter(PipelineRun.pipeline_id == pipeline_id)
          .order_by(PipelineRun.started_at.desc())
          .first()
    )
    return run.started_at if run else None


def _latest_run_id(db, pipeline_id):
    """Return the id of the most recent PipelineRun for this pipeline."""
    run = (
        db.query(PipelineRun)
          .filter(PipelineRun.pipeline_id == pipeline_id)
          .order_by(PipelineRun.started_at.desc())
          .first()
    )
    return run.id if run else None


def _has_new_run_since(db, pipeline_id, since_started_at):
    """
    Return True if a run with started_at > since_started_at exists.
    Comparing by timestamp avoids false positives when runs are inserted
    out of order and an older run has a higher id than the failed one.
    """
    if since_started_at is None:
        return False
    return (
        db.query(PipelineRun)
          .filter(PipelineRun.pipeline_id == pipeline_id,
                  PipelineRun.started_at > since_started_at)
          .count()
    ) > 0


def _log_email(
    db: Session,
    *,
    pipeline_id: int,
    pipeline_name: str,
    run_id: int | None,
    email_type: str,
    recipient_email: str,
    recipient_role: str,
    cycle_number: int,
    initial_sent_at: datetime | None = None,
    escalation_sent_at: datetime | None = None,
    new_run_found: bool | None = None,
    status: str = "sent",
    notes: str | None = None,
) -> None:
    """Write one row to email_log. Never raises — logging failure must not stop the cycle."""
    try:
        entry = EmailLog(
            pipeline_id        = pipeline_id,
            pipeline_name      = pipeline_name,
            run_id             = run_id,
            email_type         = email_type,
            recipient_email    = recipient_email,
            recipient_role     = recipient_role,
            sent_at            = datetime.utcnow(),
            initial_sent_at    = initial_sent_at,
            escalation_sent_at = escalation_sent_at,
            cycle_number       = cycle_number,
            new_run_found      = new_run_found,
            status             = status,
            notes              = notes,
        )
        db.add(entry)
        db.commit()
    except Exception:
        logger.exception("Failed to write email_log row for pipeline %s", pipeline_id)


def _build_email_payload(pipeline: Pipeline, failed_run_id: int | None) -> dict[str, Any]:
    """Build the minimal dict the existing email_service templates expect."""
    latest_run = pipeline.runs[0] if pipeline.runs else None
    return {
        "id":            failed_run_id or "N/A",
        "pipeline_name": pipeline.name,
        "risk_tier":     "Medium",
        "status":        "Failed",
        "detected_at":   datetime.utcnow().isoformat(),
        "failed_node":   None,
        "error_log":     latest_run.error_message if latest_run else "",
    }


def _run_llm_analysis(
    pipeline: Pipeline, db: Session
) -> tuple[dict[str, Any] | None, str]:
    """
    Call Mistral to produce summary / root_cause / suggested_fix for the
    most-recent failed run.

    Returns
    -------
    (llm_result, full_log_text)
        llm_result    – dict from Mistral, or None on failure
        full_log_text – formatted log lines joined into a single string,
                        or the bare error_message if no PipelineLog rows exist
    """
    try:
        from app.models.pipeline import PipelineLog  # noqa: PLC0415

        latest_run = (
            db.query(PipelineRun)
              .filter(PipelineRun.pipeline_id == pipeline.id)
              .order_by(PipelineRun.started_at.desc())
              .first()
        )
        if not latest_run:
            return None, ""

        logs = (
            db.query(PipelineLog)
              .filter(PipelineLog.run_id == latest_run.id)
              .order_by(PipelineLog.timestamp.asc())
              .all()
        )
        log_dicts = [
            {
                "timestamp": l.timestamp.isoformat() if l.timestamp else "",
                "level":     l.level.value if l.level else "INFO",
                "source":    l.source,
                "message":   l.message,
            }
            for l in logs
        ]

        # Build a human-readable log string for the email body.
        # Fall back to the bare error_message when no structured logs exist.
        if log_dicts:
            full_log_text = "\n".join(
                f"[{d['timestamp']}] [{d['level']}] {d['source'] or ''} {d['message']}"
                for d in log_dicts
            )
        else:
            full_log_text = latest_run.error_message or ""

        connector_type = (
            pipeline.connector.type.value
            if pipeline.connector else "unknown"
        )

        result = mistral_service.analyze_failure(
            pipeline_name  = pipeline.name,
            connector_type = connector_type,
            error_message  = latest_run.error_message,
            logs           = log_dicts,
            metadata       = pipeline.metadata_json or {},
        )
        return result, full_log_text
    except Exception:
        logger.exception("LLM analysis failed for pipeline %s", pipeline.name)
        return None, ""


# ── core alert cycle ──────────────────────────────────────────────────────────

async def alert_cycle(pipeline_id: int) -> None:
    """
    Run the full initial → escalation → recheck loop for ONE pipeline.
    This coroutine runs until the pipeline is no longer FAILED or is
    manually removed from _active_cycles.
    """
    _active_cycles.add(pipeline_id)
    cycle = 0

    try:
        while True:
            cycle += 1
            logger.info("[alert_cycle] Pipeline %s — starting cycle %d", pipeline_id, cycle)

            # ── load timings from DB at the top of every cycle ─────────────
            llm_result: dict[str, Any] | None = None
            db = SessionLocal()
            try:
                config = _get_config(db)
                escalation_delay_minutes = config.get("Data Engineer", 2)
                # DataOps delay is always 0 (immediate) but we read it from
                # config too so an admin could add a small buffer if needed.
                initial_delay_minutes = config.get("DataOps Engineer", 0)

                pipeline: Pipeline | None = (
                    db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
                )
                if not pipeline:
                    logger.warning("[alert_cycle] Pipeline %s not found; stopping.", pipeline_id)
                    return

                # Snapshot the failed run to detect NEW runs later.
                failed_run_id = _latest_run_id(db, pipeline_id)
                failed_run_started_at = _latest_run_started_at(db, pipeline_id)

                # ── STEP A : Initial email to ALL DataOps Engineers ─────────
                if initial_delay_minutes > 0:
                    logger.info("[alert_cycle] Waiting %d min before initial email (cycle %d)",
                                initial_delay_minutes, cycle)
                    await asyncio.sleep(initial_delay_minutes * 60)

                dataops_users = _get_dataops_engineers(db)
                initial_sent_at = datetime.utcnow()

                # ── Run LLM analysis once per cycle ────────────────────────
                payload = _build_email_payload(pipeline, failed_run_id)
                llm_result, full_log_text = _run_llm_analysis(pipeline, db)
                # Use the full structured log text in the email (falls back
                # to error_message when no PipelineLog rows exist).
                if full_log_text:
                    payload["error_log"] = full_log_text
                if llm_result:
                    # Promote risk tier based on LLM confidence
                    confidence = float(llm_result.get("confidence") or 0.0)
                    if confidence >= 0.7:
                        payload["risk_tier"] = "Low"
                    elif confidence >= 0.4:
                        payload["risk_tier"] = "Medium"
                    else:
                        payload["risk_tier"] = "High"

                if not dataops_users:
                    logger.warning("[alert_cycle] No active DataOps Engineers found.")
                    _log_email(
                        db, pipeline_id=pipeline_id, pipeline_name=pipeline.name,
                        run_id=failed_run_id, email_type="initial",
                        recipient_email="N/A", recipient_role="DataOps Engineer",
                        cycle_number=cycle, initial_sent_at=initial_sent_at,
                        status="skipped", notes="No active DataOps Engineers in DB",
                    )
                else:
                    for user in dataops_users:
                        ok = email_service.send_incident_alert(
                            to_email = user.email,
                            incident = payload,
                            llm      = llm_result,
                        )
                        role_str = (user.role.value
                                    if hasattr(user.role, "value") else str(user.role))
                        _log_email(
                            db, pipeline_id=pipeline_id, pipeline_name=pipeline.name,
                            run_id=failed_run_id, email_type="initial",
                            recipient_email=user.email, recipient_role=role_str,
                            cycle_number=cycle, initial_sent_at=initial_sent_at,
                            status="sent" if ok else "failed",
                            notes=None if ok else "SMTP send returned False",
                        )
                        logger.info("[alert_cycle] Initial email → %s (%s) — %s",
                                    user.email, role_str, "OK" if ok else "FAILED")

            finally:
                db.close()

            # ── STEP B : Wait escalation_delay minutes, then check new run ──
            logger.info("[alert_cycle] Waiting %d min before escalation check (cycle %d)",
                        escalation_delay_minutes, cycle)
            await asyncio.sleep(escalation_delay_minutes * 60)

            db = SessionLocal()
            try:
                # Re-read pipeline and config (may have changed during sleep)
                config = _get_config(db)
                escalation_delay_minutes = config.get("Data Engineer", 2)

                pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
                if not pipeline:
                    return

                new_run_found = _has_new_run_since(db, pipeline_id, failed_run_started_at)

                if new_run_found:
                    # A new run appeared — pipeline is retrying; stop this cycle.
                    logger.info("[alert_cycle] New run detected for pipeline %s — "
                                "skipping escalation (cycle %d)", pipeline_id, cycle)
                    # Log a skipped escalation row for visibility
                    _log_email(
                        db, pipeline_id=pipeline_id, pipeline_name=pipeline.name,
                        run_id=failed_run_id, email_type="escalation",
                        recipient_email="N/A", recipient_role="N/A",
                        cycle_number=cycle, initial_sent_at=initial_sent_at,
                        new_run_found=True,
                        status="skipped", notes="New pipeline run detected; escalation suppressed",
                    )
                else:
                    # ── STEP C : Send escalation email to Data Eng + Lead ───
                    escalation_users = _get_escalation_recipients(db)
                    escalation_sent_at = datetime.utcnow()

                    if not escalation_users:
                        logger.warning("[alert_cycle] No escalation recipients found.")
                    else:
                        to_emails = [u.email for u in escalation_users]
                        ok = email_service.send_incident_escalation(
                            to_emails        = to_emails,
                            incident         = payload,
                            llm              = llm_result,
                            cc_email         = None,
                            primary_notified = "DataOps Engineer Team",
                        )
                        for user in escalation_users:
                            role_str = (user.role.value
                                        if hasattr(user.role, "value") else str(user.role))
                            _log_email(
                                db, pipeline_id=pipeline_id, pipeline_name=pipeline.name,
                                run_id=failed_run_id, email_type="escalation",
                                recipient_email=user.email, recipient_role=role_str,
                                cycle_number=cycle,
                                initial_sent_at=initial_sent_at,
                                escalation_sent_at=escalation_sent_at,
                                new_run_found=False,
                                status="sent" if ok else "failed",
                                notes=None if ok else "SMTP send returned False",
                            )
                            logger.info("[alert_cycle] Escalation email → %s (%s) — %s",
                                        user.email, role_str, "OK" if ok else "FAILED")

            finally:
                db.close()

            # ── STEP D : Wait RECHECK_MINUTES, then decide whether to loop ──
            logger.info("[alert_cycle] Waiting %d min before re-checking failure status "
                        "(cycle %d)", RECHECK_MINUTES, cycle)
            await asyncio.sleep(RECHECK_MINUTES * 60)

            db = SessionLocal()
            try:
                pipeline = db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()
                if not pipeline:
                    return

                if pipeline.last_run_status != RunStatus.FAILED:
                    logger.info("[alert_cycle] Pipeline %s is no longer FAILED "
                                "(status=%s) — stopping alert loop.",
                                pipeline_id, pipeline.last_run_status)
                    return
                # Still FAILED → loop back to the top for another cycle.
                logger.info("[alert_cycle] Pipeline %s still FAILED after %d min "
                            "— starting cycle %d.", pipeline_id, RECHECK_MINUTES, cycle + 1)
            finally:
                db.close()

    except asyncio.CancelledError:
        logger.info("[alert_cycle] Cycle cancelled for pipeline %s", pipeline_id)
    except Exception:
        logger.exception("[alert_cycle] Unexpected error for pipeline %s", pipeline_id)
    finally:
        _active_cycles.discard(pipeline_id)


# ── background scanner ────────────────────────────────────────────────────────

async def run_pipeline_alert_scheduler() -> None:
    """
    Long-running coroutine started once on app startup.
    Every POLL_INTERVAL_SECONDS it scans for pipelines with
    last_run_status = FAILED and starts an alert_cycle() for each
    new one (if a cycle isn't already running for that pipeline).
    """
    logger.info("[scheduler] Pipeline alert scheduler started (poll every %ds).",
                POLL_INTERVAL_SECONDS)

    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        db = SessionLocal()
        try:
            failed_pipelines: list[Pipeline] = (
                db.query(Pipeline)
                  .filter(Pipeline.last_run_status == RunStatus.FAILED)
                  .all()
            )

            for pipeline in failed_pipelines:
                if pipeline.id not in _active_cycles:
                    logger.info("[scheduler] Detected FAILED pipeline %s (id=%s) — "
                                "spawning alert_cycle.", pipeline.name, pipeline.id)
                    asyncio.create_task(alert_cycle(pipeline.id))
                else:
                    logger.debug("[scheduler] Pipeline %s already in active alert cycle.",
                                 pipeline.id)
        except Exception:
            logger.exception("[scheduler] Error scanning for failed pipelines.")
        finally:
            db.close()