"""
Telemetry Investigator Service.

Responsibilities:
- Assess whether stored pipeline logs have sufficient error detail for diagnosis.
- If logs are incomplete (only generic wrapper error), re-fetch from the source connector
  using progressive retrieval with retry, and persist the refreshed logs to DB.
- Return an investigation_timeline dict for the diagnosis normalizer to surface in the UI.

The primary use case is re-analyze triggered by the user when the initial sync captured
only "Workload failed, see run output for details." before Databricks fully flushed the
task-level error output.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.models import PipelineLog, PipelineRun, LogLevel

logger = logging.getLogger(__name__)

# ─── Generic wrapper messages that indicate incomplete telemetry ──────────────

_GENERIC_WRAPPERS = frozenset({
    "workload failed, see run output for details.",
    "workload failed",
    "workload failed.",
})


def _is_wrapper(text: str | None) -> bool:
    if not text:
        return True
    t = str(text).lower().strip().rstrip(".")
    return t in _GENERIC_WRAPPERS or (len(t) < 80 and "workload failed" in t and "see run output" in t)


# ─── Telemetry completeness assessment ───────────────────────────────────────

def assess_completeness(
    stored_logs: list[dict[str, Any]],
    error_message: str | None,
) -> dict[str, Any]:
    """
    Assess whether stored pipeline logs have actionable error detail.

    Returns a dict:
        level            "COMPLETE" | "PARTIAL" | "INSUFFICIENT"
        is_generic       True if top-level error is only a wrapper message
        has_error_lines  True if there is at least one non-wrapper ERROR log
        error_line_count Count of non-wrapper ERROR/CRITICAL lines
        detail_chars     Total characters in non-wrapper ERROR lines
        reason           Human-readable explanation
    """
    error_lines = [
        l for l in stored_logs
        if l.get("level") in ("ERROR", "CRITICAL")
        and not _is_wrapper(l.get("message"))
        and l.get("source") != "_investigation"
    ]
    detail_chars = sum(len(l.get("message") or "") for l in error_lines)
    has_error = len(error_lines) > 0
    is_generic = _is_wrapper(error_message)

    if has_error and detail_chars >= 100:
        level = "COMPLETE"
        reason = f"Retrieved {len(error_lines)} error log line(s) with {detail_chars} chars of detail."
    elif has_error and detail_chars > 0:
        level = "PARTIAL"
        reason = f"Retrieved {len(error_lines)} error log line(s) but content is brief ({detail_chars} chars)."
    elif is_generic:
        level = "INSUFFICIENT"
        reason = (
            "Only the generic Databricks wrapper error was captured. "
            "Detailed task-level output was not yet available at initial sync time."
        )
    else:
        level = "PARTIAL"
        reason = "No ERROR-level log lines were captured; top-level error message is available."

    return {
        "level": level,
        "is_generic": is_generic,
        "has_error_lines": has_error,
        "error_line_count": len(error_lines),
        "detail_chars": detail_chars,
        "reason": reason,
    }


# ─── Progressive log re-fetch ─────────────────────────────────────────────────

async def progressive_fetch_and_persist(
    db: Session,
    run: PipelineRun,
    connector_obj: Any,        # the DatabricksConnector (or any BaseConnector) instance
) -> dict[str, Any]:
    """
    Attempt to re-fetch logs from the remote connector for a given run.

    Returns:
        {
          "refetched": bool,
          "new_log_count": int,
          "completeness": dict,
          "timeline": list[dict],
        }
    """
    timeline: list[dict[str, str]] = []

    def _step(label: str, status: str, detail: str = "") -> dict:
        entry = {
            "label": label,
            "status": status,
            "detail": detail,
            "ts": datetime.utcnow().isoformat(),
        }
        timeline.append(entry)
        return entry

    pipe = run.pipeline

    _step("Detect run failure", "ok", f"Run {run.external_run_id} failed on pipeline {pipe.name}")

    try:
        _step("Re-fetch task-level logs from connector", "pending", "Calling Databricks Jobs 2.1 API with retry backoff")
        new_logs = await asyncio.to_thread(
            connector_obj.get_logs,
            pipe.external_id,
            run.external_run_id,
        )
        timeline[-1]["status"] = "ok"
        timeline[-1]["detail"] = f"Fetched {len(new_logs)} log entries"
    except Exception as exc:
        timeline[-1]["status"] = "fail"
        timeline[-1]["detail"] = f"Connector error: {str(exc)[:200]}"
        logger.warning("progressive_fetch: get_logs failed for run %s: %s", run.external_run_id, exc)
        new_logs = []

    if not new_logs:
        _step("Persist refreshed logs", "skipped", "No new logs returned — keeping existing logs")
        stored = (
            db.query(PipelineLog)
            .filter(PipelineLog.run_id == run.id)
            .order_by(PipelineLog.timestamp.asc())
            .all()
        )
        log_dicts = [
            {"timestamp": l.timestamp.isoformat(), "level": l.level.value, "source": l.source, "message": l.message}
            for l in stored
        ]
        completeness = assess_completeness(log_dicts, run.error_message)
        return {
            "refetched": False,
            "new_log_count": len(stored),
            "completeness": completeness,
            "timeline": timeline,
        }

    try:
        db.query(PipelineLog).filter(PipelineLog.run_id == run.id).delete()
        log_dicts: list[dict] = []
        for nl in new_logs:
            plog = PipelineLog(
                run_id=run.id,
                timestamp=nl.timestamp or datetime.utcnow(),
                level=LogLevel(nl.level) if nl.level in LogLevel.__members__ else LogLevel.INFO,
                source=nl.source,
                message=nl.message,
            )
            db.add(plog)
            log_dicts.append({
                "timestamp": plog.timestamp.isoformat(),
                "level": plog.level.value,
                "source": plog.source,
                "message": plog.message,
            })
        db.commit()
        _step("Persist refreshed logs", "ok", f"Saved {len(new_logs)} log entries to database")
    except Exception as exc:
        db.rollback()
        logger.error("progressive_fetch: failed to persist logs for run %s: %s", run.id, exc)
        _step("Persist refreshed logs", "fail", f"DB error: {str(exc)[:200]}")
        log_dicts = []

    completeness = assess_completeness(log_dicts, run.error_message)
    _step(
        "Telemetry completeness check",
        "ok" if completeness["level"] == "COMPLETE" else "warning",
        f"Level: {completeness['level']} — {completeness['reason']}",
    )

    return {
        "refetched": True,
        "new_log_count": len(log_dicts),
        "completeness": completeness,
        "timeline": timeline,
    }


def build_investigation_timeline(
    completeness: dict[str, Any] | None = None,
    rag_found_incidents: int = 0,
    rag_found_runbooks: int = 0,
    llm_status: str = "success",
    refetched: bool = False,
) -> list[dict[str, str]]:
    """
    Build a structured investigation timeline for the UI diagnosis panel.

    Each entry has: label, status ("ok"|"fail"|"skipped"|"warning"), detail, icon
    """
    completeness = completeness or {}
    level = completeness.get("level", "UNKNOWN")
    reason = completeness.get("reason", "")

    def _icon(status: str) -> str:
        return {"ok": "check", "fail": "x", "skipped": "skip", "warning": "warn"}.get(status, "skip")

    timeline = []

    # Step 1: Log retrieval
    if refetched:
        t_label = "Re-fetched task-level logs from connector"
        t_detail = f"Refreshed from Databricks API (retried). {reason}"
    else:
        t_label = "Task-level log retrieval"
        t_detail = reason or "Logs retrieved from database."
    t_status = "ok" if level in ("COMPLETE", "PARTIAL") else "warning"
    timeline.append({"label": t_label, "status": t_status, "detail": t_detail, "icon": _icon(t_status)})

    # Step 2: Error extraction
    ec = completeness.get("error_line_count", 0)
    ex_status = "ok" if ec > 0 else "warning"
    ex_detail = (
        f"Extracted {ec} error log line(s) with {completeness.get('detail_chars', 0)} chars of detail."
        if ec > 0
        else "No specific error lines found — only generic wrapper message."
    )
    timeline.append({"label": "Error extraction & fact parsing", "status": ex_status, "detail": ex_detail, "icon": _icon(ex_status)})

    # Step 3: KB search
    rag_ok = rag_found_incidents > 0 or rag_found_runbooks > 0
    rag_status = "ok" if rag_ok else "skipped"
    rag_detail = (
        f"Found {rag_found_incidents} similar incident(s) and {rag_found_runbooks} runbook excerpt(s)."
        if rag_ok
        else "No matching KB entries found for this error pattern."
    )
    timeline.append({"label": "Knowledge base search (RAG)", "status": rag_status, "detail": rag_detail, "icon": _icon(rag_status)})

    # Step 4: LLM synthesis
    llm_map = {
        "success": ("ok", "LLM synthesized root cause and remediation plan with available evidence."),
        "partial": ("warning", "LLM response was partial — some fields used fact-locked fallbacks."),
        "parse_failed": ("fail", "LLM response could not be parsed as structured JSON."),
        "failed": ("fail", "LLM service was unavailable or timed out."),
    }
    l_status, l_detail = llm_map.get(llm_status, ("warning", f"LLM status: {llm_status}"))
    timeline.append({"label": "LLM root cause synthesis", "status": l_status, "detail": l_detail, "icon": _icon(l_status)})

    # Step 5: Fact locking
    fl_status = "ok" if level != "INSUFFICIENT" else "warning"
    fl_detail = (
        "Verified facts were locked and protected from LLM modification."
        if fl_status == "ok"
        else "Fact locking applied with limited telemetry — some fields defaulted."
    )
    timeline.append({"label": "Fact locking & normalization", "status": fl_status, "detail": fl_detail, "icon": _icon(fl_status)})

    return timeline
