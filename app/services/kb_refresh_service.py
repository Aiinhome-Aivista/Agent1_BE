"""
Knowledge-base refresh service.

Two jobs:

  enrich_from_approved_fix(...)  — called immediately when a human approves OR
                                   modifies a fix. Sharpens the SolutionPattern
                                   with the human-curated fix, mirrors it into
                                   the vector store, and writes it into the
                                   knowledge graph (incident → pattern → fix
                                   edges). This is the "knowledge graph will be
                                   enhanced/updated by old error, history,
                                   runbook and the human-approved fix" ask.

  run_daily_refresh(...)         — the scheduled batch. Walks recent history
                                   (resolved incidents), every KNOWN/accepted
                                   solution pattern, and active runbooks, and
                                   re-enriches the vector store + graph so the
                                   whole KB stays consolidated. Driven by the
                                   KBSettings.daily_refresh_time row in SQL.

Everything is best-effort: if Arango/Chroma is down, each piece is skipped and
the rest still runs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models import Incident, IncidentStatus
from app.models.runbook import Runbook, RunbookStatus
from app.models.solution_models import SolutionFix, SolutionPattern, SolutionStatus
from app.models.kb_settings import KBSettings, get_or_create_settings
from app.services.rag_service import rag_service
from app.services.graph_enrichment_service import record_incident_outcome

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Immediate enrichment on human approval / modification
# ──────────────────────────────────────────────────────────────────────

def enrich_from_approved_fix(
    db: Session,
    *,
    incident: Incident,
    pattern: SolutionPattern | None,
    curated_root_cause: str | None = None,
    curated_fix_steps: list[str] | None = None,
    approved_by: str | None = None,
) -> dict[str, Any]:
    """Fold a human-approved (and possibly human-modified) fix back into the
    knowledge base + graph so future diagnoses are sharper."""
    out: dict[str, Any] = {"vector": False, "graph": False, "pattern_updated": False}

    # 1. Sharpen the pattern with the human-curated content (takes precedence).
    if pattern is not None:
        if curated_root_cause:
            pattern.root_cause = curated_root_cause
        if curated_fix_steps:
            pattern.fix_steps = curated_fix_steps
            pattern.fix_summary = curated_fix_steps[0] if curated_fix_steps else pattern.fix_summary
        # A human signed off → this is now authoritative knowledge.
        pattern.status = SolutionStatus.KNOWN
        db.commit()
        db.refresh(pattern)
        out["pattern_updated"] = True

    # 2. Mirror into the vector KB.
    try:
        rag_service.store_incident(
            incident_id=f"approved-{incident.id}",
            pipeline_name=incident.pipeline_name,
            error_log=incident.error_log or "",
            root_cause=curated_root_cause or (pattern.root_cause if pattern else incident.root_cause),
            suggested_fix="\n".join(
                curated_fix_steps
                or (pattern.fix_steps if pattern else None)
                or incident.remediation_plan
                or []
            ),
            confidence=pattern.confidence if pattern else incident.confidence_score,
            risk_tier="Low",
        )
        out["vector"] = True
    except Exception:
        logger.debug("enrich vector mirror failed", exc_info=True)

    # 3. Write into the knowledge graph (incident → pattern → fixes).
    try:
        sigs = [pattern.error_type] if (pattern and pattern.error_type) else []
        record_incident_outcome(
            incident_id=incident.id,
            pipeline_name=incident.pipeline_name,
            summary=(curated_root_cause or incident.agent_thought or incident.root_cause or ""),
            confidence=(pattern.confidence if pattern else incident.confidence_score) or 0.8,
            matched_pattern_signatures=sigs,
            applied_fixes=(curated_fix_steps or (pattern.fix_steps if pattern else None)
                           or incident.remediation_plan or []),
            success=True,
        )
        out["graph"] = True
    except Exception:
        logger.debug("enrich graph write failed", exc_info=True)

    logger.info("KB enrich from approved fix (incident #%s by %s): %s",
                incident.id, approved_by, out)
    return out


# ──────────────────────────────────────────────────────────────────────
# Scheduled daily refresh
# ──────────────────────────────────────────────────────────────────────

def should_run_now(row: KBSettings, now: datetime | None = None) -> bool:
    now = now or datetime.utcnow()
    if not row or not row.daily_refresh_enabled:
        return False
    if (row.daily_refresh_time or "")[:5] != now.strftime("%H:%M"):
        return False
    # Only once per calendar day.
    if row.last_run_date == now.strftime("%Y-%m-%d"):
        return False
    return True


def run_daily_refresh(db: Session, *, lookback_days: int = 30) -> dict[str, Any]:
    """Consolidate the KB from history + patterns + runbooks. Idempotent."""
    started = datetime.utcnow()
    cutoff = started - timedelta(days=lookback_days)

    counts = {
        "incidents_replayed": 0,
        "patterns_mirrored": 0,
        "runbooks_seen": 0,
        "graph_writes": 0,
        "errors": 0,
    }

    # 1. Replay recent resolved incidents (history) into vectors + graph.
    try:
        resolved = (
            db.query(Incident)
            .filter(
                Incident.status == IncidentStatus.REMEDIATED,
                Incident.detected_at >= cutoff,
            )
            .all()
        )
        for inc in resolved:
            try:
                rag_service.store_incident(
                    incident_id=f"daily-{inc.id}",
                    pipeline_name=inc.pipeline_name,
                    error_log=inc.error_log or "",
                    root_cause=inc.root_cause,
                    suggested_fix="\n".join(inc.remediation_plan or []),
                    confidence=inc.confidence_score,
                    risk_tier=inc.risk_tier,
                )
                record_incident_outcome(
                    incident_id=inc.id,
                    pipeline_name=inc.pipeline_name,
                    summary=inc.agent_thought or inc.root_cause or "",
                    confidence=inc.confidence_score or 0.7,
                    applied_fixes=inc.remediation_plan or [],
                    success=True,
                )
                counts["incidents_replayed"] += 1
                counts["graph_writes"] += 1
            except Exception:
                counts["errors"] += 1
    except Exception:
        logger.exception("daily refresh: incident replay failed")
        counts["errors"] += 1

    # 2. Re-mirror every known / accepted solution pattern into vectors.
    try:
        patterns = (
            db.query(SolutionPattern)
            .filter(SolutionPattern.status == SolutionStatus.KNOWN)
            .all()
        )
        for p in patterns:
            try:
                fix_text = p.fix_summary or "\n".join(p.fix_steps or [])
                # Prefer a real ingested code fix's explanation if present.
                code_fix = next((f for f in (p.fixes or []) if f.has_code), None)
                if code_fix and code_fix.explanation:
                    fix_text = f"{fix_text}\n\nCode change: {code_fix.explanation}"
                rag_service.store_incident(
                    incident_id=f"kb-daily-{p.id}",
                    pipeline_name=p.title,
                    error_log=p.error_excerpt or p.title,
                    root_cause=p.root_cause,
                    suggested_fix=fix_text,
                    confidence=p.confidence,
                    risk_tier="Low" if (p.confidence or 0) >= 0.7 else "Medium",
                )
                counts["patterns_mirrored"] += 1
            except Exception:
                counts["errors"] += 1
    except Exception:
        logger.exception("daily refresh: pattern mirror failed")
        counts["errors"] += 1

    # 3. Note active runbooks (already enriched on upload; counted for the UI).
    try:
        counts["runbooks_seen"] = (
            db.query(Runbook).filter(Runbook.status == RunbookStatus.ACTIVE).count()
        )
    except Exception:
        pass

    # 4. Stamp the settings row.
    row = get_or_create_settings(db)
    row.last_run_at = started
    row.last_run_date = started.strftime("%Y-%m-%d")
    summary = {
        "ran_at": started.isoformat(),
        "duration_ms": int((datetime.utcnow() - started).total_seconds() * 1000),
        **counts,
    }
    row.last_run_summary = json.dumps(summary)
    db.commit()

    logger.info("KB daily refresh complete: %s", summary)
    return summary


def maybe_run_scheduled_refresh() -> None:
    """Entry point for the APScheduler minute-tick. Opens its own session."""
    from app.core.database import SessionLocal  # noqa: PLC0415
    db = SessionLocal()
    try:
        row = get_or_create_settings(db)
        if should_run_now(row):
            logger.info("KB daily refresh time reached (%s) — running", row.daily_refresh_time)
            run_daily_refresh(db)
    except Exception:
        logger.exception("scheduled KB refresh check failed")
    finally:
        db.close()
