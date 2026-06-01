"""
Performance & observability metrics endpoints.

  GET /metrics/pipelines      - per-pipeline rollups (24h window by default)
  GET /metrics/pipelines/{id} - drill-down for one pipeline
  GET /metrics/rag            - RAG (vector retrieval) performance
  GET /metrics/llm            - Mistral call latency / success rate
  GET /metrics/system         - one-shot dashboard summary combining all three
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models import User
from app.models.agent_models import Incident
from app.services.metrics_service import metrics_service
from app.services.vector_service import get_vector_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _safe_pipeline_perf(db: Session, hours: int) -> list[dict[str, Any]]:
    """Never raise: an empty list is much friendlier than a 500 on a dashboard."""
    try:
        return metrics_service.pipeline_performance(db, hours=hours)
    except Exception:
        logger.exception("pipeline_performance failed")
        return []


def _safe_rag_stats() -> dict[str, int]:
    try:
        vector_service = get_vector_service()
        return vector_service.stats()
    except Exception:
        logger.exception("vector_service.stats failed")
        return {"incidents": 0, "runbooks": 0}


def _safe_rag_summary() -> dict[str, Any]:
    try:
        return metrics_service.rag_summary()
    except Exception:
        logger.exception("rag_summary failed")
        return {
            "incidents": {"query_count": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0,
                          "hit_rate": 0.0, "avg_top_similarity": 0.0},
            "runbooks":  {"query_count": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0,
                          "hit_rate": 0.0, "avg_top_similarity": 0.0},
        }


def _safe_llm_summary() -> dict[str, Any]:
    try:
        return metrics_service.llm_summary()
    except Exception:
        logger.exception("llm_summary failed")
        return {
            "call_count": 0, "success_rate": 1.0,
            "avg_latency_ms": 0.0, "p95_latency_ms": 0.0,
            "avg_prompt_chars": 0,
        }


@router.get("/pipelines")
def pipelines_performance(
    hours: int = Query(24, ge=1, le=24 * 30),
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
) -> list[dict[str, Any]]:
    return _safe_pipeline_perf(db, hours)


@router.get("/pipelines/{pipeline_id}")
def pipeline_performance_detail(
    pipeline_id: int,
    hours: int = Query(24, ge=1, le=24 * 30),
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
) -> dict[str, Any]:
    rows = _safe_pipeline_perf(db, hours)
    for r in rows:
        if r["pipeline_id"] == pipeline_id:
            return r
    raise HTTPException(404, "Pipeline not found or no runs in window")


@router.get("/rag")
def rag_performance(user: User = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "collections": _safe_rag_stats(),
        "summary":     _safe_rag_summary(),
    }


@router.get("/llm")
def llm_performance(user: User = Depends(get_current_user)) -> dict[str, Any]:
    return _safe_llm_summary()


@router.get("/system")
def system_metrics(
    hours: int = Query(24, ge=1, le=24 * 30),
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
) -> dict[str, Any]:
    """One-shot summary the dashboard can paint in a single call."""
    pipelines = _safe_pipeline_perf(db, hours)

    runs_total       = sum(p["runs"] for p in pipelines)
    runs_failed      = sum(p["failed"] for p in pipelines)
    runs_succeeded   = sum(p["succeeded"] for p in pipelines)
    overall_success  = (runs_succeeded / (runs_succeeded + runs_failed) * 100.0
                        if (runs_succeeded + runs_failed) else 100.0)

    return {
        "window_hours": hours,
        "pipelines": {
            "count":            len(pipelines),
            "runs_total":       runs_total,
            "runs_succeeded":   runs_succeeded,
            "runs_failed":      runs_failed,
            "success_rate_pct": round(overall_success, 2),
            "top_5_busiest":    pipelines[:5],
        },
        "rag":  {
            "collections": _safe_rag_stats(),
            "summary":     _safe_rag_summary(),
        },
        "llm":  _safe_llm_summary(),
    }


@router.get("/summary")
def get_metrics_summary(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict[str, Any]:
    total_tickets = db.query(Incident).count()
    ai_resolved = db.query(Incident).filter(Incident.status == "Remediated").count()
    human_resolved = db.query(Incident).filter(Incident.status.in_(["Failed", "Escalated"])).count()
    tickets_solved = ai_resolved + human_resolved
    open_incidents = total_tickets - tickets_solved
    jira_tickets_created = db.query(Incident).filter(Incident.jira_ticket_key.isnot(None)).count()
    ai_resolution_pct = (ai_resolved / total_tickets * 100.0) if total_tickets > 0 else 0.0

    # Compute MTTR from resolved incidents (minutes between detected & resolved)
    resolved = (
        db.query(Incident)
        .filter(Incident.resolved_at.isnot(None), Incident.detected_at.isnot(None))
        .all()
    )
    durations = [
        (inc.resolved_at - inc.detected_at).total_seconds() / 60.0
        for inc in resolved
        if inc.resolved_at and inc.detected_at and inc.resolved_at >= inc.detected_at
    ]
    mttr_avg_minutes = round(sum(durations) / len(durations), 1) if durations else 0.0

    # Knowledge-base stats (learning loop)
    try:
        from app.services.solution_kb_service import solution_kb_service   # noqa: PLC0415
        kb = solution_kb_service.stats(db)
    except Exception:
        kb = {"patterns_total": 0, "patterns_known": 0,
              "patterns_auto_fixable": 0, "human_prs_ingested": 0}

    return {
        "total_tickets": total_tickets,
        "tickets_solved": tickets_solved,
        "ai_resolved": ai_resolved,
        "human_resolved": human_resolved,
        "ai_resolution_pct": ai_resolution_pct,
        "mttr_avg_minutes": mttr_avg_minutes,
        "open_incidents": open_incidents,
        "jira_tickets_created": jira_tickets_created,
        "kb": kb,
    }


@router.get("/health")
def get_system_health(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    # Fetch all incidents from the last 14 days to ensure recent incidents are caught
    window_days = 14
    start_time = datetime.utcnow() - timedelta(days=window_days)
    recent_incidents = db.query(Incident).filter(Incident.detected_at >= start_time).all()
    
    # Group by day string (e.g., "05/18")
    daily_stats = {}
    for i in range(window_days - 1, -1, -1):
        day_date = datetime.utcnow() - timedelta(days=i)
        day_str = day_date.strftime("%m/%d") # e.g. "05/18"
        daily_stats[day_str] = {
            "time": day_str,
            "tickets_raised": 0,
            "tickets_ai_solved": 0,
            "tickets_human_solved": 0,
            "mttr_minutes": 0.0,
            "success_rate": 0.0,
            "_total_mttr": 0.0,
            "_resolved_count": 0
        }
        
    for inc in recent_incidents:
        if not inc.detected_at:
            continue
        day_str = inc.detected_at.strftime("%m/%d")
        if day_str not in daily_stats:
            continue
            
        stats = daily_stats[day_str]
        stats["tickets_raised"] += 1
        
        if inc.status == "Remediated":
            stats["tickets_ai_solved"] += 1
            if inc.resolved_at:
                mttr = (inc.resolved_at - inc.detected_at).total_seconds() / 60.0
                stats["_total_mttr"] += mttr
                stats["_resolved_count"] += 1
        elif inc.status in ["Failed", "Escalated"]:
            stats["tickets_human_solved"] += 1
            
    # Calculate final averages and percentages
    result = []
    for day_str, stats in daily_stats.items():
        if stats["tickets_raised"] > 0:
            stats["success_rate"] = (stats["tickets_ai_solved"] / stats["tickets_raised"]) * 100.0
        if stats["_resolved_count"] > 0:
            stats["mttr_minutes"] = stats["_total_mttr"] / stats["_resolved_count"]
            
        # Clean up temporary keys
        del stats["_total_mttr"]
        del stats["_resolved_count"]
        result.append(stats)
        
    return result
