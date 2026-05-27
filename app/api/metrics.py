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
