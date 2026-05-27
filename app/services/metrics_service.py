"""
Lightweight in-process metrics collector.

Tracks three kinds of things:

  1. RAG retrieval metrics  (per query: latency, hits, top-similarity)
  2. LLM call metrics       (Mistral latency, success vs error)
  3. Pipeline performance   (derived on-demand from the SQL tables)

For 1 & 2 we keep a bounded ring buffer in memory – no extra dependency.
For 3 we run aggregate SQL queries when /metrics endpoints are hit; that
keeps the source of truth in the DB and avoids double-bookkeeping.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.models import Pipeline, PipelineRun, RunStatus

logger = logging.getLogger(__name__)


_RING_SIZE = 500   # keep the last 500 samples per metric stream


class _Ring:
    """Tiny thread-safe ring buffer."""

    def __init__(self, size: int = _RING_SIZE) -> None:
        self._dq: deque[dict[str, Any]] = deque(maxlen=size)
        self._lock = threading.Lock()

    def push(self, sample: dict[str, Any]) -> None:
        with self._lock:
            self._dq.append(sample)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._dq)


class MetricsService:
    def __init__(self) -> None:
        self._rag_incidents = _Ring()
        self._rag_runbooks  = _Ring()
        self._llm_calls     = _Ring()

    # ────────────────────────────────────────────────────────────────
    # RAG
    # ────────────────────────────────────────────────────────────────

    def record_rag_query(
        self,
        kind: str,
        latency_ms: float,
        hits: int,
        top_similarity: float,
    ) -> None:
        sample = {
            "ts":             datetime.utcnow().isoformat(),
            "latency_ms":     round(latency_ms, 2),
            "hits":           hits,
            "top_similarity": round(top_similarity, 4),
        }
        if kind == "incidents":
            self._rag_incidents.push(sample)
        else:
            self._rag_runbooks.push(sample)

    def rag_summary(self) -> dict[str, Any]:
        return {
            "incidents": self._summarise_rag(self._rag_incidents.snapshot()),
            "runbooks":  self._summarise_rag(self._rag_runbooks.snapshot()),
        }

    @staticmethod
    def _summarise_rag(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {
                "query_count":          0,
                "avg_latency_ms":       0.0,
                "p95_latency_ms":       0.0,
                "hit_rate":             0.0,
                "avg_top_similarity":   0.0,
            }
        latencies = sorted(s["latency_ms"] for s in samples)
        hits_arr  = [s["hits"] for s in samples]
        sims_arr  = [s["top_similarity"] for s in samples]
        return {
            "query_count":        len(samples),
            "avg_latency_ms":     round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms":     round(_pct(latencies, 95), 2),
            "hit_rate":           round(sum(1 for h in hits_arr if h > 0) / len(hits_arr), 3),
            "avg_top_similarity": round(sum(sims_arr) / len(sims_arr), 3),
        }

    # ────────────────────────────────────────────────────────────────
    # LLM
    # ────────────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        model: str,
        latency_ms: float,
        success: bool,
        prompt_chars: int | None = None,
        response_chars: int | None = None,
    ) -> None:
        self._llm_calls.push({
            "ts":             datetime.utcnow().isoformat(),
            "model":          model,
            "latency_ms":     round(latency_ms, 2),
            "success":        bool(success),
            "prompt_chars":   prompt_chars or 0,
            "response_chars": response_chars or 0,
        })

    def llm_summary(self) -> dict[str, Any]:
        samples = self._llm_calls.snapshot()
        if not samples:
            return {
                "call_count":     0,
                "success_rate":   1.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "avg_prompt_chars": 0,
            }
        lat = sorted(s["latency_ms"] for s in samples)
        ok  = sum(1 for s in samples if s["success"])
        return {
            "call_count":       len(samples),
            "success_rate":     round(ok / len(samples), 3),
            "avg_latency_ms":   round(sum(lat) / len(lat), 2),
            "p95_latency_ms":   round(_pct(lat, 95), 2),
            "avg_prompt_chars": int(sum(s["prompt_chars"]   for s in samples) / len(samples)),
        }

    # ────────────────────────────────────────────────────────────────
    # Pipeline performance (computed from DB on demand)
    # ────────────────────────────────────────────────────────────────

    def pipeline_performance(self, db: Session, hours: int = 24) -> list[dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        try:
            rows = (
                db.query(
                    Pipeline.id.label("id"),
                    Pipeline.name.label("name"),
                    func.count(PipelineRun.id).label("runs"),
                    func.sum(
                        case((PipelineRun.status == RunStatus.SUCCEEDED, 1), else_=0)
                    ).label("succeeded"),
                    func.sum(
                        case((PipelineRun.status == RunStatus.FAILED, 1), else_=0)
                    ).label("failed"),
                    func.avg(PipelineRun.duration_seconds).label("avg_duration"),
                    func.min(PipelineRun.duration_seconds).label("min_duration"),
                    func.max(PipelineRun.duration_seconds).label("max_duration"),
                )
                .outerjoin(PipelineRun, PipelineRun.pipeline_id == Pipeline.id)
                .filter((PipelineRun.started_at >= cutoff) | (PipelineRun.id.is_(None)))
                .group_by(Pipeline.id, Pipeline.name)
                .all()
            )
        except Exception:
            logger.exception("pipeline_performance aggregate query failed")
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            runs      = int(r.runs or 0)
            succeeded = int(r.succeeded or 0)
            failed    = int(r.failed or 0)
            done      = succeeded + failed
            sr        = (succeeded / done * 100.0) if done else 100.0

            # p95/p99 require fetching duration_seconds rows directly
            try:
                p50, p95, p99 = self._duration_pcts(db, r.id, cutoff)
            except Exception:
                logger.exception("duration percentile query failed for pipeline %s", r.id)
                p50, p95, p99 = 0.0, 0.0, 0.0

            out.append({
                "pipeline_id":      r.id,
                "pipeline_name":    r.name,
                "runs":             runs,
                "succeeded":        succeeded,
                "failed":           failed,
                "success_rate_pct": round(sr, 2),
                "avg_duration_sec": round(float(r.avg_duration or 0.0), 2),
                "min_duration_sec": round(float(r.min_duration or 0.0), 2),
                "max_duration_sec": round(float(r.max_duration or 0.0), 2),
                "p50_duration_sec": p50,
                "p95_duration_sec": p95,
                "p99_duration_sec": p99,
            })

        out.sort(key=lambda x: x["runs"], reverse=True)
        return out

    @staticmethod
    def _duration_pcts(db: Session, pipeline_id: int, cutoff: datetime) -> tuple[float, float, float]:
        durations = [
            float(d[0])
            for d in db.query(PipelineRun.duration_seconds)
                       .filter(
                            PipelineRun.pipeline_id == pipeline_id,
                            PipelineRun.started_at >= cutoff,
                            PipelineRun.duration_seconds.isnot(None),
                       )
                       .all()
            if d[0] is not None
        ]
        if not durations:
            return 0.0, 0.0, 0.0
        durations.sort()
        return (
            round(_pct(durations, 50), 2),
            round(_pct(durations, 95), 2),
            round(_pct(durations, 99), 2),
        )


def _pct(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(round((pct / 100.0) * (len(sorted_values) - 1)))))
    return float(sorted_values[k])


metrics_service = MetricsService()
