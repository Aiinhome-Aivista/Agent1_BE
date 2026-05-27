"""
Graph (ArangoDB) endpoints.

Mount in app/main.py:

    from app.api import graph as graph_router
    app.include_router(graph_router.router, prefix="/api/v1")

──────────────────────────────────────────────────────────────────────
Endpoints
──────────────────────────────────────────────────────────────────────
  GET  /api/v1/graph/stats
        → vertex + edge counts per collection (Metrics page badge)

  POST /api/v1/graph/recommend
       Body: { "error_text": "...", "component": "optional", "top_k": 5 }
        → graph-ranked fix candidates (NOT a Mistral call —
           just a graph traversal, useful for the UI "Quick fixes"
           drawer before the user even runs full diagnosis)

  GET  /api/v1/graph/runbook/{runbook_id}/subgraph
        → nodes + edges around a runbook, for the Knowledge Graph viewer

  GET  /api/v1/graph/fix/{fix_id_b64}/explain
        → why was this fix suggested? Lists matched patterns, supporting
           runbooks, and prior incident outcomes for that one fix.

  POST /api/v1/graph/enrich-runbook/{runbook_id}
        → manual re-trigger of the enrichment for an existing runbook
           (useful after you swap LLM models or change the extraction
           prompt). Usually you don't call this — runbook upload triggers
           enrichment automatically.
"""
from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import arango_service as graph
from app.services import graph_rag_service
from app.services import graph_enrichment_service


router = APIRouter(prefix="/graph", tags=["graph"])


# ── stats ─────────────────────────────────────────────────────────────


@router.get("/stats")
def stats() -> dict[str, Any]:
    return graph.graph_stats()


# ── recommend ─────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    error_text: str = Field(..., description="error message + relevant log lines")
    component: str | None = Field(None, description="optional component filter, e.g. 'Databricks'")
    top_k: int = 5


@router.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    if not graph.is_enabled():
        return {"enabled": False, "candidates": []}
    candidates = graph.recommend_fixes_for_query(
        req.error_text, component=req.component, top_k=req.top_k,
    )
    return {"enabled": True, "candidates": candidates}


# ── subgraph (for UI viewer) ──────────────────────────────────────────


@router.get("/runbook/{runbook_id}/subgraph")
def runbook_subgraph(runbook_id: str) -> dict[str, Any]:
    return graph.get_runbook_subgraph(runbook_id)


# ── explainer ─────────────────────────────────────────────────────────


@router.get("/fix/{fix_id_b64}/explain")
def explain_fix(fix_id_b64: str) -> dict[str, Any]:
    """fix_id is base64-url-encoded because '/' isn't path-safe."""
    try:
        fix_id = base64.urlsafe_b64decode(fix_id_b64.encode()).decode()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid fix_id encoding")
    return graph_rag_service.explain_recommendation(query="", fix_id=fix_id)


# ── manual enrichment (admin) ─────────────────────────────────────────


class EnrichRequest(BaseModel):
    title: str
    category: str
    text: str
    risk_level: str | None = None
    description: str | None = None


@router.post("/enrich-runbook/{runbook_id}")
def enrich_runbook(runbook_id: str, req: EnrichRequest) -> dict[str, Any]:
    """
    Manually re-enrich a runbook from its already-extracted text.
    The auto path is wired into the upload endpoint — see docs/INTEGRATION.md.
    """
    return graph_enrichment_service.enrich_runbook(
        runbook_id=runbook_id,
        title=req.title,
        category=req.category,
        text=req.text,
        risk_level=req.risk_level,
        description=req.description,
    )


# ── incident outcome callback ─────────────────────────────────────────


class IncidentOutcome(BaseModel):
    incident_id: str
    pipeline_name: str
    summary: str
    confidence: float = 0.0
    matched_pattern_signatures: list[str] = []
    applied_fixes: list[str] = []
    success: bool = True


@router.post("/incident-outcome")
def incident_outcome(req: IncidentOutcome) -> dict[str, Any]:
    """
    Call this after an incident is resolved. The graph learns which
    fix_actions actually worked — that boosts/penalises their score
    in future `recommend` calls.
    """
    return graph_enrichment_service.record_incident_outcome(
        incident_id=req.incident_id,
        pipeline_name=req.pipeline_name,
        summary=req.summary,
        confidence=req.confidence,
        matched_pattern_signatures=req.matched_pattern_signatures,
        applied_fixes=req.applied_fixes,
        success=req.success,
    )
