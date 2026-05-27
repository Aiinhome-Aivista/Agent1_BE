"""
Graph-RAG Service
─────────────────
At diagnosis time we already have:

  - Chroma: similar past incidents + raw runbook chunks (semantic)

This service adds the second leg:

  - Arango: STRUCTURED fix candidates, ranked by how many past
            incidents used each fix successfully.

The two are combined into one ``context_block`` that's appended to the
Mistral prompt. The Mistral prompt already tells the model:

    "If the GRAPH or RAG context contains a clearly applicable fix,
     prefer it; cite the runbook or past incident by name when you do."

Public surface:

  build_graph_context(error_text, component=...) -> str
      A ready-to-concat string for the prompt.

  combined_context_block(chroma_block, error_text, component=...) -> str
      Stitches Chroma's existing context with the new graph section,
      so existing incident_service code only needs a one-line change.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services import arango_service as graph

logger = logging.getLogger(__name__)


def build_graph_context(
    error_text: str,
    *,
    component: str | None = None,
    top_k: int = 5,
) -> str:
    """Render the ranked fix candidates from the graph as a prompt fragment."""
    if not graph.is_enabled():
        return ""

    try:
        candidates = graph.recommend_fixes_for_query(
            error_text, component=component, top_k=top_k,
        )
    except Exception as e:
        logger.warning("graph context build failed: %s", e)
        return ""

    if not candidates:
        return ""

    lines: list[str] = []
    lines.append("Graph-ranked fix candidates "
                 "(score = pattern match × keyword hits + 2·past successes − failures):")
    for i, c in enumerate(candidates, start=1):
        rb_list = ", ".join(c.get("supporting_runbooks") or []) or "—"
        pat_list = ", ".join(c.get("matched_patterns") or []) or "—"
        lines.append(
            f"  #{i} [score={c.get('score', 0):.1f}, "
            f"past_success={c.get('success_count', 0)}, "
            f"past_failure={c.get('failure_count', 0)}] "
            f"{c.get('fix','').strip()}"
        )
        lines.append(f"     • matches patterns: {pat_list}")
        lines.append(f"     • from runbooks:    {rb_list}")
    return "\n".join(lines)


def combined_context_block(
    chroma_context: str | None,
    error_text: str,
    *,
    component: str | None = None,
    top_k: int = 5,
) -> str:
    """
    Drop-in replacement for whatever string `incident_service.py`
    currently passes as `context_block=` to mistral_service.analyze_failure.

    Just wrap the existing chroma_context like this:

        context_block = combined_context_block(
            chroma_context = rag_service.build_context_block(...),
            error_text     = error_message_with_log_tail,
            component      = inc.connector_type,
        )
    """
    parts: list[str] = []
    if chroma_context and chroma_context.strip():
        parts.append("--- SEMANTIC CONTEXT (Chroma) ---")
        parts.append(chroma_context.strip())

    graph_part = build_graph_context(error_text, component=component, top_k=top_k)
    if graph_part:
        parts.append("--- STRUCTURED CONTEXT (Graph) ---")
        parts.append(graph_part)

    return "\n\n".join(parts) if parts else ""


def explain_recommendation(query: str, fix_id: str) -> dict[str, Any]:
    """
    "Why was this fix recommended?" — used by the UI explainer.
    Returns the matched patterns, the supporting runbooks, and the
    incident history for that one fix.
    """
    if not graph.is_enabled():
        return {"enabled": False}

    db = graph.get_db()
    if db is None:
        return {"enabled": False}

    aql = """
    LET fix = DOCUMENT(@fix_id)
    FILTER fix != null
    LET patterns = (
      FOR p, e IN 1..1 INBOUND fix._id pattern_uses_fix
        RETURN { signature: p.signature, weight: e.weight, keywords: p.keywords }
    )
    LET runbooks = (
      FOR rb, e IN 1..1 INBOUND fix._id runbook_has_step
        RETURN { title: rb.title, category: rb.category, order: e.order }
    )
    LET incidents = (
      FOR i, e IN 1..1 INBOUND fix._id incident_used_fix
        RETURN { id: i.incident_id, pipeline: i.pipeline_name,
                 summary: i.summary, success: e.success }
    )
    RETURN {
      enabled:  true,
      fix:      fix.text,
      patterns: patterns,
      runbooks: runbooks,
      incidents: incidents,
      success_rate: LENGTH(incidents) == 0 ? null
        : LENGTH(incidents[* FILTER CURRENT.success]) / LENGTH(incidents)
    }
    """
    try:
        cur = db.aql.execute(aql, bind_vars={"fix_id": fix_id})
        return next(iter(cur), {"enabled": True, "fix": "", "patterns": [],
                                "runbooks": [], "incidents": []})
    except Exception as e:
        logger.warning("explain_recommendation failed: %s", e)
        return {"enabled": True, "error": str(e)}
