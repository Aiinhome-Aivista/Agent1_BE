"""
Retrieval-Augmented Generation orchestration.

The previous `rag_service.py` had `search_similar_logs` accidentally nested
INSIDE `store_log`, so it was never callable. This version fixes that and
adds:

    - store_incident(...)        write a resolved incident into the vector store
    - search_similar_incidents() retrieve the top-K historical incidents
    - search_runbooks()          retrieve top-K runbook chunks
    - build_context_block()      format retrieved snippets into a prompt
                                 fragment that Mistral can consume

Also records simple metrics (latency, hit count) into the in-process
metrics_service so the UI can show RAG health.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.vector_service import get_vector_service

logger = logging.getLogger(__name__)


class RagService:
    # ──────────────────────────────────────────────────────────────────
    # WRITE: incidents
    # ──────────────────────────────────────────────────────────────────

    def store_incident(
        self,
        *,
        incident_id: int | str,
        pipeline_name: str,
        error_log: str,
        root_cause: str | None,
        suggested_fix: str | None,
        confidence: float | None,
        risk_tier: str | None,
    ) -> str:
        """
        Persist a resolved incident so future similar failures can retrieve it.

        The "text" we embed is a compact narrative that combines the error log
        with the diagnosis – this way both a future error log AND a future
        free-text query both work.
        """
        narrative = (
            f"Pipeline: {pipeline_name}\n"
            f"Error log:\n{(error_log or '').strip()[:4000]}\n\n"
            f"Root cause: {(root_cause or '').strip()[:1500]}\n"
            f"Fix: {(suggested_fix or '').strip()[:2000]}"
        )

        vec = embedding_service.embed(narrative)
        doc_id = f"inc-{incident_id}-{uuid.uuid4().hex[:8]}"
        vector_service = get_vector_service()
        vector_service.add_incident(
            doc_id=doc_id,
            text=narrative,
            embedding=vec,
            metadata={
                "incident_id":   str(incident_id),
                "pipeline_name": pipeline_name,
                "risk_tier":     risk_tier or "Medium",
                "confidence":    float(confidence or 0.0),
                "kind":          "incident",
            },
        )
        return doc_id

    # ──────────────────────────────────────────────────────────────────
    # READ: incidents
    # ──────────────────────────────────────────────────────────────────

    def search_similar_incidents(
        self,
        query: str,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        k = k or settings.RAG_TOP_K_INCIDENTS
        t0 = time.perf_counter()

        emb     = embedding_service.embed(query)
        vector_service = get_vector_service()
        results = vector_service.query_incidents(emb, k=k)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Filter out very low-similarity matches – they add noise to the prompt
        results = [r for r in results if r["similarity"] >= settings.RAG_MIN_SIMILARITY]

        # Lazy-import to avoid a hard cycle if metrics_service is not yet loaded
        try:
            from app.services.metrics_service import metrics_service
            metrics_service.record_rag_query(
                kind="incidents",
                latency_ms=elapsed_ms,
                hits=len(results),
                top_similarity=results[0]["similarity"] if results else 0.0,
            )
        except Exception:
            pass

        return results

    # ──────────────────────────────────────────────────────────────────
    # READ: runbooks
    # ──────────────────────────────────────────────────────────────────

    def search_runbooks(
        self,
        query: str,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        k = k or settings.RAG_TOP_K_RUNBOOKS
        t0 = time.perf_counter()

        emb     = embedding_service.embed(query)
        vector_service = get_vector_service()
        results = vector_service.query_runbooks(emb, k=k)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results = [r for r in results if r["similarity"] >= settings.RAG_MIN_SIMILARITY]

        try:
            from app.services.metrics_service import metrics_service
            metrics_service.record_rag_query(
                kind="runbooks",
                latency_ms=elapsed_ms,
                hits=len(results),
                top_similarity=results[0]["similarity"] if results else 0.0,
            )
        except Exception:
            pass

        return results

    # ──────────────────────────────────────────────────────────────────
    # Prompt context builder
    # ──────────────────────────────────────────────────────────────────

    def build_context_block(
        self,
        similar_incidents: list[dict[str, Any]],
        runbook_chunks:    list[dict[str, Any]],
    ) -> str:
        """
        Render retrieved context into a markdown block the LLM can read.
        Returns empty string when nothing relevant was found, so the caller
        can skip injecting it.
        """
        parts: list[str] = []

        if similar_incidents:
            parts.append("## Similar historical incidents\n")
            for i, hit in enumerate(similar_incidents, start=1):
                m   = hit.get("metadata") or {}
                sim = hit.get("similarity", 0.0)
                parts.append(
                    f"### {i}. {m.get('pipeline_name','(unknown)')}  "
                    f"(similarity {sim:.0%}, risk {m.get('risk_tier','?')})\n"
                    f"{(hit.get('document') or '').strip()[:1200]}\n"
                )

        if runbook_chunks:
            parts.append("## Relevant runbook excerpts\n")
            for i, hit in enumerate(runbook_chunks, start=1):
                m     = hit.get("metadata") or {}
                title = m.get("title") or m.get("source_filename") or "Runbook"
                sim   = hit.get("similarity", 0.0)
                parts.append(
                    f"### {i}. {title}  (similarity {sim:.0%})\n"
                    f"{(hit.get('document') or '').strip()[:1200]}\n"
                )

        return "\n".join(parts)


rag_service = RagService()
