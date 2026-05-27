"""
Vector store backed by Chroma (local persistent mode).

Replaces the previous in-memory Qdrant client.

Two collections are maintained:
  - incidents : embeddings of historical pipeline-failure logs + their
                resolutions. Used to retrieve "similar past incidents" before
                we ask Mistral for a diagnosis.
  - runbooks  : embeddings of chunks extracted from user-uploaded SOP files
                (PDF / DOCX / MD / TXT).

Both collections use cosine distance and the same embedding model
(all-MiniLM-L6-v2, 384-dim) supplied by EmbeddingService.

Resilience note
---------------
Chroma's on-disk schema has changed between minor 0.5.x releases. If the
SQLite metadata under ``CHROMA_DB_PATH`` was written by a different chromadb
version, opening the collections can raise ``KeyError: '_type'`` (or similar
configuration-deserialisation errors) deep inside chromadb. That blows up the
entire backend at import time, which is brutal.

To make startup self-healing, this module catches that class of error,
wipes the on-disk Chroma directory, and rebuilds the collections fresh.
You lose any indexed data when this happens (which is the only safe option:
the stored config can't be decoded so the data can't be read either), but
the server keeps running instead of crash-looping.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

logger = logging.getLogger(__name__)


# Errors raised by chromadb when it cannot deserialise an existing collection's
# stored configuration (typically because the on-disk format is from a
# different chromadb minor version than the one currently installed).
_CHROMA_CONFIG_ERROR_HINTS = (
    "_type",                # KeyError: '_type'  (seen on 0.5.x cross-version)
    "configuration",
    "from_json",
    "CollectionConfiguration",
)


def _looks_like_chroma_config_error(exc: BaseException) -> bool:
    """Heuristic: is this an on-disk schema mismatch we should recover from?"""
    msg = str(exc) or ""
    tb_repr = repr(exc)
    blob = (msg + " " + tb_repr).lower()
    return any(hint.lower() in blob for hint in _CHROMA_CONFIG_ERROR_HINTS)


class VectorService:
    """
    Thin wrapper around chromadb.PersistentClient.

    Centralises the collection names, lets the rest of the code stay agnostic
    about whether the backend is Chroma or Qdrant, and gives us one place to
    instrument retrieval performance.
    """

    INCIDENTS_COLLECTION = "incidents"
    RUNBOOKS_COLLECTION  = "runbooks"

    def __init__(self) -> None:
        self._chroma_path = Path(settings.CHROMA_DB_PATH)
        self._chroma_path.mkdir(parents=True, exist_ok=True)

        try:
            self._open_or_create()
        except Exception as exc:
            # If Chroma can't read its own on-disk config, the only safe
            # recovery is to wipe and start fresh. We log loudly so the
            # user knows their indexed data is gone.
            if _looks_like_chroma_config_error(exc):
                logger.warning(
                    "Chroma config schema mismatch at %s (%s). "
                    "Wiping the directory and rebuilding fresh collections.",
                    self._chroma_path, exc,
                )
                self._wipe_and_retry()
            else:
                # Some other error — re-raise, the user needs to see it.
                raise

        logger.info(
            "Chroma ready at %s (incidents=%d, runbooks=%d)",
            self._chroma_path,
            self.incidents.count(),
            self.runbooks.count(),
        )

    # ------------------------------------------------------------------
    # Init / recovery
    # ------------------------------------------------------------------

    def _open_or_create(self) -> None:
        """Open a PersistentClient and (idempotently) create both collections."""
        # PersistentClient writes to disk so vectors survive restarts.
        # anonymized_telemetry=False keeps Chroma from phoning home.
        # allow_reset=True is required for client.reset() to be allowed.
        self.client = chromadb.PersistentClient(
            path=str(self._chroma_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # get_or_create makes restarts idempotent.
        # metadata={"hnsw:space": "cosine"} so similarity ≈ 1 - distance.
        self.incidents = self.client.get_or_create_collection(
            name=self.INCIDENTS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.runbooks = self.client.get_or_create_collection(
            name=self.RUNBOOKS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _wipe_and_retry(self) -> None:
        """Nuclear option: delete the Chroma directory and start fresh."""

        import gc
        import os
        import time

        logger.warning("Starting Chroma recovery process...")

        try:
            # Release all references
            self.incidents = None
            self.runbooks = None
            self.client = None

            gc.collect()

            # Windows SQLite lock release
            time.sleep(2)

            if self._chroma_path.exists():
                logger.warning(
                    "Deleting corrupted Chroma DB directory: %s",
                    self._chroma_path,
                )

                shutil.rmtree(self._chroma_path)

                logger.warning("Chroma DB directory deleted")

            # recreate clean folder
            self._chroma_path.mkdir(parents=True, exist_ok=True)

            time.sleep(1)

        except Exception as exc:
            logger.exception(
                "Failed to wipe Chroma directory %s",
                self._chroma_path,
            )
            raise RuntimeError(
                f"Could not reset corrupted Chroma DB: {exc}"
            ) from exc

        logger.warning("Recreating fresh Chroma collections...")

        self._open_or_create()
    # ------------------------------------------------------------------
    # Inserts
    # ------------------------------------------------------------------

    def add_incident(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Upsert one historical incident vector."""
        self.incidents.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[self._clean_meta(metadata)],
        )

    def add_runbook_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Bulk upsert chunks for a single uploaded runbook."""
        if not ids:
            return
        self.runbooks.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[self._clean_meta(m) for m in metadatas],
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_incidents(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        return self._query(self.incidents, query_embedding, k)

    def query_runbooks(
        self,
        query_embedding: list[float],
        k: int = 4,
    ) -> list[dict[str, Any]]:
        return self._query(self.runbooks, query_embedding, k)

    @staticmethod
    def _query(coll, embedding: list[float], k: int) -> list[dict[str, Any]]:
        if coll.count() == 0:
            return []
        res = coll.query(
            query_embeddings=[embedding],
            n_results=min(k, coll.count()),
            include=["documents", "metadatas", "distances"],
        )

        out: list[dict[str, Any]] = []
        ids        = (res.get("ids")        or [[]])[0]
        docs       = (res.get("documents")  or [[]])[0]
        metas      = (res.get("metadatas")  or [[]])[0]
        distances  = (res.get("distances")  or [[]])[0]

        for i, _id in enumerate(ids):
            distance   = float(distances[i]) if i < len(distances) else 1.0
            # cosine distance is in [0, 2] for Chroma; similarity ≈ 1 - distance
            similarity = max(0.0, 1.0 - distance)
            out.append({
                "id":         _id,
                "document":   docs[i]  if i < len(docs)  else "",
                "metadata":   metas[i] if i < len(metas) else {},
                "distance":   distance,
                "similarity": similarity,
            })
        return out

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def delete_runbook(self, runbook_id: str) -> int:
        """Remove every chunk belonging to one runbook id. Returns count deleted."""
        existing = self.runbooks.get(where={"runbook_id": runbook_id}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self.runbooks.delete(ids=ids)
        return len(ids)

    def stats(self) -> dict[str, int]:
        return {
            "incidents": self.incidents.count(),
            "runbooks":  self.runbooks.count(),
        }

    @staticmethod
    def _clean_meta(m: dict[str, Any]) -> dict[str, Any]:
        """Chroma only accepts str|int|float|bool in metadata values."""
        cleaned: dict[str, Any] = {}
        for k, v in (m or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            else:
                cleaned[k] = str(v)
        return cleaned


# Module-level singleton (instantiated lazily on first import)
# vector_service = VectorService()
# Lazy singleton instance
_vector_service = None


def get_vector_service():
    global _vector_service

    if _vector_service is None:
        _vector_service = VectorService()

    return _vector_service