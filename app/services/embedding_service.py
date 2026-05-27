"""
Sentence-Transformers wrapper used by both incident retrieval and runbook
ingestion.

Same model (all-MiniLM-L6-v2, 384-dim) for both so the two Chroma collections
live in the same vector space and we can mix them in a single query if we
later want hybrid retrieval.
"""
from __future__ import annotations

import logging
import time

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", self.MODEL_NAME)
        t0 = time.perf_counter()
        self.model = SentenceTransformer(self.MODEL_NAME)
        logger.info("Embedding model loaded in %.2fs", time.perf_counter() - t0)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return arr.tolist()

    # Back-compat alias for any older callers
    def create_embedding(self, text: str) -> list[float]:
        return self.embed(text)


embedding_service = EmbeddingService()
