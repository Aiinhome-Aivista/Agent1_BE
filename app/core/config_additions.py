"""
ADD THESE FIELDS to your existing app/core/config.py Settings class.

Don't replace the whole file — your file already has DATABASE_URL,
SECRET_KEY, MISTRAL_*, CHROMA_*, RUNBOOKS_*, etc. Just paste these
inside the existing class, then make sure your .env has the
corresponding keys (see .env.additions).
"""

from pathlib import Path
# from pydantic_settings import BaseSettings, SettingsConfigDict   # already imported


class _ExampleSettingsAdditions:
    """
    ──────────────────────────────────────────────────────────────────
    LLM mode-switch knobs
    ──────────────────────────────────────────────────────────────────
    These replace/extend the single MISTRAL_API_KEY+MISTRAL_MODEL pair.
    `mistral_service` keeps working as before because of the
    back-compat shim, but new code should read from these.
    """
    MISTRAL_MODE: str = "Cloud"            # "Cloud" | "Local"
    MISTRAL_API_KEY: str = ""              # used in Cloud mode
    MODEL_NAME: str = "mistral-small-latest"  # used in Cloud mode
    MISTRAL_LOCAL_URL: str = "http://localhost:11434"  # used in Local mode (Ollama default)
    MISTRAL_LOCAL_MODEL: str = "mistral:latest"        # used in Local mode

    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2000

    # Kept for backward compatibility with the old config — points at
    # the same value as MODEL_NAME so legacy callers like
    # `mistral_service.model` still work.
    MISTRAL_MODEL: str = "mistral-small-latest"

    """
    ──────────────────────────────────────────────────────────────────
    ArangoDB graph knobs
    ──────────────────────────────────────────────────────────────────
    Set ARANGO_ENABLED=false to completely turn off the graph layer —
    the diagnosis flow will fall back to Chroma-only.
    """
    ARANGO_ENABLED: bool = True
    ARANGO_URL: str = "http://localhost:8529"
    ARANGO_USER: str = "root"
    ARANGO_PASSWORD: str = ""
    ARANGO_DB: str = "pipeline_graph"

    # How many graph-ranked fix candidates to add to the LLM prompt
    GRAPH_RAG_TOP_K: int = 5
