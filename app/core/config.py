"""
Application configuration loaded from environment variables.
"""
from functools import lru_cache
from pathlib import Path

# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Database
    DATABASE_URL: str = "mysql+pymysql://root:rootpass@localhost:3306/pipeline_monitor"

    # Security
    SECRET_KEY: str = "dev-secret-change-in-prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ENCRYPTION_KEY: str = ""  # Fernet key for encrypting connector credentials

    # Mistral
    MISTRAL_MODE: str = "Cloud"            # "Cloud" | "Local" | "Gemini"
    MISTRAL_API_KEY: str = "IotlgX9OC7gWRj0WqHuT5xdhT1LNkNne"
    MODEL_NAME: str = "mistral-small-latest"
    MISTRAL_LOCAL_URL: str = "http://localhost:11434"
    MISTRAL_LOCAL_MODEL: str = "mistral:latest"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2000
    MISTRAL_MODEL: str = "mistral-small-latest"
    
    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"

    # ArangoDB
    ARANGO_ENABLED: bool = True
    ARANGO_URL: str = "http://localhost:8529"
    ARANGO_USER: str = "root"
    ARANGO_PASSWORD: str = "arango_pass"
    ARANGO_DB: str = "pipeline_graph"
    GRAPH_RAG_TOP_K: int = 5

    # Jira Configuration
    JIRA_BASE_URL: str = ""
    JIRA_USER_EMAIL: str = ""
    JIRA_API_TOKEN: str = ""
    JIRA_PROJECT_KEY: str = "DATAOPS"
    JIRA_HUMAN_ASSIGNEE_ACCOUNT_ID: str = ""

    # CORS
    FRONTEND_URL: str = "http://localhost:5173"

    # Polling
    PIPELINE_SYNC_INTERVAL: int = 60
    LOG_FETCH_INTERVAL: int = 30
    ESCALATION_CHECK_INTERVAL: int = 30    # seconds (30s) — how often the
                                            # scheduler checks active incidents

    # Redis (optional)
    REDIS_URL: str = "redis://localhost:6379/0"

    # ─── NEW: RAG + Runbooks ────────────────────────────────────────────────
    # Where Chroma DB stores its persistent files
    CHROMA_DB_PATH: str = "./data/chroma"

    # Where uploaded runbook source files (pdf/docx/md/txt) are stored on disk
    RUNBOOKS_DIR: str = "./data/runbooks"

    # Embedding model dimension – must match the model in EmbeddingService
    EMBEDDING_DIM: int = 384

    # RAG retrieval params
    RAG_TOP_K_INCIDENTS: int = 5        # top-N similar historical incidents
    RAG_TOP_K_RUNBOOKS: int = 4         # top-N runbook chunks
    RAG_MIN_SIMILARITY: float = 0.25    # below this we treat result as noise

    # Runbook chunking (used when ingesting uploaded docs into vector store)
    RUNBOOK_CHUNK_CHARS: int = 1200     # ~250-300 tokens
    RUNBOOK_CHUNK_OVERLAP: int = 150


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # Make sure local dirs exist – this lets the demo run zero-config
    Path(s.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    Path(s.RUNBOOKS_DIR).mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
