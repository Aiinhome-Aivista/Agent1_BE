"""
Pytest configuration for REAL integration tests.

These tests hit real ADF / Databricks / GitHub / Mistral. Each provider's tests
skip cleanly when its credentials aren't present, so you can run the suite even
if you only have one provider configured.

Configure by:
  1. cp .env.test.example .env.test
  2. fill in only the providers you have access to
  3. pip install -r requirements-test.txt
  4. pytest -m integration

The Git auto-fix test is destructive (it files a real GitHub issue) and is
gated behind both `-m destructive` AND env var RUN_DESTRUCTIVE_TESTS=1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Make `app.*` importable regardless of where pytest is invoked from
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env.test (preferred) or fall back to .env
ENV_TEST = ROOT / ".env.test"
ENV_DEFAULT = ROOT / ".env"
if ENV_TEST.exists():
    load_dotenv(ENV_TEST, override=False)
elif ENV_DEFAULT.exists():
    load_dotenv(ENV_DEFAULT, override=False)

# ---------------------------------------------------------------------------
# Required credential groups
# ---------------------------------------------------------------------------
ADF_VARS = (
    "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET",
    "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_FACTORY_NAME",
)
DATABRICKS_VARS = ("DATABRICKS_WORKSPACE_URL", "DATABRICKS_TOKEN")
GIT_VARS = ("GITHUB_TOKEN", "GITHUB_OWNER")  # GITHUB_REPO is optional
MISTRAL_VARS = ("MISTRAL_API_KEY",)


def _missing(group: tuple[str, ...]) -> list[str]:
    return [v for v in group if not os.environ.get(v)]


def skip_if_missing(group: tuple[str, ...], label: str) -> None:
    """Skip the current test if any of the env vars in `group` is missing."""
    miss = _missing(group)
    if miss:
        pytest.skip(f"{label} integration disabled: missing {', '.join(miss)}")


# ---------------------------------------------------------------------------
# Credential-dict fixtures (shape matches what the connectors expect)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def adf_credentials() -> dict[str, Any]:
    skip_if_missing(ADF_VARS, "ADF")
    return {
        "tenant_id":       os.environ["AZURE_TENANT_ID"],
        "client_id":       os.environ["AZURE_CLIENT_ID"],
        "client_secret":   os.environ["AZURE_CLIENT_SECRET"],
        "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
        "resource_group":  os.environ["AZURE_RESOURCE_GROUP"],
        "factory_name":    os.environ["AZURE_FACTORY_NAME"],
    }


@pytest.fixture(scope="session")
def databricks_credentials() -> dict[str, Any]:
    skip_if_missing(DATABRICKS_VARS, "Databricks")
    return {
        "workspace_url":         os.environ["DATABRICKS_WORKSPACE_URL"],
        "personal_access_token": os.environ["DATABRICKS_TOKEN"],
    }


@pytest.fixture(scope="session")
def git_credentials() -> dict[str, Any]:
    skip_if_missing(GIT_VARS, "GitHub")
    return {
        "provider": "github",
        "token":    os.environ["GITHUB_TOKEN"],
        "owner":    os.environ["GITHUB_OWNER"],
        "repo":     os.environ.get("GITHUB_REPO") or None,
    }


@pytest.fixture(scope="session")
def mistral_configured() -> None:
    skip_if_missing(MISTRAL_VARS, "Mistral")


# ---------------------------------------------------------------------------
# Connector instances
# ---------------------------------------------------------------------------
@pytest.fixture
def adf_connector(adf_credentials):
    from app.connectors import ADFConnector
    return ADFConnector(adf_credentials)


@pytest.fixture
def databricks_connector(databricks_credentials):
    from app.connectors import DatabricksConnector
    return DatabricksConnector(databricks_credentials)


@pytest.fixture
def git_connector(git_credentials):
    from app.connectors import GitConnector
    return GitConnector(git_credentials)


# ---------------------------------------------------------------------------
# In-memory DB session (for the sync end-to-end test)
# ---------------------------------------------------------------------------
@pytest.fixture
def memory_db():
    """Spin up a fresh SQLite-in-memory DB with the full schema for one test."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.core.database import Base

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# Destructive-test gate
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.destructive unless RUN_DESTRUCTIVE_TESTS=1."""
    if os.environ.get("RUN_DESTRUCTIVE_TESTS") == "1":
        return
    skip_destructive = pytest.mark.skip(
        reason="destructive test - set RUN_DESTRUCTIVE_TESTS=1 to enable"
    )
    for item in items:
        if "destructive" in item.keywords:
            item.add_marker(skip_destructive)
