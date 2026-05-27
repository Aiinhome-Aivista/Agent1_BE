"""
Base connector contract.

Every concrete connector (ADF, Databricks, Git) must implement these methods so
the rest of the app can treat them uniformly.

NormalizedPipeline / NormalizedRun / NormalizedLog are simple dataclasses used
as the common shape returned by all providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NormalizedPipeline:
    external_id: str
    name: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedRun:
    external_run_id: str
    status: str  # one of RunStatus values
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedLog:
    timestamp: datetime
    level: str  # DEBUG/INFO/WARNING/ERROR/CRITICAL
    message: str
    source: str | None = None


class BaseConnector(ABC):
    """Abstract base class for all pipeline source connectors."""

    type_name: str = "BASE"

    def __init__(self, credentials: dict[str, Any]):
        self.credentials = credentials

    # --- lifecycle ---------------------------------------------------------
    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """Quickly validate credentials. Returns (success, human_message)."""

    @abstractmethod
    def list_pipelines(self) -> list[NormalizedPipeline]:
        """Enumerate pipelines available for this account."""

    @abstractmethod
    def list_runs(self, pipeline_external_id: str, limit: int = 25) -> list[NormalizedRun]:
        """Recent runs for a pipeline, newest first."""

    @abstractmethod
    def get_logs(self, pipeline_external_id: str, run_external_id: str) -> list[NormalizedLog]:
        """Fetch logs for a specific run."""

    # --- optional capabilities ---------------------------------------------
    def supports_auto_fix(self) -> bool:
        """Whether this connector can apply LLM-suggested fixes back to the source."""
        return False

    def apply_fix(self, pipeline_external_id: str, patch: str) -> tuple[bool, str]:
        """Apply a patch returned by the LLM. Default: not supported."""
        return False, f"{self.type_name} connector does not support automatic fixes."
