"""Pydantic schemas for Pipeline domain."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.pipeline import RunStatus, LogLevel


class PipelineOut(BaseModel):
    id: int
    connector_id: int
    external_id: str
    name: str
    description: str | None
    last_run_status: RunStatus | None
    last_run_at: datetime | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PipelineLogOut(BaseModel):
    id: int
    timestamp: datetime
    level: LogLevel
    source: str | None
    message: str

    model_config = ConfigDict(from_attributes=True)


class ErrorAnalysisOut(BaseModel):
    id: int
    summary: str
    root_cause: str | None
    suggested_fix: str | None
    fix_patch: str | None
    confidence: float | None
    auto_fix_applied: bool
    auto_fix_result: str | None
    model: str | None
    raw_response: dict[str, Any] | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PipelineRunOut(BaseModel):
    id: int
    pipeline_id: int
    external_run_id: str
    status: RunStatus
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    error_message: str | None
    analysis: ErrorAnalysisOut | None = None

    model_config = ConfigDict(from_attributes=True)


class PipelineDetailOut(PipelineOut):
    """Pipeline with recent runs included."""
    runs: list[PipelineRunOut] = []


class DashboardStats(BaseModel):
    total_connectors: int
    connected_connectors: int = 0
    total_pipelines: int
    healthy_pipelines: int = 0
    failed_pipelines: int = 0
    fleet_health_rate: float = 0.0
    runs_last_24h: int
    total_runs: int = 0
    success_rate_24h: float | None = None
    failed_runs_24h: int
    total_failed_runs: int = 0
    pending_analyses: int
