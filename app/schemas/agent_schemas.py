"""Pydantic schemas for the five agent-loop sections."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.agent_models import IncidentStatus


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

class TimelineEntry(BaseModel):
    ts:     str
    stage:  str
    agent:  str
    detail: str


class ToolCallRecord(BaseModel):
    tool:        str
    args:        dict[str, Any] = {}
    result:      dict[str, Any] | None = None
    status:      str | None = None
    duration_ms: int | None = None


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:                int
    run_id:            int | None
    pipeline_name:     str
    status:            str
    risk_tier:         str
    detected_at:       datetime
    resolved_at:       datetime | None
    error_log:         str
    failed_node:       str | None
    root_cause:        str | None
    proposed_action:   str | None
    agent_thought:     str | None
    remediation_plan:  list[str]
    similar_incidents: list[str]
    confidence_score:  float | None
    tool_calls:        list[dict]
    timeline:          list[dict]
    approval_required: bool
    approved_by:       str | None
    approved_at:       datetime | None

    # ─── NEW: email-dispatch lifecycle (drives the Incident Timeline page) ──
    initial_email_sent_at:       datetime | None = None
    initial_email_recipient:     str | None      = None
    initial_email_role:          str | None      = None
    escalation_email_sent_at:    datetime | None = None
    escalation_email_recipients: list[dict] | None = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class MemoryEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:               int
    kind:             str
    title:            str
    summary:          str
    payload:          dict[str, Any]
    tags:             list[str]
    success:          bool | None
    times_referenced: int
    created_at:       datetime
    # only present on search results
    similarity:       float | None = None


class MemoryEntryCreate(BaseModel):
    kind:    str
    title:   str
    summary: str
    payload: dict[str, Any] = {}
    tags:    list[str] = []
    success: bool | None = None


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class RecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:            int
    pipeline_id:   str | None
    pipeline_name: str
    title:         str
    detail:        str
    savings:       str
    risk:          str
    status:        str
    created_at:    datetime


class RecommendationUpdate(BaseModel):
    status: str   # open / accepted / dismissed


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:          int
    ts:          datetime
    type:        str
    msg:         str
    agent_role:  str | None
    incident_id: str | None
    actor:       str | None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgentStatusOut(BaseModel):
    role:             str
    name:             str
    description:      str
    color:            str
    status:           str   # idle / thinking / acting / error
    last_action:      str
    tasks_completed:  int


class ToolSpecOut(BaseModel):
    name:        str
    description: str
    args_schema: dict[str, str]
    risk:        str   # low / medium / high
