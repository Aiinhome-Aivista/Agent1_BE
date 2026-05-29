"""
Models for the five agent-loop sections:
  - Incident       : failed run tracked through its resolution lifecycle
  - IncidentEvent  : journey log — every mail sent, rerun detected, escalation, etc.
  - MemoryEntry    : three-tier RAG store (episodic / semantic / procedural)
  - Recommendation : LLM-generated optimisation suggestions
  - AuditLog       : persistent decision log streamed to the Audit Trail page

Incident lifecycle columns track the email-dispatch and escalation state so the
frontend Incident Timeline can render the exact times and recipients.
"""
import enum
from datetime import datetime

# pyrefly: ignore [missing-import]
from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer,
    JSON, String, Text,
)
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import relationship

from app.core.database import Base


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

class IncidentStatus(str, enum.Enum):
    DETECTED         = "Detected"
    REASONING        = "Reasoning"
    PLANNING         = "Planning"
    AWAITING_APPROVAL = "Awaiting Approval"
    PROCESSING       = "Processing"
    EXECUTING        = "Executing"
    EVALUATING       = "Evaluating"
    REMEDIATED       = "Remediated"
    FAILED           = "Failed"
    ESCALATED        = "Escalated"


class IncidentEventType(str, enum.Enum):
    PIPELINE_FAILED       = "PIPELINE_FAILED"
    INITIAL_MAIL_SENT     = "INITIAL_MAIL_SENT"
    ESCALATION_CHECK      = "ESCALATION_CHECK"
    ESCALATION_MAIL_SENT  = "ESCALATION_MAIL_SENT"
    RERUN_DETECTED        = "RERUN_DETECTED"
    RERUN_SUCCEEDED       = "RERUN_SUCCEEDED"
    RERUN_FAILED          = "RERUN_FAILED"
    RESOLVED              = "RESOLVED"
    JIRA_TICKET_CREATED   = "JIRA_TICKET_CREATED"


class Incident(Base):
    __tablename__ = "incidents"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    # Link to the PipelineRun that triggered this incident
    run_id           = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
                              nullable=True, unique=True, index=True)
    pipeline_name    = Column(String(512), nullable=False)
    status           = Column(Enum(IncidentStatus), default=IncidentStatus.DETECTED)
    risk_tier        = Column(String(16), default="Medium")   # Low / Medium / High
    detected_at      = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at      = Column(DateTime, nullable=True)

    # Diagnostic data
    error_log        = Column(Text,    default="")
    failed_node      = Column(String(255), nullable=True)
    root_cause       = Column(Text,    nullable=True)
    proposed_action  = Column(Text,    nullable=True)
    agent_thought    = Column(Text,    nullable=True)
    remediation_plan = Column(JSON,    default=list)    # ["step 1", "step 2", …]
    similar_incidents = Column(JSON,   default=list)   # list of memory entry ids
    confidence_score = Column(Float,   nullable=True)
    tool_calls       = Column(JSON,    default=list)
    timeline         = Column(JSON,    default=list)   # [{ts,stage,agent,detail}]

    approval_required = Column(Boolean, default=False)
    approved_by       = Column(String(255), nullable=True)
    approved_at       = Column(DateTime, nullable=True)

    # ─── NEW: email-dispatch lifecycle ────────────────────────────────────
    # Filled in by incident_notifier when the initial mail is sent and again
    # when the escalation mail fires. Surfaced to the frontend so the
    # Incident Timeline page can show real timestamps, not estimates.

    # First (level-1) notification — single recipient (the first DataOps Eng.)
    initial_email_sent_at    = Column(DateTime, nullable=True)
    initial_email_recipient  = Column(String(255), nullable=True)  # email address
    initial_email_role       = Column(String(64),  nullable=True)  # e.g. "DataOps Engineer"

    # Escalation notification — list of recipients with their roles, e.g.
    # [{"email": "lead@…", "role": "Data Platform Lead"}, …]
    escalation_email_sent_at    = Column(DateTime, nullable=True)
    escalation_email_recipients = Column(JSON,     default=list)

    # ─── Pipeline-level tracking (scheduler-driven escalation) ────────────
    pipeline_id         = Column(Integer, ForeignKey("pipelines.id", ondelete="CASCADE"),
                                 nullable=True, index=True)
    is_active           = Column(Boolean, default=True, index=True)
    escalation_count    = Column(Integer, default=0)
    last_escalation_at  = Column(DateTime, nullable=True)
    last_known_run_count = Column(Integer, default=0)

    # ─── Jira Integration ─────────────────────────────────────────────────
    jira_ticket_key     = Column(String(255), nullable=True)
    jira_ticket_url     = Column(String(512), nullable=True)


# ---------------------------------------------------------------------------
# Incident Event (journey log)
# ---------------------------------------------------------------------------

class IncidentEvent(Base):
    __tablename__ = "incident_events"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    incident_id      = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"),
                              nullable=False, index=True)
    event_type       = Column(String(50), nullable=False)  # IncidentEventType value
    escalation_level = Column(String(16), nullable=True)    # L1, L1_L2_L3
    recipients       = Column(JSON, default=list)           # [{email, role}]
    related_run_id   = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
                              nullable=True)
    details          = Column(Text, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow, index=True)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    kind             = Column(String(32), nullable=False, index=True)
    title            = Column(String(512), nullable=False)
    summary          = Column(Text, nullable=False)
    payload          = Column(JSON, default=dict)
    tags             = Column(JSON, default=list)
    success          = Column(Boolean, nullable=True)
    times_referenced = Column(Integer, default=0)
    created_at       = Column(DateTime, default=datetime.utcnow, index=True)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class Recommendation(Base):
    __tablename__ = "recommendations"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_id   = Column(String(64),  nullable=True)
    pipeline_name = Column(String(512), nullable=False)
    title         = Column(String(512), nullable=False)
    detail        = Column(Text,        nullable=False)
    savings       = Column(String(255), default="")
    risk          = Column(String(16),  default="Low")   # Low / Medium / High
    status        = Column(String(32),  default="open")  # open / accepted / dismissed
    created_at    = Column(DateTime,    default=datetime.utcnow, index=True)


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ts          = Column(DateTime, default=datetime.utcnow, index=True)
    type        = Column(String(32), default="info")   # info/warn/error/agent/tool
    msg         = Column(Text, nullable=False)
    agent_role  = Column(String(64),  nullable=True)
    incident_id = Column(String(64),  nullable=True)
    actor       = Column(String(255), nullable=True)
