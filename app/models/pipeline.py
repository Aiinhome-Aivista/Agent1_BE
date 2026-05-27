"""
Pipeline domain models.

- Pipeline       : a synced pipeline definition from a connector (ADF pipeline,
                   Databricks job, or Git CI workflow)
- PipelineRun    : a single execution of that pipeline
- PipelineLog    : a log line emitted during a run
- ErrorAnalysis  : LLM-generated diagnosis + fix attached to a failed run
"""
import enum
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, DateTime, Enum, Text, ForeignKey, JSON, Float, Boolean
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class RunStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Pipeline(Base):
    __tablename__ = "pipelines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    connector_id = Column(Integer, ForeignKey("connectors.id", ondelete="CASCADE"), nullable=False)

    # The provider's native ID for this pipeline (ADF pipeline name, Databricks
    # job_id, GitHub workflow id, etc.)
    external_id = Column(String(255), nullable=False, index=True)
    name = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)

    # Provider-specific metadata (raw JSON from the source)
    metadata_json = Column("metadata", JSON, nullable=True, default=dict)

    last_run_status = Column(Enum(RunStatus), nullable=True)
    last_run_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    connector = relationship("Connector", back_populates="pipelines")
    runs = relationship("PipelineRun", back_populates="pipeline",
                        cascade="all, delete-orphan", order_by="PipelineRun.started_at.desc()")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id", ondelete="CASCADE"), nullable=False)

    # Provider's native run id
    external_run_id = Column(String(255), nullable=False, index=True)
    status = Column(Enum(RunStatus), default=RunStatus.UNKNOWN)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    error_message = Column(Text, nullable=True)
    raw_payload = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    pipeline = relationship("Pipeline", back_populates="runs")
    logs = relationship("PipelineLog", back_populates="run",
                        cascade="all, delete-orphan", order_by="PipelineLog.timestamp")
    analysis = relationship("ErrorAnalysis", back_populates="run",
                            uselist=False, cascade="all, delete-orphan")


class PipelineLog(Base):
    __tablename__ = "pipeline_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
                    nullable=False, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(Enum(LogLevel), default=LogLevel.INFO)
    source = Column(String(255), nullable=True)   # e.g. activity name
    message = Column(Text, nullable=False)

    run = relationship("PipelineRun", back_populates="logs")


class ErrorAnalysis(Base):
    __tablename__ = "error_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
                    unique=True, nullable=False)

    # Mistral output
    summary = Column(Text, nullable=False)        # one-line diagnosis
    root_cause = Column(Text, nullable=True)
    suggested_fix = Column(Text, nullable=True)   # human-readable steps
    fix_patch = Column(Text, nullable=True)       # optional code/config patch
    confidence = Column(Float, nullable=True)     # 0..1
    auto_fix_applied = Column(Boolean, default=False)
    auto_fix_result = Column(Text, nullable=True)

    model = Column(String(64), nullable=True)
    raw_response = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("PipelineRun", back_populates="analysis")
