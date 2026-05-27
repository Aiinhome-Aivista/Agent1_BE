"""
Runbook model.

A Runbook is a user-uploaded SOP file (PDF / DOCX / Markdown / TXT) that
has been stored to disk and ingested into the Chroma 'runbooks' collection
in chunks. We keep the file metadata in SQL so the UI can list/delete and
the Chroma chunks reference back via runbook_id.
"""
import enum
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Integer, JSON, String, Text,
)

from app.core.database import Base


class RunbookStatus(str, enum.Enum):
    PROCESSING = "PROCESSING"   # uploaded, ingest in flight
    ACTIVE     = "ACTIVE"       # ingested, indexed in Chroma
    FAILED     = "FAILED"       # ingest blew up
    ARCHIVED   = "ARCHIVED"     # soft-removed by user


class RunbookSource(str, enum.Enum):
    PDF      = "PDF"
    DOCX     = "DOCX"
    MARKDOWN = "MARKDOWN"
    TXT      = "TXT"


class Runbook(Base):
    __tablename__ = "runbooks"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    title         = Column(String(512), nullable=False)
    category      = Column(String(128), default="General")
    description   = Column(Text,        default="")

    source        = Column(Enum(RunbookSource), default=RunbookSource.MARKDOWN)
    source_filename = Column(String(512), nullable=True)
    storage_path  = Column(String(1024), nullable=True)
    size_bytes    = Column(Integer, default=0)

    # Chunk + indexing stats
    chunk_count   = Column(Integer, default=0)
    status        = Column(Enum(RunbookStatus), default=RunbookStatus.PROCESSING)
    ingest_error  = Column(Text, nullable=True)

    risk_level    = Column(String(16), default="Medium")    # Low / Medium / High
    tags          = Column(JSON, default=list)
    rag_enabled   = Column(Boolean, default=True)
    ai_approved   = Column(Boolean, default=True)
    human_verified = Column(Boolean, default=False)

    uploaded_by   = Column(String(255), nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    relevance_score = Column(Integer, default=50) 
