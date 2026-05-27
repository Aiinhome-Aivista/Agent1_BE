"""
Pydantic schemas for runbook endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RunbookBase(BaseModel):
    title:        str = Field(..., max_length=512)
    category:     str = Field("General", max_length=128)
    description:  str = ""
    risk_level:   str = Field("Medium", pattern="^(Low|Medium|High)$")
    rag_enabled:  bool = True
    tags:         list[str] = Field(default_factory=list)


class RunbookCreate(RunbookBase):
    """Used when no file is attached – metadata-only entry."""
    pass


class RunbookOut(RunbookBase):
    model_config = ConfigDict(from_attributes=True)

    id:               int
    source:           str
    source_filename:  Optional[str]  = None
    size_bytes:       int            = 0
    chunk_count:      int            = 0
    status:           str
    ingest_error:     Optional[str]  = None
    ai_approved:      bool
    human_verified:   bool
    uploaded_by:      Optional[str]  = None
    created_at:       datetime
    updated_at:       datetime
    relevance_score: int = 50



class RunbookSearchHit(BaseModel):
    runbook_id:   Optional[int]
    title:        Optional[str]
    chunk_index:  Optional[int]
    similarity:   float
    snippet:      str


class RunbookSearchResponse(BaseModel):
    query:   str
    hits:    list[RunbookSearchHit]
    elapsed_ms: float
