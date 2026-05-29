"""
Runbooks REST API.

Endpoints:
  POST   /runbooks/upload     - upload a PDF / DOCX / MD / TXT file,
                                store on disk, chunk + embed into Chroma
  GET    /runbooks            - list runbooks (excludes ARCHIVED by default)
  GET    /runbooks/{id}       - fetch a single runbook
  DELETE /runbooks/{id}       - hard-delete + remove its Chroma chunks
  POST   /runbooks/{id}/archive - soft-archive
  GET    /runbooks/{id}/download - stream the original uploaded file
  POST   /runbooks/search     - vector-search runbook chunks (debug / test)
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException,
    UploadFile, Query,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models import User
from app.models.runbook import Runbook, RunbookSource, RunbookStatus
from app.schemas.runbook import (
    RunbookOut, RunbookSearchHit, RunbookSearchResponse,
)
from app.services.document_service import SUPPORTED_EXTS, document_service
from app.services.embedding_service import embedding_service
from app.services.llm_runbook_service import (
    ALLOWED_CATEGORIES,
    suggest_metadata as llm_suggest_runbook_metadata,
)
from app.services.rag_service import rag_service
from app.services.vector_service import get_vector_service
from app.services.graph_enrichment_service import enrich_runbook

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runbooks", tags=["runbooks"])


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

_EXT_TO_SOURCE = {
    ".pdf":  RunbookSource.PDF,
    ".docx": RunbookSource.DOCX,
    ".md":   RunbookSource.MARKDOWN,
    ".txt":  RunbookSource.TXT,
}

MAX_BYTES = 50 * 1024 * 1024  # 25 MB cap per file


def _safe_filename(name: str) -> str:
    """Avoid collisions and path traversal."""
    base = Path(name).name
    return f"{uuid.uuid4().hex[:8]}_{base}"


def _ingest_runbook(runbook_id: int) -> None:
    """
    Background task: read the file from disk, chunk it, embed each chunk,
    upsert into Chroma, then update the DB row.

    Runs in a fresh DB session because background tasks can outlive the
    request that scheduled them.
    """
    from app.core.database import SessionLocal           # noqa: PLC0415
    db: Session = SessionLocal()
    try:
        rb = db.query(Runbook).filter(Runbook.id == runbook_id).first()
        if not rb or not rb.storage_path:
            return

        try:
            text = document_service.extract_text(rb.storage_path)
            if not text.strip():
                raise ValueError("Document is empty or unreadable")

            chunks = document_service.chunk(text)
            if not chunks:
                raise ValueError("Document produced 0 chunks")

            embeddings = embedding_service.embed_batch(chunks)
            vector_service = get_vector_service()

            ids       : list[str]  = []
            metadatas : list[dict] = []
            for i, ch in enumerate(chunks):
                ids.append(f"rb-{rb.id}-chunk-{i}")
                metadatas.append({
                    "runbook_id":      str(rb.id),
                    "title":           rb.title,
                    "category":        rb.category,
                    "source":          rb.source.value if hasattr(rb.source, "value") else str(rb.source),
                    "source_filename": rb.source_filename or "",
                    "chunk_index":     i,
                    "risk_level":      rb.risk_level,
                })

            vector_service.add_runbook_chunks(
                ids=ids,
                texts=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            rb.chunk_count  = len(chunks)
            rb.status       = RunbookStatus.ACTIVE
            rb.ingest_error = None
            db.commit()
            logger.info("Runbook %s ingested: %d chunks", rb.id, len(chunks))

            # NEW: build the graph subgraph for this runbook
            enrich_runbook(
                runbook_id  = rb.id,
                title       = rb.title,
                category    = rb.category,
                text        = text,
                risk_level  = rb.risk_level,
                description = rb.description,
            )

        except Exception as e:
            logger.exception("Runbook %s ingest failed", rb.id)
            rb.status       = RunbookStatus.FAILED
            rb.ingest_error = str(e)[:1000]
            db.commit()

    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────────

class RunbookSuggestion(BaseModel):
    """Returned by POST /runbooks/analyze — what the LLM thinks the file is.

    The frontend pre-fills its form from this and lets the user edit before
    committing via POST /runbooks/upload."""
    title:       str
    category:    str
    description: str
    steps:       list[str]
    risk_level:  str
    tags:        list[str]
    relevance_score: int = 50 
    llm_used:    bool = True
    model:       Optional[str] = None
    latency_ms:  Optional[float] = None
    extracted_chars: int = 0


@router.post("/analyze", response_model=RunbookSuggestion)
async def analyze_runbook(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """
    PHASE-1 of upload: user picks a file, we extract its text and ask Mistral
    to propose a title / category / description / steps / tags. Nothing is
    persisted yet — the user gets a chance to review and edit before they
    commit via POST /upload.
    """
    if not file or not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            400, f"Unsupported file type {ext!r}. Allowed: {sorted(SUPPORTED_EXTS)}"
        )

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_BYTES // (1024*1024)} MB limit")
    if not raw:
        raise HTTPException(400, "Empty file")

    # Write to a *temp* path so document_service can read it. We delete it
    # right after — the commit step (POST /upload) will receive the file
    # again from the browser, and that's the copy we persist for the long
    # haul.
    tmp_path = Path(settings.RUNBOOKS_DIR) / f"_analyze_{uuid.uuid4().hex[:10]}{ext}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(raw)

    try:
        try:
            text = document_service.extract_text(str(tmp_path))
        except Exception as e:
            logger.exception("Text extraction failed during analyze")
            raise HTTPException(422, f"Could not read document text: {e}")

        if not text or not text.strip():
            raise HTTPException(422, "Document appears to be empty or unreadable (scanned-image PDF?)")

        suggestion = llm_suggest_runbook_metadata(text, filename=file.filename)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return RunbookSuggestion(
        title       = suggestion["title"],
        category    = suggestion["category"],
        description = suggestion["description"],
        steps       = suggestion["steps"],
        risk_level  = suggestion["risk_level"],
        tags        = suggestion["tags"],
        relevance_score = suggestion.get("relevance_score", 50),
        llm_used    = suggestion.get("llm_used", True),
        model       = suggestion.get("model"),
        latency_ms  = suggestion.get("latency_ms"),
        extracted_chars = len(text),
    )


@router.post("/upload", response_model=RunbookOut, status_code=201)
async def upload_runbook(
    background_tasks: BackgroundTasks,
    file:        UploadFile = File(...),
    title:       Optional[str] = Form(None),
    category:    str  = Form("ADF"),
    description: str  = Form(""),
    risk_level:  str  = Form("Medium"),
    tags_csv:    str  = Form(""),
    rag_enabled: bool = Form(True),
    relevance_score: int = Form(50), 
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    """PHASE-2 of upload: user has reviewed the LLM-suggested metadata
    (or filled it in by hand) and is committing. We store the file and
    kick off background vector ingestion."""

    if not file or not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            400, f"Unsupported file type {ext!r}. Allowed: {sorted(SUPPORTED_EXTS)}"
        )

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_BYTES // (1024*1024)} MB limit")
    if not raw:
        raise HTTPException(400, "Empty file")

    safe_name = _safe_filename(file.filename)
    out_path  = Path(settings.RUNBOOKS_DIR) / safe_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)

    tags = [t.strip() for t in (tags_csv or "").split(",") if t.strip()]

    # Force the category to one of the four allowed values. Anything else
    # (legacy "Airflow", "General", etc.) gets snapped to "ADF" so we don't
    # silently keep junk values in the column.
    safe_category = category if category in ALLOWED_CATEGORIES else "ADF"

    rb = Runbook(
        title=title or Path(file.filename).stem,
        category=safe_category,
        description=description,
        source=_EXT_TO_SOURCE.get(ext, RunbookSource.TXT),
        source_filename=file.filename,
        storage_path=str(out_path),
        size_bytes=len(raw),
        risk_level=risk_level if risk_level in {"Low", "Medium", "High"} else "Medium",
        tags=tags,
        rag_enabled=bool(rag_enabled),
        status=RunbookStatus.PROCESSING,
        uploaded_by=user.email if user else None,
        relevance_score=max(0, min(100, relevance_score)),
    )
    db.add(rb)
    db.commit()
    db.refresh(rb)

    background_tasks.add_task(_ingest_runbook, rb.id)
    return rb


# ──────────────────────────────────────────────────────────────────
# List / get / archive / delete
# ──────────────────────────────────────────────────────────────────

@router.get("", response_model=list[RunbookOut])
def list_runbooks(
    include_archived: bool = Query(False),
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    q = db.query(Runbook)
    if not include_archived:
        q = q.filter(Runbook.status != RunbookStatus.ARCHIVED)
    return q.order_by(desc(Runbook.created_at)).all()


@router.get("/{rb_id}", response_model=RunbookOut)
def get_runbook(
    rb_id: int,
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    rb = db.query(Runbook).filter(Runbook.id == rb_id).first()
    if not rb:
        raise HTTPException(404, "Runbook not found")
    return rb


@router.post("/{rb_id}/archive", response_model=RunbookOut)
def archive_runbook(
    rb_id: int,
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    rb = db.query(Runbook).filter(Runbook.id == rb_id).first()
    if not rb:
        raise HTTPException(404, "Runbook not found")
    rb.status = RunbookStatus.ARCHIVED
    db.commit()
    db.refresh(rb)
    return rb


@router.delete("/{rb_id}", status_code=204)
def delete_runbook(
    rb_id: int,
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    rb = db.query(Runbook).filter(Runbook.id == rb_id).first()
    if not rb:
        return

    # Remove vector chunks
    try:
        vector_service = get_vector_service()
        vector_service.delete_runbook(str(rb.id))
    except Exception:
        logger.warning("Vector cleanup failed for runbook %s", rb.id, exc_info=True)

    # Remove file on disk (best-effort)
    if rb.storage_path:
        try:
            Path(rb.storage_path).unlink(missing_ok=True)
        except Exception:
            logger.warning("File delete failed for %s", rb.storage_path, exc_info=True)

    db.delete(rb)
    db.commit()


@router.get("/{rb_id}/download")
def download_runbook(
    rb_id: int,
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    rb = db.query(Runbook).filter(Runbook.id == rb_id).first()
    if not rb or not rb.storage_path:
        raise HTTPException(404, "File not available")

    p = Path(rb.storage_path)
    if not p.exists():
        raise HTTPException(404, "File missing on disk")

    return FileResponse(
        str(p),
        filename=rb.source_filename or p.name,
        media_type="application/octet-stream",
    )


# ──────────────────────────────────────────────────────────────────
# Search (debug / test)
# ──────────────────────────────────────────────────────────────────

class _SearchBody(BaseModel):
    query: str
    k: int = 5


@router.post("/search", response_model=RunbookSearchResponse)
def search_runbooks(
    body: _SearchBody,
    user: User = Depends(get_current_user),
):
    t0 = time.perf_counter()
    raw = rag_service.search_runbooks(body.query, k=body.k)

    hits = [
        RunbookSearchHit(
            runbook_id=int((r.get("metadata") or {}).get("runbook_id") or 0) or None,
            title=(r.get("metadata") or {}).get("title"),
            chunk_index=int((r.get("metadata") or {}).get("chunk_index") or 0),
            similarity=r["similarity"],
            snippet=(r.get("document") or "")[:600],
        )
        for r in raw
    ]
    return RunbookSearchResponse(
        query=body.query,
        hits=hits,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
