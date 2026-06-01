"""
Knowledge-base schedule + manual-refresh + human-fix-enrichment API.

  GET  /kb/settings                  - current daily-refresh schedule (from SQL)
  PUT  /kb/settings                  - set enabled + daily time ("HH:MM" UTC)
  GET  /kb/status                    - last-run info + KB counts
  POST /kb/refresh                   - run the consolidation now
  POST /incidents/{id}/update-fix    - human edits/approves the fix → the KB and
                                       knowledge graph are enhanced from old
                                       errors, history, runbooks and this fix
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models import User
from app.models.agent_models import Incident
from app.models.kb_settings import KBSettings, get_or_create_settings
from app.services import error_signature as sig
from app.services import kb_refresh_service
from app.services.solution_kb_service import solution_kb_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge-base"])


def _settings_dict(row: KBSettings) -> dict[str, Any]:
    summary = None
    if row.last_run_summary:
        try:
            summary = json.loads(row.last_run_summary)
        except Exception:
            summary = None
    return {
        "daily_refresh_enabled": row.daily_refresh_enabled,
        "daily_refresh_time": row.daily_refresh_time,
        "timezone": "UTC",
        "last_run_at": row.last_run_at.isoformat() if row.last_run_at else None,
        "last_run_summary": summary,
    }


@router.get("/kb/settings")
def get_kb_settings(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return _settings_dict(get_or_create_settings(db))


class KBSettingsUpdate(BaseModel):
    daily_refresh_enabled: Optional[bool] = None
    daily_refresh_time: Optional[str] = None  # "HH:MM"

    @field_validator("daily_refresh_time")
    @classmethod
    def _check_time(cls, v):
        if v is None:
            return v
        import re
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError("daily_refresh_time must be 'HH:MM' 24-hour")
        return v


@router.put("/kb/settings")
def update_kb_settings(
    body: KBSettingsUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    row = get_or_create_settings(db)
    if body.daily_refresh_enabled is not None:
        row.daily_refresh_enabled = body.daily_refresh_enabled
    if body.daily_refresh_time is not None:
        row.daily_refresh_time = body.daily_refresh_time
    db.commit()
    db.refresh(row)
    return _settings_dict(row)


@router.get("/kb/status")
def kb_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return {
        "settings": _settings_dict(get_or_create_settings(db)),
        "kb": solution_kb_service.stats(db),
    }


@router.post("/kb/refresh")
def kb_refresh_now(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Run the consolidation immediately (history + patterns + runbooks →
    vector store + knowledge graph)."""
    return kb_refresh_service.run_daily_refresh(db)


class UpdateFixBody(BaseModel):
    root_cause: Optional[str] = None
    fix_steps: Optional[list[str]] = None
    approve: bool = True


@router.post("/incidents/{incident_id}/update-fix")
def update_fix(
    incident_id: int,
    body: UpdateFixBody,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """A human reviewed (and possibly modified) the proposed fix. Persist the
    edits onto the incident + its solution pattern, reinforce confidence, and
    enhance the knowledge base + graph from history, runbooks and this fix."""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(404, "Incident not found")

    # Apply human edits to the incident itself.
    if body.root_cause is not None:
        inc.root_cause = body.root_cause
    if body.fix_steps is not None:
        inc.remediation_plan = body.fix_steps
        inc.proposed_action = (body.fix_steps[0] if body.fix_steps else inc.proposed_action)
    db.commit()

    # Resolve the matching pattern.
    signature = sig.compute_signature(inc.error_log or "", component=inc.pipeline_name)
    cls = solution_kb_service.classify(
        db, error_text=inc.error_log or "", component=inc.pipeline_name,
        llm_confidence=inc.confidence_score or 0.0,
    )
    pattern = cls.pattern

    # Reinforce + enrich.
    if body.approve and signature:
        solution_kb_service.reinforce(
            db, signature=signature, accepted=True,
            llm_confidence=inc.confidence_score or 0.0,
        )
        # re-read pattern after reinforcement
        pattern = solution_kb_service.classify(
            db, error_text=inc.error_log or "", component=inc.pipeline_name,
            llm_confidence=inc.confidence_score or 0.0,
        ).pattern

    enrich = kb_refresh_service.enrich_from_approved_fix(
        db,
        incident=inc,
        pattern=pattern,
        curated_root_cause=body.root_cause,
        curated_fix_steps=body.fix_steps,
        approved_by=user.email if user else None,
    )

    return {
        "ok": True,
        "enriched": enrich,
        "confidence": pattern.confidence if pattern else inc.confidence_score,
        "pattern_id": pattern.id if pattern else None,
    }
