"""
Solution Knowledge Base + auto-fix REST API.

  GET  /solutions                  - list solution patterns (the learning KB)
  GET  /solutions/stats            - counts for the metrics page
  GET  /solutions/{id}             - one pattern incl. its code fixes
  POST /solutions/classify         - explain how an error text would classify
  POST /incidents/{id}/raise-pr    - generate a fix and open a real PR
  POST /incidents/{id}/ingest-pr   - ingest a merged human PR back into the KB
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models import User
from app.models.agent_models import Incident
from app.models.solution_models import SolutionFix, SolutionPattern
from app.services import error_signature as sig
from app.services.solution_kb_service import solution_kb_service
from app.services import pr_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["solutions"])


# ── serialization ─────────────────────────────────────────────────────
def _fix_dict(f: SolutionFix) -> dict[str, Any]:
    return {
        "id": f.id,
        "origin": f.origin.value if hasattr(f.origin, "value") else str(f.origin),
        "file_path": f.file_path,
        "has_code": f.has_code,
        "explanation": f.explanation,
        "language": f.language,
        "pr_url": f.pr_url,
        "pr_number": f.pr_number,
        "merged_by": f.merged_by,
        "created_at": f.created_at.isoformat() if f.created_at else None,
    }


def _pattern_dict(p: SolutionPattern, *, include_fixes: bool = False) -> dict[str, Any]:
    d = {
        "id": p.id,
        "signature": p.signature,
        "title": p.title,
        "category": p.category,
        "component": p.component,
        "error_type": p.error_type,
        "support_group": p.support_group,
        "root_cause": p.root_cause,
        "fix_summary": p.fix_summary,
        "fix_steps": p.fix_steps or [],
        "occurrence_count": p.occurrence_count,
        "acceptance_count": p.acceptance_count,
        "rejection_count": p.rejection_count,
        "confidence": p.confidence,
        "status": p.status.value if hasattr(p.status, "value") else str(p.status),
        "is_auto_fixable": p.is_auto_fixable,
        "first_seen_at": p.first_seen_at.isoformat() if p.first_seen_at else None,
        "last_seen_at": p.last_seen_at.isoformat() if p.last_seen_at else None,
        "last_accepted_at": p.last_accepted_at.isoformat() if p.last_accepted_at else None,
    }
    if include_fixes:
        d["fixes"] = [_fix_dict(f) for f in (p.fixes or [])]
    return d


# ── KB browsing ───────────────────────────────────────────────────────
@router.get("/solutions")
def list_solutions(
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    rows = (
        db.query(SolutionPattern)
        .order_by(desc(SolutionPattern.last_seen_at))
        .limit(limit)
        .all()
    )
    return [_pattern_dict(p) for p in rows]


@router.get("/solutions/stats")
def solution_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return solution_kb_service.stats(db)


@router.get("/solutions/{pattern_id}")
def get_solution(
    pattern_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    p = db.query(SolutionPattern).filter(SolutionPattern.id == pattern_id).first()
    if not p:
        raise HTTPException(404, "Solution pattern not found")
    return _pattern_dict(p, include_fixes=True)


class ClassifyBody(BaseModel):
    error_text: str
    component: Optional[str] = None
    llm_confidence: float = 0.0


@router.post("/solutions/classify")
def classify_error(
    body: ClassifyBody,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    c = solution_kb_service.classify(
        db, error_text=body.error_text, component=body.component,
        llm_confidence=body.llm_confidence,
    )
    return {
        "is_known": c.is_known,
        "auto_fix": c.auto_fix,
        "signature": c.signature,
        "error_type": c.error_type,
        "reason": c.reason,
        "pattern": _pattern_dict(c.pattern) if c.pattern else None,
    }


# ── auto-fix actions on an incident ────────────────────────────────────
@router.post("/incidents/{incident_id}/raise-pr")
def raise_pr(
    incident_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(404, "Incident not found")

    # Resolve the pattern for this incident's error (best-effort).
    component = None
    try:
        component = inc.pipeline_name
    except Exception:
        pass
    cls = solution_kb_service.classify(
        db, error_text=inc.error_log or "", component=component,
        llm_confidence=inc.confidence_score or 0.0,
    )
    result = pr_service.raise_pr_for_incident(db, inc, pattern=cls.pattern)

    # If a PR was opened, stamp it on the incident timeline.
    if result.get("ok") and result.get("mode") == "pr" and result.get("pr_url"):
        tl = list(inc.timeline or [])
        tl.append({
            "ts": __import__("datetime").datetime.utcnow().isoformat(),
            "stage": "PR Raised",
            "agent": "remediation",
            "detail": f"Opened PR: {result['pr_url']}",
        })
        inc.timeline = tl
        db.commit()
    return result


class IngestPRBody(BaseModel):
    pr_url: str
    diff: str
    pr_number: Optional[int] = None
    merged_by: Optional[str] = None
    file_path: Optional[str] = None
    new_content: Optional[str] = None
    explanation: str = ""
    # Either give the signature/pattern_id directly, or let us derive it from
    # the incident's error text.
    signature: Optional[str] = None
    pattern_id: Optional[int] = None


@router.post("/incidents/{incident_id}/ingest-pr")
def ingest_pr(
    incident_id: int,
    body: IngestPRBody,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Ingest a human-merged PR back into the KB (the loop-closing step).

    After this, the same error signature becomes auto-fixable: next time it
    occurs the agent reuses this exact change instead of only notifying."""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(404, "Incident not found")

    signature = body.signature
    pattern_id = body.pattern_id
    if not signature and not pattern_id:
        signature = sig.compute_signature(inc.error_log or "", component=inc.pipeline_name)

    pattern = solution_kb_service.ingest_merged_pr(
        db,
        signature=signature,
        pattern_id=pattern_id,
        pr_url=body.pr_url,
        pr_number=body.pr_number,
        merged_by=body.merged_by or (user.email if user else None),
        diff=body.diff,
        file_path=body.file_path,
        new_content=body.new_content,
        explanation=body.explanation,
    )
    if pattern is None:
        raise HTTPException(
            404,
            "No matching solution pattern for this incident — diagnose it first "
            "so a pattern exists, then ingest the PR.",
        )
    return {
        "ok": True,
        "pattern_id": pattern.id,
        "is_auto_fixable": pattern.is_auto_fixable,
        "confidence": pattern.confidence,
        "acceptance_count": pattern.acceptance_count,
    }
