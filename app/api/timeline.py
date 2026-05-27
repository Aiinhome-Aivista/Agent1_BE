"""
timeline.py - Incident Timeline API

GET /timeline/incidents
    Left-panel list of all pipelines with alert activity.

GET /timeline/incidents/{pipeline_id}?run_id=...
    4-step flow:
      Step 1 — Incident Detection   : pipeline_name, summary, detection_time
      Step 2 — Initial Notification : sent_to (per cycle with timestamp)
      Step 3 — Escalation           : sent_to (per cycle with timestamp, unique per cycle)
      Step 4 — Resolution           : pipeline status (SUCCEEDED / FAILED / etc.)
"""

from datetime import datetime
from collections import defaultdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models import User, PipelineRun, ErrorAnalysis, Pipeline
from app.models.agent_models import Incident, IncidentStatus
from app.models.alert_models import EmailLog

router = APIRouter(prefix="/timeline", tags=["timeline"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class IncidentSummary(BaseModel):
    """One card in the left-panel list."""
    pipeline_id:     int
    pipeline_name:   str
    run_id:          int | None
    latest_status:   str
    is_escalated:    bool
    last_activity:   datetime
    cycle_count:     int
    pipeline_status: str | None
    incident_status: str | None
    resolved_at:     datetime | None
    model_config = ConfigDict(from_attributes=False)


class RecipientOut(BaseModel):
    email:  str
    role:   str
    status: str        # "sent" | "failed" | "skipped"
    model_config = ConfigDict(from_attributes=False)


class CycleOut(BaseModel):
    """One alert cycle — timestamp + who was emailed at that moment."""
    cycle_number: int
    sent_at:      datetime
    recipients:   list[RecipientOut]
    model_config = ConfigDict(from_attributes=False)


# ── Per-step schemas (each step returns only what it needs) ───────────────────

class Step1Out(BaseModel):
    """Incident Detection"""
    step:           Literal[1]
    title:          Literal["Incident Detection"]
    pipeline_name:  str
    summary:        str | None     # from ErrorAnalysis
    detection_time: datetime
    model_config = ConfigDict(from_attributes=False)


class Step2Out(BaseModel):
    """Initial Notification"""
    step:   Literal[2]
    title:  Literal["Initial Notification"]
    cycles: list[CycleOut]         # each cycle: timestamp + recipients
    model_config = ConfigDict(from_attributes=False)


class Step3Out(BaseModel):
    """Escalation"""
    step:          Literal[3]
    title:         Literal["Escalation"]
    status:        str             # "done" | "waiting" | "skipped"
    new_run_found: bool | None
    cycles:        list[CycleOut]  # each cycle: timestamp + unique recipients
    model_config = ConfigDict(from_attributes=False)


class Step4Out(BaseModel):
    """Resolution"""
    step:            Literal[4]
    title:           Literal["Resolution"]
    pipeline_status: str | None    # SUCCEEDED / FAILED / RUNNING etc.
    incident_status: str | None    # Remediated / Escalated / etc.
    resolved_at:     datetime | None
    model_config = ConfigDict(from_attributes=False)


class IncidentTimelineOut(BaseModel):
    pipeline_id:   int
    pipeline_name: str
    run_id:        int | None
    is_escalated:  bool
    step1:         Step1Out
    step2:         Step2Out
    step3:         Step3Out
    step4:         Step4Out
    model_config = ConfigDict(from_attributes=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedup_recipients(rows) -> list[RecipientOut]:
    """Unique emails — prefer 'sent' > 'failed' > 'skipped'."""
    priority = {"sent": 0, "failed": 1, "skipped": 2}
    seen: dict[str, dict] = {}
    for r in rows:
        email = r.recipient_email
        if email not in seen or priority.get(r.status, 9) < priority.get(seen[email]["status"], 9):
            seen[email] = {"email": email, "role": r.recipient_role, "status": r.status}
    return [RecipientOut(**v) for v in seen.values()]


def _build_cycles(rows, exclude_status: str | None = None) -> list[CycleOut]:
    """
    Group rows by cycle_number (ascending).
    Each cycle gets the earliest sent_at of that cycle and deduplicated recipients.
    """
    buckets: dict[int, dict] = defaultdict(lambda: {"sent_at": None, "rows": []})
    for r in rows:
        if exclude_status and r.status == exclude_status:
            continue
        cn = r.cycle_number or 1
        buckets[cn]["rows"].append(r)
        if buckets[cn]["sent_at"] is None or r.sent_at < buckets[cn]["sent_at"]:
            buckets[cn]["sent_at"] = r.sent_at

    return [
        CycleOut(
            cycle_number = cn,
            sent_at      = buckets[cn]["sent_at"],
            recipients   = _dedup_recipients(buckets[cn]["rows"]),
        )
        for cn in sorted(buckets.keys())
    ]


def _fetch_run(db: Session, run_id: int | None):
    if run_id is None:
        return None
    return (
        db.query(PipelineRun)
          .options(joinedload(PipelineRun.analysis))
          .filter(PipelineRun.id == run_id)
          .first()
    )


def _fetch_incident(db: Session, run_id: int | None):
    if run_id is None:
        return None
    return db.query(Incident).filter(Incident.run_id == run_id).first()


def _fetch_pipeline(db: Session, pipeline_id: int):
    return db.query(Pipeline).filter(Pipeline.id == pipeline_id).first()


def _get_summary(run) -> str | None:
    if run is None:
        return None
    analysis = getattr(run, "analysis", None)
    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else None
    return getattr(analysis, "summary", None) if analysis else None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/incidents", response_model=list[IncidentSummary])
def list_incidents(
    db:   Session = Depends(get_db),
    user: User    = Depends(get_current_user),
):
    rows: list[EmailLog] = (
        db.query(EmailLog)
          .order_by(EmailLog.sent_at.desc())
          .all()
    )

    seen: dict[tuple, dict] = {}
    for row in rows:
        key = (row.pipeline_id, row.run_id)
        if key not in seen:
            seen[key] = {
                "pipeline_id":   row.pipeline_id,
                "pipeline_name": row.pipeline_name,
                "run_id":        row.run_id,
                "latest_status": row.status,
                "is_escalated":  row.email_type == "escalation" and row.status != "skipped",
                "last_activity": row.sent_at,
                "cycle_count":   row.cycle_number,
            }
        else:
            entry = seen[key]
            if row.sent_at > entry["last_activity"]:
                entry["last_activity"] = row.sent_at
                entry["latest_status"] = row.status
            if row.cycle_number > entry["cycle_count"]:
                entry["cycle_count"] = row.cycle_number
            if row.email_type == "escalation" and row.status != "skipped":
                entry["is_escalated"] = True

    results = sorted(seen.values(), key=lambda x: x["last_activity"], reverse=True)

    enriched = []
    for r in results:
        pipe     = _fetch_pipeline(db, r["pipeline_id"])
        incident = _fetch_incident(db, r["run_id"])
        r["pipeline_status"] = pipe.last_run_status.value if pipe and pipe.last_run_status else None
        r["incident_status"] = incident.status.value      if incident else None
        r["resolved_at"]     = (
            incident.resolved_at
            if incident and incident.status.value == "Remediated" else None
        )
        enriched.append(IncidentSummary(**r))
    return enriched


@router.get("/incidents/{pipeline_id}", response_model=IncidentTimelineOut)
def get_incident_timeline(
    pipeline_id: int,
    run_id:      int | None = None,
    db:          Session    = Depends(get_db),
    user:        User       = Depends(get_current_user),
):
    q = db.query(EmailLog).filter(EmailLog.pipeline_id == pipeline_id)
    if run_id is not None:
        q = q.filter(EmailLog.run_id == run_id)
    rows: list[EmailLog] = q.order_by(EmailLog.sent_at.asc()).all()

    if not rows:
        raise HTTPException(404, "No timeline data found for this pipeline")

    pipeline_name    = rows[0].pipeline_name
    effective_run_id = run_id or rows[0].run_id

    run      = _fetch_run(db, effective_run_id)
    incident = _fetch_incident(db, effective_run_id)
    pipe     = _fetch_pipeline(db, pipeline_id)

    pipeline_status = pipe.last_run_status.value if pipe and pipe.last_run_status else None
    incident_status = incident.status.value      if incident else None

    initial_rows    = [r for r in rows if r.email_type == "initial"]
    escalation_rows = [r for r in rows if r.email_type == "escalation"]
    is_escalated    = any(r.status != "skipped" for r in escalation_rows)

    # ── Step 1: Incident Detection ────────────────────────────────────────────
    step1 = Step1Out(
        step           = 1,
        title          = "Incident Detection",
        pipeline_name  = pipeline_name,
        summary        = _get_summary(run),
        detection_time = rows[0].initial_sent_at or rows[0].sent_at,
    )

    # ── Step 2: Initial Notification ─────────────────────────────────────────
    all_initial_cycles = _build_cycles(initial_rows)
    step2 = Step2Out(
        step   = 2,
        title  = "Initial Notification",
        cycles = all_initial_cycles[:1],   # show only cycle 1
    )

    # ── Step 3: Escalation ────────────────────────────────────────────────────
    new_run_found = escalation_rows[0].new_run_found if escalation_rows else None
    if not escalation_rows:
        esc_status = "waiting"
    elif new_run_found:
        esc_status = "skipped"
    else:
        esc_status = "done"

    step3 = Step3Out(
        step          = 3,
        title         = "Escalation",
        status        = esc_status,
        new_run_found = new_run_found,
        cycles        = _build_cycles(escalation_rows, exclude_status="skipped"),
    )

    # ── Step 4: Resolution ────────────────────────────────────────────────────
    step4 = Step4Out(
        step            = 4,
        title           = "Resolution",
        pipeline_status = pipeline_status,
        incident_status = incident_status,
        resolved_at     = (
            incident.resolved_at
            if incident and incident_status == "Remediated" else None
        ),
    )

    return IncidentTimelineOut(
        pipeline_id   = pipeline_id,
        pipeline_name = pipeline_name,
        run_id        = effective_run_id,
        is_escalated  = is_escalated,
        step1         = step1,
        step2         = step2,
        step3         = step3,
        step4         = step4,
    )