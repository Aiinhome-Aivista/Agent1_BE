"""
API routes for the five agent-loop sections:
  GET/DELETE /incidents          — Incident Loop
  POST       /incidents/{id}/approve|reject
  GET        /agents             — Agent Mesh
  GET        /tools
  GET/POST   /memory             — Memory
  GET        /memory/search
  GET/POST   /recommendations    — Optimize (Recommendations)
  POST       /recommendations/{id}
  GET        /audit              — Audit Trail
  POST       /audit/write        — internal helper (sync_service logs)

Hotfixes applied here vs. the previous version:
  * Removed a duplicate `reject_incident` route (the second one shadowed
    the first and was sync, breaking the async broadcast).
  * Use the `mistral_service` SINGLETON instead of calling
    `MistralService.analyze_failure(...)` as if it were a classmethod.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models import User
from app.models.agent_models import (
    AuditLog, Incident, IncidentStatus, MemoryEntry, Recommendation,
)
from app.schemas.agent_schemas import (
    AgentStatusOut, AuditLogOut, IncidentOut, MemoryEntryCreate,
    MemoryEntryOut, RecommendationOut, RecommendationUpdate, ToolSpecOut,
)
from app.services.mistral_service import mistral_service
from app.websockets.manager import manager

router = APIRouter(tags=["agent-loop"])


# ---------------------------------------------------------------------------
# Agent-status in-memory store (refreshed by websocket events / sync)
# ---------------------------------------------------------------------------

_AGENT_DEFINITIONS: list[dict[str, Any]] = [
    {"role": "orchestrator", "name": "Orchestrator",  "description": "Dispatches sub-agents and manages the incident lifecycle", "color": "slate"},
    {"role": "monitoring",   "name": "Monitoring",    "description": "Watches pipeline runs for anomalies and new failures",       "color": "blue"},
    {"role": "diagnosis",    "name": "Diagnosis",     "description": "Queries Mistral to identify root causes via logs + RAG",     "color": "purple"},
    {"role": "remediation",  "name": "Remediation",   "description": "Applies or proposes fixes against the connector API",        "color": "amber"},
    {"role": "optimization", "name": "Optimization",  "description": "Generates performance and reliability recommendations",      "color": "emerald"},
    {"role": "learning",     "name": "Learning",      "description": "Writes resolved incidents to episodic memory (RAG store)",   "color": "indigo"},
]

_TOOL_REGISTRY: list[dict[str, Any]] = [
    {"name": "retry_run",         "description": "Re-trigger a failed pipeline run",                     "args_schema": {"run_id": "str"},                          "risk": "medium"},
    {"name": "quarantine_run",    "description": "Mark a run as quarantined to prevent cascading failures", "args_schema": {"run_id": "str"},                        "risk": "high"},
    {"name": "get_run_logs",      "description": "Fetch full log output for a run",                      "args_schema": {"run_id": "str", "tail": "int"},            "risk": "low"},
    {"name": "diagnose_run",      "description": "Call Mistral LLM for root-cause analysis",             "args_schema": {"run_id": "str", "force": "bool"},          "risk": "low"},
    {"name": "patch_config",      "description": "Apply a config patch suggested by the LLM",            "args_schema": {"pipeline_id": "str", "patch": "str"},      "risk": "high"},
    {"name": "notify_slack",      "description": "Send an incident alert to a Slack channel",            "args_schema": {"channel": "str", "message": "str"},        "risk": "low"},
    {"name": "create_jira_ticket","description": "Open a Jira ticket for a failed incident",             "args_schema": {"summary": "str", "description": "str"},    "risk": "low"},
    {"name": "memory_search",     "description": "RAG search across episodic/procedural memory",         "args_schema": {"query": "str", "kind": "str", "k": "int"}, "risk": "low"},
    {"name": "memory_write",      "description": "Persist a resolution to episodic memory",              "args_schema": {"title": "str", "summary": "str"},          "risk": "low"},
    {"name": "runbook_search",    "description": "Vector search across uploaded runbooks",               "args_schema": {"query": "str", "k": "int"},                "risk": "low"},
]

_agent_states: dict[str, dict[str, Any]] = {
    d["role"]: {"status": "idle", "last_action": "", "tasks_completed": 0}
    for d in _AGENT_DEFINITIONS
}


def _agent_out(d: dict) -> AgentStatusOut:
    s = _agent_states.get(d["role"], {})
    return AgentStatusOut(
        role=d["role"],
        name=d["name"],
        description=d["description"],
        color=d["color"],
        status=s.get("status", "idle"),
        last_action=s.get("last_action", ""),
        tasks_completed=s.get("tasks_completed", 0),
    )


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

@router.get("/incidents", response_model=list[IncidentOut])
def list_incidents(
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return (
        db.query(Incident)
        .order_by(desc(Incident.detected_at))
        .limit(limit)
        .all()
    )


@router.get("/incidents/{incident_id}", response_model=IncidentOut)
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(404, "Incident not found")
    return inc


@router.post("/incidents/trigger")
async def trigger_synthetic_incident(
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
):
    """Inject a synthetic incident so the full agent loop can be watched live."""
    from app.services.incident_service import run_synthetic_incident     # noqa: PLC0415
    background_tasks.add_task(run_synthetic_incident)
    return {"ok": True, "msg": "Synthetic incident triggered — watch the Incident Loop page"}


@router.post("/incidents/{incident_id}/approve", response_model=IncidentOut)
async def approve_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(404, "Incident not found")
    if inc.status != IncidentStatus.AWAITING_APPROVAL:
        raise HTTPException(400, "Incident is not awaiting approval")

    inc.approved_by = user.email
    inc.approved_at = datetime.utcnow()

    tl = list(inc.timeline or [])
    tl.append({
        "ts": datetime.utcnow().isoformat(),
        "stage": "Executing",
        "agent": "orchestrator",
        "detail": f"Approved by {user.email} — executing remediation plan",
    })
    tl.append({
        "ts": datetime.utcnow().isoformat(),
        "stage": "Remediated",
        "agent": "orchestrator",
        "detail": "Fix approved and execution recorded",
    })
    inc.timeline = tl
    inc.resolved_at = datetime.utcnow()
    inc.status = IncidentStatus.REMEDIATED
    db.commit()
    db.refresh(inc)

    from app.services.incident_service import _incident_to_dict          # noqa: PLC0415
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    return inc


@router.post("/incidents/{incident_id}/reject", response_model=IncidentOut)
async def reject_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(404, "Incident not found")

    tl = list(inc.timeline or [])
    tl.append({
        "ts": datetime.utcnow().isoformat(),
        "stage": "Escalated",
        "agent": "orchestrator",
        "detail": f"Rejected by {user.email} — escalated for manual review",
    })
    inc.timeline = tl
    inc.status = IncidentStatus.ESCALATED
    db.commit()
    db.refresh(inc)

    from app.services.incident_service import _incident_to_dict          # noqa: PLC0415
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    return inc


@router.delete("/incidents/{incident_id}", status_code=204)
def delete_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if inc:
        db.delete(inc)
        db.commit()


@router.delete("/incidents")
def bulk_delete_incidents(
    status: str = Query("closed"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Incident)
    if status == "closed":
        q = q.filter(Incident.status.in_([
            IncidentStatus.REMEDIATED, IncidentStatus.ESCALATED, IncidentStatus.FAILED,
        ]))
    elif status == "open":
        q = q.filter(Incident.status.notin_([
            IncidentStatus.REMEDIATED, IncidentStatus.ESCALATED, IncidentStatus.FAILED,
        ]))
    count = q.count()
    q.delete(synchronize_session=False)
    db.commit()
    return {"deleted": count, "status": status}


# ---------------------------------------------------------------------------
# Agents + Tools
# ---------------------------------------------------------------------------

@router.get("/agents", response_model=list[AgentStatusOut])
def list_agents(user: User = Depends(get_current_user)):
    return [_agent_out(d) for d in _AGENT_DEFINITIONS]


@router.get("/tools", response_model=list[ToolSpecOut])
def list_tools(user: User = Depends(get_current_user)):
    return [ToolSpecOut(**t) for t in _TOOL_REGISTRY]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@router.get("/memory", response_model=list[MemoryEntryOut])
def list_memory(
    kind:  str = Query("episodic"),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return (
        db.query(MemoryEntry)
        .filter(MemoryEntry.kind == kind)
        .order_by(desc(MemoryEntry.created_at))
        .limit(limit)
        .all()
    )


@router.get("/memory/search", response_model=list[MemoryEntryOut])
def search_memory(
    q:    str = Query(...),
    kind: str = Query("episodic"),
    k:    int = Query(8, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    pattern = f"%{q}%"
    rows = (
        db.query(MemoryEntry)
        .filter(
            MemoryEntry.kind == kind,
            or_(
                MemoryEntry.title.ilike(pattern),
                MemoryEntry.summary.ilike(pattern),
            ),
        )
        .order_by(desc(MemoryEntry.created_at))
        .limit(k)
        .all()
    )
    out: list[MemoryEntryOut] = []
    for row in rows:
        m = MemoryEntryOut.model_validate(row)
        m.similarity = 1.0 if q.lower() in (row.title or "").lower() else 0.75
        out.append(m)
    return out


@router.post("/memory", response_model=MemoryEntryOut, status_code=201)
def create_memory(
    body: MemoryEntryCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    entry = MemoryEntry(**body.model_dump())
    db.add(entry); db.commit(); db.refresh(entry)
    return entry


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@router.get("/recommendations", response_model=list[RecommendationOut])
def list_recommendations(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return (
        db.query(Recommendation)
        .order_by(desc(Recommendation.created_at))
        .limit(100)
        .all()
    )


@router.post("/recommendations/regenerate")
def regenerate_recommendations(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Ask Mistral to produce fresh recommendations from recent incidents."""
    from app.models.agent_models import Incident as _Inc
    recent_incidents = (
        db.query(_Inc)
        .filter(_Inc.root_cause.isnot(None))
        .order_by(desc(_Inc.detected_at))
        .limit(10)
        .all()
    )

    if not recent_incidents:
        return {"count": 0, "items": []}

    incident_summaries = "\n".join(
        f"- Pipeline: {i.pipeline_name}, Risk: {i.risk_tier}, "
        f"Cause: {i.root_cause or 'unknown'}"
        for i in recent_incidents
    )
    prompt = (
        "Given these recent pipeline incidents, produce 3 optimisation recommendations.\n"
        f"{incident_summaries}\n\n"
        "Return JSON: {\"recommendations\": [{\"pipeline_name\":\"...\",\"title\":\"...\","
        "\"detail\":\"...\",\"savings\":\"...\",\"risk\":\"Low|Medium|High\"}]}"
    )

    # Use the singleton, not the class
    result = mistral_service.analyze_failure(
        "system", "recommendations", None, [], {"prompt": prompt}
    )
    raw = result.get("raw_response", {})

    items: list[Recommendation] = []
    for r in (raw.get("recommendations") or [])[:5]:
        rec = Recommendation(
            pipeline_name=r.get("pipeline_name", "general"),
            title=r.get("title", "Untitled")[:512],
            detail=r.get("detail", "")[:4000],
            savings=r.get("savings", "")[:255],
            risk=r.get("risk", "Low"),
            status="open",
        )
        db.add(rec); items.append(rec)
    db.commit()
    for rec in items:
        db.refresh(rec)

    return {
        "count": len(items),
        "items": [RecommendationOut.model_validate(r) for r in items],
    }


@router.post("/recommendations/{rec_id}", response_model=RecommendationOut)
def update_recommendation(
    rec_id: int,
    body: RecommendationUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rec = db.query(Recommendation).filter(Recommendation.id == rec_id).first()
    if not rec:
        raise HTTPException(404, "Recommendation not found")
    rec.status = body.status
    db.commit(); db.refresh(rec)
    return rec


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------

@router.get("/audit", response_model=list[AuditLogOut])
def list_audit(
    limit: int = Query(300, le=1000),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return (
        db.query(AuditLog)
        .order_by(desc(AuditLog.ts))
        .limit(limit)
        .all()
    )
