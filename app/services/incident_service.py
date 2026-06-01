"""
Incident service.

NEW PIPELINE (this file is the "before-Mistral" hook):

  logs
   │
   ▼
  vector similarity search over Chroma "incidents" + "runbooks"
   │
   ▼
  build prompt context block (similar past incidents + runbook excerpts)
   │
   ▼
  Mistral analyze_failure(..., context_block=...)
   │
   ▼
  persist Incident + write the new incident back into Chroma "incidents"
  so future failures benefit from this one
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.models import PipelineRun, PipelineLog, Pipeline
from app.models.agent_models import (
    AuditLog, Incident, IncidentEvent, IncidentEventType, IncidentStatus, MemoryEntry, Recommendation,
)
from app.services.mistral_service import mistral_service
from app.services.incident_notifier import (
    notify_incident_diagnosed,
)
from app.services.rag_service import rag_service
from app.websockets.manager import manager
from app.services.jira_service import create_jira_ticket
from app.core.config import settings
from app.services.graph_rag_service import combined_context_block
from app.services.graph_enrichment_service import record_incident_outcome
from app.services.solution_kb_service import solution_kb_service
from app.services.support_routing import route as route_support
from app.services import error_signature as sig
from app.services import pr_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _incident_to_dict(inc: Incident) -> dict:
    return {
        "id": inc.id,
        "run_id": inc.run_id,
        "pipeline_name": inc.pipeline_name,
        "status": inc.status.value if isinstance(inc.status, IncidentStatus) else inc.status,
        "risk_tier": inc.risk_tier,
        "detected_at": inc.detected_at.isoformat() if inc.detected_at else None,
        "resolved_at": inc.resolved_at.isoformat() if inc.resolved_at else None,
        "error_log": inc.error_log,
        "failed_node": inc.failed_node,
        "root_cause": inc.root_cause,
        "proposed_action": inc.proposed_action,
        "agent_thought": inc.agent_thought,
        "remediation_plan": inc.remediation_plan or [],
        "similar_incidents": inc.similar_incidents or [],
        "confidence_score": inc.confidence_score,
        "tool_calls": inc.tool_calls or [],
        "timeline": inc.timeline or [],
        "approval_required": inc.approval_required,
        "approved_by": inc.approved_by,
        "approved_at": inc.approved_at.isoformat() if inc.approved_at else None,
        # ─── NEW: surface email-dispatch lifecycle to the WS clients so the
        # Incident Timeline page updates live as mails go out.
        "initial_email_sent_at":
            inc.initial_email_sent_at.isoformat() if inc.initial_email_sent_at else None,
        "initial_email_recipient":     inc.initial_email_recipient,
        "initial_email_role":          inc.initial_email_role,
        "escalation_email_sent_at":
            inc.escalation_email_sent_at.isoformat() if inc.escalation_email_sent_at else None,
        "escalation_email_recipients": inc.escalation_email_recipients or [],
        # ─── Pipeline-level escalation tracking
        "pipeline_id":          inc.pipeline_id,
        "is_active":            inc.is_active if inc.is_active is not None else True,
        "escalation_count":     inc.escalation_count or 0,
        "last_escalation_at":
            inc.last_escalation_at.isoformat() if inc.last_escalation_at else None,
        "last_known_run_count": inc.last_known_run_count or 0,
        "jira_ticket_key": inc.jira_ticket_key,
        "jira_ticket_url": inc.jira_ticket_url,
    }


async def _broadcast(user_id: int, incident: Incident) -> None:
    try:
        await manager.broadcast({"event": "incident", "payload": _incident_to_dict(incident)})
    except Exception as e:
        logger.debug("broadcast failed: %s", e)


async def _broadcast_log(user_id: int, log: AuditLog) -> None:
    try:
        await manager.broadcast({
            "event": "log",
            "payload": {
                "id": log.id,
                "time": log.ts.strftime("%H:%M:%S") if log.ts else "",
                "msg": log.msg,
                "type": log.type,
                "agent_role": log.agent_role,
                "incident_id": str(log.incident_id) if log.incident_id else None,
            },
        })
    except Exception:
        pass


async def _log(
    db: Session, user_id: int, msg: str, type_: str = "agent",
    agent_role: str | None = None, incident_id: int | None = None,
) -> None:
    entry = AuditLog(
        msg=msg, type=type_, agent_role=agent_role,
        incident_id=str(incident_id) if incident_id else None,
    )
    db.add(entry); db.commit(); db.refresh(entry)
    await _broadcast_log(user_id, entry)


def _set_status(db: Session, inc: Incident, status: IncidentStatus, detail: str) -> None:
    inc.status = status
    tl = list(inc.timeline or [])
    tl.append({
        "ts": datetime.utcnow().isoformat(),
        "stage": status.value,
        "agent": "orchestrator",
        "detail": detail,
    })
    inc.timeline = tl
    db.commit()


def _log_incident_event(
    db: Session,
    incident_id: int,
    event_type: str,
    *,
    escalation_level: str | None = None,
    recipients: list[dict] | None = None,
    related_run_id: int | None = None,
    details: str | None = None,
) -> IncidentEvent:
    """Insert a row into ``incident_events`` (journey table)."""
    evt = IncidentEvent(
        incident_id=incident_id,
        event_type=event_type,
        escalation_level=escalation_level,
        recipients=recipients or [],
        related_run_id=related_run_id,
        details=details,
    )
    db.add(evt)
    db.commit()
    db.refresh(evt)
    return evt


def _risk_from_confidence(confidence: float) -> str:
    if confidence >= 0.7:
        return "Low"
    if confidence >= 0.4:
        return "Medium"
    return "High"


def _parse_plan(suggested_fix: str | None) -> list[str]:
    if not suggested_fix:
        return []
    lines = suggested_fix.strip().splitlines()
    steps: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
                       "1)", "2)", "3)", "4)", "5)", "-", "*", "•"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
                break
        if stripped:
            steps.append(stripped)
    return steps[:8]


# ─── agent-state helpers (unchanged) ────────────────────────────────────────

def _set_agent_state(role: str, status: str, last_action: str = "") -> None:
    try:
        from app.api.agent_api import _agent_states          # noqa: PLC0415
        if role in _agent_states:
            _agent_states[role]["status"] = status
            if last_action:
                _agent_states[role]["last_action"] = last_action
            if status == "idle":
                _agent_states[role]["tasks_completed"] += 1
    except Exception:
        pass


async def _agent_start(role: str, action: str) -> None:
    _set_agent_state(role, "thinking", action)
    await manager.broadcast({
        "event": "agent_started",
        "payload": {"role": role, "name": role.capitalize(), "last_action": action},
    })


async def _agent_done(role: str) -> None:
    _set_agent_state(role, "idle")
    await manager.broadcast({
        "event": "agent_completed",
        "payload": {"role": role, "name": role.capitalize(), "last_action": ""},
    })


# ---------------------------------------------------------------------------
# NEW: pre-flight RAG retrieval
# ---------------------------------------------------------------------------

def _build_rag_context(error_query: str) -> tuple[str, list[dict], list[dict]]:
    """
    Run two parallel vector searches and assemble a single context block
    suitable for injection into the Mistral prompt.

    Returns:
        context_block (str)            - rendered markdown, possibly empty
        similar_incidents (list[dict]) - raw hits for storage on the Incident
        runbook_hits (list[dict])      - raw runbook hits
    """
    if not (error_query or "").strip():
        return "", [], []

    try:
        similar_incidents = rag_service.search_similar_incidents(error_query)
    except Exception:
        logger.exception("similar-incidents search failed")
        similar_incidents = []

    try:
        runbook_hits = rag_service.search_runbooks(error_query)
    except Exception:
        logger.exception("runbook search failed")
        runbook_hits = []

    block = rag_service.build_context_block(similar_incidents, runbook_hits)
    return block, similar_incidents, runbook_hits


# ---------------------------------------------------------------------------
# Main entry point — called from sync_service
# ---------------------------------------------------------------------------

async def process_failed_run(
    db: Session,
    run: PipelineRun,
    user_id: int,
) -> None:
    """Create an Incident for a newly-failed run and drive it through the loop."""

    existing = db.query(Incident).filter(Incident.run_id == run.id).first()
    if existing:
        return

    pipe: Pipeline = run.pipeline
    connector = pipe.connector

    # Count current runs on this pipeline (for the scheduler to compare later)
    current_run_count = (
        db.query(PipelineRun)
        .filter(PipelineRun.pipeline_id == pipe.id)
        .count()
    )

    # 1. DETECTED ---------------------------------------------------------
    await _agent_start("monitoring", f"Detecting failure on {pipe.name}")
    inc = Incident(
        run_id=run.id,
        pipeline_id=pipe.id,
        pipeline_name=pipe.name,
        status=IncidentStatus.DETECTED,
        risk_tier="Medium",
        error_log=run.error_message or "",
        is_active=True,
        escalation_count=0,
        last_known_run_count=current_run_count,
        timeline=[{
            "ts": datetime.utcnow().isoformat(),
            "stage": "Detected",
            "agent": "monitoring",
            "detail": f"Run {run.external_run_id} failed on connector {connector.type.value}",
        }],
    )
    db.add(inc); db.commit(); db.refresh(inc)

    # Log PIPELINE_FAILED event for the journey table
    _log_incident_event(
        db, inc.id,
        IncidentEventType.PIPELINE_FAILED.value,
        related_run_id=run.id,
        details=f"Pipeline '{pipe.name}' run {run.external_run_id} failed: {run.error_message or 'unknown error'}",
    )

    await _broadcast(user_id, inc)
    await _log(db, user_id,
               f"[monitoring] Detected failure on {pipe.name} — run {run.external_run_id}",
               "agent", "monitoring", inc.id)
    await _agent_done("monitoring")

    # 2. RAG RETRIEVAL  (NEW – this is the flow change) -------------------
    await _agent_start("diagnosis", f"Searching vector DB for similar incidents and runbooks")

    logs = (
        db.query(PipelineLog)
        .filter(PipelineLog.run_id == run.id)
        .order_by(PipelineLog.timestamp.asc())
        .all()
    )
    log_dicts: list[dict[str, Any]] = [
        {
            "timestamp": l.timestamp.isoformat() if l.timestamp else "",
            "level":     l.level.value if l.level else "INFO",
            "source":    l.source,
            "message":   l.message,
        }
        for l in logs
    ]

    # Build a single query string from error + the most recent ERROR lines.
    error_lines = [l["message"] for l in log_dicts
                   if l.get("level") in {"ERROR", "CRITICAL"}][-10:]
    rag_query = " ".join(filter(None, [
        run.error_message or "",
        *error_lines,
    ]))[:4000]

    context_block, similar_incidents, runbook_hits = await asyncio.to_thread(
        _build_rag_context, rag_query
    )

    similar_ids = [
        h.get("metadata", {}).get("incident_id") or h.get("id")
        for h in similar_incidents
    ]
    inc.similar_incidents = [s for s in similar_ids if s]

    await _log(
        db, user_id,
        f"[diagnosis] RAG: found {len(similar_incidents)} similar incidents, "
        f"{len(runbook_hits)} runbook excerpts",
        "agent", "diagnosis", inc.id,
    )

    # 3. REASONING (Mistral, now WITH context) ---------------------------
    _set_status(db, inc, IncidentStatus.REASONING,
                "Diagnosis agent querying Mistral with retrieved context")
    await _broadcast(user_id, inc)
    
    error_text_for_graph = (run.error_message or "") + "\n" + "\n".join(
        l.get("message", "")
        for l in log_dicts
        if str(l.get("level", "")).upper() in {"ERROR", "CRITICAL"}
    )[-2000:]
    
    final_context_block = combined_context_block(
        chroma_context = context_block,
        error_text = error_text_for_graph,
        component  = connector.type.value,
        top_k      = settings.GRAPH_RAG_TOP_K,
    )

    await _log(db, user_id,
               f"[diagnosis] Querying Mistral ({mistral_service.model}) "
               f"(context: {len(final_context_block)} chars)",
               "agent", "diagnosis", inc.id)

    result = await asyncio.to_thread(
        mistral_service.analyze_failure,
        pipe.name,
        connector.type.value,
        run.error_message,
        log_dicts,
        pipe.metadata_json or {},
        final_context_block,
    )

    # 4. PLANNING --------------------------------------------------------
    await _agent_start("diagnosis", "Planning remediation steps")
    confidence = float(result.get("confidence") or 0.0)
    risk       = _risk_from_confidence(confidence)
    plan       = _parse_plan(result.get("suggested_fix"))

    inc.root_cause       = result.get("root_cause") or ""
    inc.agent_thought    = result.get("summary") or ""
    inc.proposed_action  = (result.get("suggested_fix") or "")[:200]
    inc.remediation_plan = plan
    inc.confidence_score = confidence
    inc.risk_tier        = risk

    tl = list(inc.timeline or [])
    tl.append({
        "ts": datetime.utcnow().isoformat(),
        "stage": "Planning",
        "agent": "diagnosis",
        "detail": (f"Root cause identified (confidence={confidence:.0%}, "
                   f"used_context={result.get('used_context', False)}): "
                   f"{result.get('summary','')[:120]}"),
    })
    inc.timeline = tl
    inc.status   = IncidentStatus.PLANNING
    db.commit()
    await _broadcast(user_id, inc)
    await _log(db, user_id,
               f"[diagnosis] Root cause: {result.get('root_cause','')[:200]}",
               "agent", "diagnosis", inc.id)
    await _log(db, user_id,
               f"[diagnosis] Confidence {confidence:.0%} · Risk tier → {risk}",
               "agent", "diagnosis", inc.id)
    await _agent_done("diagnosis")

    # 4b. EMAIL THE FIRST DATAOPS ENGINEER WITH DIAGNOSIS + SOLUTION
    primary_email, primary_role = await notify_incident_diagnosed(inc.id, result)
    if primary_email:
        # Log INITIAL_MAIL_SENT event for the journey table
        _log_incident_event(
            db, inc.id,
            IncidentEventType.INITIAL_MAIL_SENT.value,
            escalation_level="L1",
            recipients=[{"email": primary_email, "role": primary_role or "DataOps Engineer"}],
            related_run_id=run.id,
            details=f"Initial alert sent to {primary_role or 'recipient'} <{primary_email}>",
        )
        await _log(db, user_id,
                   f"[orchestrator] Initial email sent to {primary_role or 'recipient'} "
                   f"<{primary_email}>",
                   "info", "orchestrator", inc.id)

    # 4c. CLASSIFY (known vs new) + RECORD IN KNOWLEDGE BASE -------------
    classification = solution_kb_service.classify(
        db,
        error_text=inc.error_log or "",
        component=connector.type.value,
        llm_confidence=confidence,
    )

    pattern = solution_kb_service.record_occurrence(
        db,
        error_text=inc.error_log or "",
        component=connector.type.value,
        category=connector.type.value,
        root_cause=inc.root_cause or "",
        fix_summary=inc.proposed_action or "",
        fix_steps=inc.remediation_plan or [],
        llm_confidence=confidence,
    )

    # Route to the appropriate support group / individual.
    decision = route_support(
        category=connector.type.value,
        error_type=classification.error_type,
        is_known=classification.is_known,
        confidence=pattern.confidence if pattern else confidence,
    )
    if pattern is not None and not pattern.support_group:
        pattern.support_group = decision.group
        db.commit()

    # Record a journey event reflecting the classification.
    _log_incident_event(
        db, inc.id,
        "KNOWN_ERROR_MATCHED" if classification.is_known else "NEW_ERROR_DETECTED",
        details=(
            f"{classification.error_type} · {classification.reason} · "
            f"routed to {decision.group}"
        ),
    )
    await _log(
        db, user_id,
        f"[diagnosis] Classified as {'KNOWN' if classification.is_known else 'NEW'} "
        f"error ({classification.error_type}); {classification.reason}",
        "agent", "diagnosis", inc.id,
    )

    # 4d. CREATE JIRA TICKET (with RCA + routing) ------------------------
    fix_text = "\n- ".join(inc.remediation_plan) if inc.remediation_plan else inc.proposed_action
    if fix_text and inc.remediation_plan:
        fix_text = "- " + fix_text

    kb_note = (
        f"\n\nKnowledge base: signature {classification.signature[:12]}…, "
        f"seen {pattern.occurrence_count if pattern else 1}x, "
        f"agent confidence {(pattern.confidence if pattern else confidence):.0%}."
    )
    if classification.is_known:
        jira_desc = (
            f"KNOWN error matched in knowledge base.\n\n"
            f"Root Cause: {inc.root_cause}\n\nSuggested Fix:\n{fix_text}{kb_note}"
        )
    else:
        jira_desc = (
            f"NEW error type — manual investigation required. The agent will "
            f"learn the fix once a human PR is merged and ingested.\n\n"
            f"Root Cause: {inc.root_cause}\n\nSuggested Fix:\n{fix_text}{kb_note}"
        )

    jira_ticket = create_jira_ticket(
        summary=f"[{classification.error_type}] Incident in {pipe.name}",
        description=jira_desc,
        assign_to_human=True,
        labels=decision.labels,
        assignee_account_id=decision.assignee_account_id,
        support_group=decision.group,
    )
    inc.jira_ticket_key = jira_ticket.get("key")
    inc.jira_ticket_url = f"{settings.JIRA_BASE_URL.rstrip('/')}/browse/{inc.jira_ticket_key}" if settings.JIRA_BASE_URL else ""

    _log_incident_event(
        db, inc.id,
        IncidentEventType.JIRA_TICKET_CREATED.value,
        details=f"Created Jira ticket {inc.jira_ticket_key} → {decision.group}"
    )

    db.commit()
    await _broadcast(user_id, inc)
    await _log(db, user_id, f"[orchestrator] Created Jira ticket {inc.jira_ticket_key} (group: {decision.group})", "info", "orchestrator", inc.id)

    # 4e. KNOWN + AUTO-FIXABLE → write code and raise a PR (best-effort).
    #      NEW or low-confidence → notify only (handled by the email above).
    if classification.auto_fix:
        await _agent_start("remediation", "Writing code fix and raising PR")
        try:
            pr_result = await asyncio.to_thread(
                pr_service.raise_pr_for_incident, db, inc, pattern
            )
        except Exception as e:
            logger.warning("auto-PR failed for incident #%s: %s", inc.id, e)
            pr_result = {"ok": False, "reason": str(e)}

        if pr_result.get("ok") and pr_result.get("mode") == "pr":
            _log_incident_event(
                db, inc.id, "PR_RAISED",
                details=f"Auto-PR opened: {pr_result.get('pr_url')}",
            )
            await _log(db, user_id,
                       f"[remediation] Auto-PR opened for known error: {pr_result.get('pr_url')}",
                       "tool", "remediation", inc.id)
        else:
            await _log(db, user_id,
                       f"[remediation] Auto-PR not raised ({pr_result.get('reason') or pr_result.get('message')}) — left for human action",
                       "warn", "remediation", inc.id)
        await _agent_done("remediation")

    # 5. APPROVAL GATE ---------------------------------------------------
    inc.approval_required = True
    _set_status(db, inc, IncidentStatus.AWAITING_APPROVAL,
                "Human approval required before remediating")
    db.commit()
    await _broadcast(user_id, inc)
    await _log(db, user_id,
               f"[orchestrator] Awaiting approval — risk={risk}, confidence={confidence:.0%}",
               "warn", "orchestrator", inc.id)
    # The recurring escalation scheduler (escalation_scheduler.py) now
    # handles the follow-up checks every ESCALATION_CHECK_INTERVAL.
    # No one-shot asyncio.sleep timer is needed any more.
    _write_memory_and_recommendation(db, inc, confidence)
    _store_incident_in_vectors(inc, run.error_message or "")
    return

    # 6. EXECUTING -------------------------------------------------------
    await _agent_start("remediation", plan[0] if plan else "N/A")
    _set_status(db, inc, IncidentStatus.EXECUTING, "Remediation agent executing plan")
    await _broadcast(user_id, inc)
    await _log(db, user_id, f"[remediation] Applying fix: {plan[0] if plan else 'N/A'}",
               "tool", "remediation", inc.id)
    await _agent_done("remediation")

    # 7. EVALUATING ------------------------------------------------------
    _set_status(db, inc, IncidentStatus.EVALUATING, "Evaluating result")
    await _broadcast(user_id, inc)

    # 8. REMEDIATED + write back to vector store -------------------------
    await _agent_start("learning", f"Writing resolution for {pipe.name} to memory")
    inc.resolved_at = datetime.utcnow()
    _set_status(db, inc, IncidentStatus.REMEDIATED,
                "Pipeline analysis complete — fix suggestions documented")
    await _broadcast(user_id, inc)
    await _log(db, user_id,
               f"[learning] Incident {inc.id} remediated — writing to memory & vector store",
               "agent", "learning", inc.id)
    await _agent_done("learning")

    _write_memory_and_recommendation(db, inc, confidence)
    _store_incident_in_vectors(inc, run.error_message or "")
    
    try:
        record_incident_outcome(
            incident_id=inc.id,
            pipeline_name=inc.pipeline_name,
            summary=inc.agent_thought or inc.root_cause,
            confidence=inc.confidence_score,
            matched_pattern_signatures=[p["signature"] for p in result.get("matched_patterns", [])] if result and "matched_patterns" in result else [],
            applied_fixes=inc.remediation_plan or [],
            success=True,
        )
    except Exception as e:
        logger.warning(f"Failed to record incident outcome to graph: {e}")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _store_incident_in_vectors(inc: Incident, raw_error: str) -> None:
    """Best-effort write of the new incident into the Chroma 'incidents'
    collection so future similar failures retrieve it."""
    try:
        rag_service.store_incident(
            incident_id=inc.id,
            pipeline_name=inc.pipeline_name,
            error_log=inc.error_log or raw_error,
            root_cause=inc.root_cause,
            suggested_fix="\n".join(inc.remediation_plan or []),
            confidence=inc.confidence_score,
            risk_tier=inc.risk_tier,
        )
    except Exception:
        logger.exception("Failed to store incident #%s in vector DB", inc.id)


def _write_memory_and_recommendation(db: Session, inc: Incident, confidence: float) -> None:
    """Persist episodic memory + recommendation after a completed incident."""
    mem = MemoryEntry(
        kind="episodic",
        title=f"Failure on {inc.pipeline_name}",
        summary=inc.agent_thought or inc.root_cause or "(no summary)",
        payload={
            "incident_id":   inc.id,
            "pipeline_name": inc.pipeline_name,
            "root_cause":    inc.root_cause,
            "fix_steps":     inc.remediation_plan,
            "confidence":    inc.confidence_score,
        },
        tags=[inc.pipeline_name.lower().replace(" ", "_"), (inc.risk_tier or "").lower(), "auto"],
        success=inc.status == IncidentStatus.REMEDIATED,
    )
    db.add(mem)

    if inc.remediation_plan:
        rec = Recommendation(
            pipeline_name=inc.pipeline_name,
            title=f"Recurring fix: {inc.pipeline_name}",
            detail=inc.agent_thought or "",
            savings=f"~{max(1, int(confidence * 30))} min MTTR reduction",
            risk=inc.risk_tier,
            status="open",
        )
        db.add(rec)

    db.commit()


# ─── Synthetic demo incident (unchanged from your version, but now also
# goes through the RAG pipeline) ─────────────────────────────────────────────

DEMO_ERROR_LOG = """\
[2025-05-10 03:47:12] [ERROR] Task 'transform_customers' failed — exit code 1
[2025-05-10 03:47:12] [ERROR] KeyError: 'customer_id' in transform_customers()
  File 'pipeline/transform.py', line 42, in transform_customers
    df['customer_id'] = df['id'].astype(str)
KeyError: 'id'
[2025-05-10 03:47:08] [WARN]  Column 'id' not found in upstream schema (schema changed at 02:45)
[2025-05-10 03:47:01] [INFO]  Fetched 14,203 rows from source table
[2025-05-10 03:46:58] [INFO]  Starting run: job_id=demo-9201, attempt=1
[2025-05-10 03:47:15] [ERROR] Retry 2/3 failed — same error
[2025-05-10 03:47:21] [ERROR] Retry 3/3 failed — pipeline aborted
"""


async def run_synthetic_incident() -> None:
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        await _synthetic_inner(db)
    except Exception:
        logger.exception("synthetic incident failed")
    finally:
        db.close()


async def _synthetic_inner(db: Session) -> None:
    await _agent_start("monitoring", "Scanning pipeline health metrics")
    inc = Incident(
        run_id=None,
        pipeline_name="demo-data-pipeline",
        status=IncidentStatus.DETECTED,
        risk_tier="Medium",
        error_log=DEMO_ERROR_LOG,
        timeline=[{
            "ts": datetime.utcnow().isoformat(),
            "stage": "Detected",
            "agent": "monitoring",
            "detail": "Synthetic demo: transform_customers failed with KeyError",
        }],
    )
    db.add(inc); db.commit(); db.refresh(inc)
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    await _log(db, 0, "[monitoring] Demo pipeline failure detected — triggering agent loop",
               "agent", "monitoring", inc.id)
    await _agent_done("monitoring")
    await asyncio.sleep(1.0)

    # NEW: RAG retrieval first
    context_block, similar_incidents, runbook_hits = await asyncio.to_thread(
        _build_rag_context, "KeyError customer_id transform schema drift demo-data-pipeline"
    )
    inc.similar_incidents = [
        h.get("metadata", {}).get("incident_id") or h.get("id")
        for h in similar_incidents
    ]
    db.commit()
    await _log(db, 0,
               f"[diagnosis] RAG: found {len(similar_incidents)} similar incidents, "
               f"{len(runbook_hits)} runbook excerpts",
               "agent", "diagnosis", inc.id)

    await _agent_start("diagnosis", "Analysing demo pipeline logs with Mistral + RAG")
    _set_status(db, inc, IncidentStatus.REASONING,
                "Diagnosis agent querying Mistral with retrieved context")
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    await _log(db, 0, f"[diagnosis] Querying {mistral_service.model}…",
               "agent", "diagnosis", inc.id)

    result = await asyncio.to_thread(
        mistral_service.analyze_failure,
        "demo-data-pipeline", "synthetic",
        "KeyError: 'id' in transform_customers",
        [{"timestamp": datetime.utcnow().isoformat(), "level": "ERROR",
          "source": "transform.py", "message": DEMO_ERROR_LOG}],
        {"is_demo": True},
        combined_context_block(
            chroma_context=context_block,
            error_text=DEMO_ERROR_LOG[-2000:],
            component="synthetic",
            top_k=settings.GRAPH_RAG_TOP_K,
        ),
    )

    confidence = float(result.get("confidence") or 0.68)
    risk = _risk_from_confidence(confidence)
    plan = _parse_plan(result.get("suggested_fix")) or [
        "Check upstream table for recent schema changes (column rename: 'id' → 'customer_id')",
        "Update transform mapping in pipeline/transform.py line 42",
        "Backfill the 14,203 rows from the source table",
        "Re-trigger pipeline run after applying the fix",
        "Add schema validation step to catch future drift",
    ]
    inc.root_cause = (result.get("root_cause") or
                      "Upstream schema change renamed column 'id' to 'customer_id', "
                      "causing a KeyError in transform step at pipeline/transform.py:42")
    inc.agent_thought = result.get("summary") or "Schema drift in upstream table — column 'id' no longer exists"
    inc.proposed_action = plan[0][:200]
    inc.remediation_plan = plan
    inc.confidence_score = confidence
    inc.risk_tier = risk

    tl = list(inc.timeline or [])
    tl.append({"ts": datetime.utcnow().isoformat(), "stage": "Planning",
               "agent": "diagnosis",
               "detail": f"Root cause identified (confidence={confidence:.0%}): {inc.agent_thought[:120]}"})
    inc.timeline = tl
    inc.status = IncidentStatus.PLANNING
    db.commit()
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    await _log(db, 0, f"[diagnosis] Root cause: {inc.root_cause[:200]}",
               "agent", "diagnosis", inc.id)
    await _log(db, 0, f"[diagnosis] Confidence={confidence:.0%} → Risk tier={risk}",
               "agent", "diagnosis", inc.id)
    await _agent_done("diagnosis")
    await asyncio.sleep(0.6)

    # Email the first DataOps Engineer with diagnosis + solution
    primary_email, primary_role = await notify_incident_diagnosed(inc.id, result)
    if primary_email:
        await _log(db, 0,
                   f"[orchestrator] Initial email sent to {primary_role or 'recipient'} "
                   f"<{primary_email}>",
                   "info", "orchestrator", inc.id)

    inc.approval_required = True
    tl2 = list(inc.timeline or [])
    tl2.append({"ts": datetime.utcnow().isoformat(), "stage": "Awaiting Approval",
                "agent": "orchestrator",
                "detail": f"Approval required (risk={risk})"})
    inc.timeline = tl2
    inc.status = IncidentStatus.AWAITING_APPROVAL
    db.commit()
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    await _log(db, 0, f"[orchestrator] GATE: paused for approval. risk={risk}",
               "warn", "orchestrator", inc.id)
    # Escalation is now handled by the recurring scheduler
    # in escalation_scheduler.py (no one-shot timer needed).
    _write_memory_and_recommendation(db, inc, confidence)
    _store_incident_in_vectors(inc, DEMO_ERROR_LOG)
    return

    await _agent_start("remediation", plan[0])
    tl3 = list(inc.timeline or [])
    tl3.append({"ts": datetime.utcnow().isoformat(), "stage": "Executing",
                "agent": "remediation", "detail": "Remediation agent executing plan"})
    inc.timeline = tl3
    inc.status = IncidentStatus.EXECUTING
    db.commit()
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    await _log(db, 0, f"[remediation] {plan[0]}", "tool", "remediation", inc.id)
    await asyncio.sleep(0.6)
    await _agent_done("remediation")

    tl4 = list(inc.timeline or [])
    tl4.append({"ts": datetime.utcnow().isoformat(), "stage": "Evaluating",
                "agent": "remediation", "detail": "Verifying pipeline recovery"})
    inc.timeline = tl4
    inc.status = IncidentStatus.EVALUATING
    db.commit()
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    await asyncio.sleep(0.5)

    await _agent_start("learning", f"Writing resolution for {inc.pipeline_name} to memory")
    inc.resolved_at = datetime.utcnow()
    tl5 = list(inc.timeline or [])
    tl5.append({"ts": datetime.utcnow().isoformat(), "stage": "Remediated",
                "agent": "learning", "detail": "Demo incident closed — loop complete"})
    inc.timeline = tl5
    inc.status = IncidentStatus.REMEDIATED
    db.commit()
    await manager.broadcast({"event": "incident", "payload": _incident_to_dict(inc)})
    await _log(db, 0,
               f"[learning] Demo incident #{inc.id} remediated. Episodic memory + vector store updated.",
               "agent", "learning", inc.id)
    await _agent_done("learning")
    _write_memory_and_recommendation(db, inc, confidence)
    _store_incident_in_vectors(inc, DEMO_ERROR_LOG)
    
    try:
        record_incident_outcome(
            incident_id=inc.id,
            pipeline_name=inc.pipeline_name,
            summary=inc.agent_thought or inc.root_cause,
            confidence=inc.confidence_score,
            matched_pattern_signatures=[p["signature"] for p in result.get("matched_patterns", [])] if result and "matched_patterns" in result else [],
            applied_fixes=inc.remediation_plan or [],
            success=True,
        )
    except Exception as e:
        logger.warning(f"Failed to record demo incident outcome to graph: {e}")
