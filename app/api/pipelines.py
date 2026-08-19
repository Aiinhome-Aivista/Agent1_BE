"""Pipeline-related read endpoints + analysis + auto-fix."""
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import traceback
import sys

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_current_user
from app.connectors import get_connector
from app.core.database import get_db
from app.core.security import decrypt_secret
from app.models import (
    User, Connector, Pipeline, PipelineRun, PipelineLog, ErrorAnalysis, RunStatus,
)
from app.schemas import (
    PipelineOut, PipelineDetailOut, PipelineRunOut, PipelineLogOut,
    ErrorAnalysisOut, DashboardStats,
)
from app.services.mistral_service import mistral_service
from sqlalchemy import desc, func


router = APIRouter(tags=["pipelines"])


def _save_analysis_to_disk(analysis, run: "PipelineRun"):
    try:
        connector_name = run.pipeline.connector.name
        external_run_id = run.external_run_id
        created_at = analysis.created_at.strftime("%Y%m%d_%H%M%S") if analysis.created_at else "unknown"

        # sanitise connector name for use as a folder name
        safe_connector_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in connector_name)

        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        save_dir = BASE_DIR / "analyse_results" / safe_connector_name / external_run_id / created_at
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "analysis.json", "w") as f:
            json.dump(
                {
                    "summary": analysis.summary,
                    "root_cause": analysis.root_cause,
                    "suggested_fix": analysis.suggested_fix,
                },
                f,
                indent=2,
            )
        print(f"[DEBUG] Saved to: {save_dir}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to save analysis: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)


def _user_pipeline(db: Session, pipeline_id: int, user: User) -> Pipeline:
    pipe = (
        db.query(Pipeline)
        .join(Connector, Connector.id == Pipeline.connector_id)
        .filter(Pipeline.id == pipeline_id)
        .first()
    )
    if not pipe:
        raise HTTPException(404, "Pipeline not found")
    return pipe


def _user_run(db: Session, run_id: int, user: User) -> PipelineRun:
    run = (
        db.query(PipelineRun)
        .join(Pipeline, Pipeline.id == PipelineRun.pipeline_id)
        .join(Connector, Connector.id == Pipeline.connector_id)
        .filter(PipelineRun.id == run_id)
        .first()
    )
    if not run:
        raise HTTPException(404, "Run not found")
    return run


# ---------- Dashboard --------------------------------------------------------

@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DashboardStats:
    cutoff = datetime.utcnow() - timedelta(hours=24)

    base_runs = (
        db.query(PipelineRun)
        .join(Pipeline, Pipeline.id == PipelineRun.pipeline_id)
        .join(Connector, Connector.id == Pipeline.connector_id)
    )

    total_connectors = db.query(Connector).count()
    total_pipelines = (
        db.query(Pipeline).join(Connector).count()
    )

    runs_24h_q = base_runs.filter(PipelineRun.started_at >= cutoff)
    runs_last_24h = runs_24h_q.count()
    failed_runs_24h = runs_24h_q.filter(PipelineRun.status == RunStatus.FAILED).count()
    succeeded_24h = runs_24h_q.filter(PipelineRun.status == RunStatus.SUCCEEDED).count()
    completed = failed_runs_24h + succeeded_24h
    success_rate = (succeeded_24h / completed * 100.0) if completed else 100.0

    pending_analyses = (
        db.query(PipelineRun)
        .join(Pipeline).join(Connector)
        .outerjoin(ErrorAnalysis, ErrorAnalysis.run_id == PipelineRun.id)
        .filter(
            PipelineRun.status == RunStatus.FAILED,
            ErrorAnalysis.id.is_(None),
        )
        .count()
    )

    return DashboardStats(
        total_connectors=total_connectors,
        total_pipelines=total_pipelines,
        runs_last_24h=runs_last_24h,
        success_rate_24h=round(success_rate, 1),
        failed_runs_24h=failed_runs_24h,
        pending_analyses=pending_analyses,
    )


# ---------- Pipelines --------------------------------------------------------

@router.get("/pipelines", response_model=list[PipelineOut])
def list_pipelines(
    connector_id: int | None = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Build the base query, eagerly load the latest run per pipeline
    # so reconcile_last_run() can compare without extra DB round-trips.
    from sqlalchemy.orm import joinedload as jl
    q = (
        db.query(Pipeline)
        .join(Connector, Connector.id == Pipeline.connector_id)
        .options(jl(Pipeline.runs))
    )
    if connector_id is not None:
        q = q.filter(Pipeline.connector_id == connector_id)
    pipelines = q.order_by(
        Pipeline.last_run_at.is_(None),
        desc(Pipeline.last_run_at),
    ).all()

    # Self-heal: if the stored last_run_status/last_run_at doesn't match the
    # actual latest run (can happen if a run transitions RUNNING->FAILED between
    # sync cycles), fix it in-place and persist so the cache stays fresh.
    healed = False
    for pipe in pipelines:
        if pipe.reconcile_last_run():
            healed = True
    if healed:
        try:
            db.commit()
        except Exception:
            db.rollback()

    return pipelines


@router.get("/pipelines/{pipeline_id}", response_model=PipelineDetailOut)
def get_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    pipe = _user_pipeline(db, pipeline_id, user)
    runs = (
        db.query(PipelineRun)
        .options(joinedload(PipelineRun.analysis))
        .filter(PipelineRun.pipeline_id == pipe.id)
        .order_by(
            PipelineRun.started_at.is_(None),
            desc(PipelineRun.started_at)
        )
            .limit(25)
        .all()
    )
    out = PipelineDetailOut.model_validate(pipe)
    out.runs = [PipelineRunOut.model_validate(r) for r in runs]
    return out


# ---------- Runs / logs / analysis -----------------------------------------

@router.get("/runs/{run_id}", response_model=PipelineRunOut)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = _user_run(db, run_id, user)
    return run


@router.get("/runs/{run_id}/logs", response_model=list[PipelineLogOut])
def get_run_logs(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = _user_run(db, run_id, user)
    return (
        db.query(PipelineLog)
        .filter(PipelineLog.run_id == run.id)
        .order_by(PipelineLog.timestamp.asc(), PipelineLog.id.asc())
        .all()
    )


@router.get("/runs/{run_id}/analysis", response_model=ErrorAnalysisOut)
def get_run_analysis(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = _user_run(db, run_id, user)
    if not run.analysis:
        raise HTTPException(404, "No analysis available for this run yet")
    _save_analysis_to_disk(run.analysis, run)
    return run.analysis


@router.post("/runs/{run_id}/analyze", response_model=ErrorAnalysisOut)
def trigger_analysis(
    run_id: int,
    force: bool = Query(False, description="Re-run even if analysis exists"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Manually trigger LLM analysis on any run (failed or otherwise)."""
    run = _user_run(db, run_id, user)
    if run.analysis and not force:
        _save_analysis_to_disk(run.analysis, run)
        return run.analysis

    pipe = run.pipeline
    connector = pipe.connector
    logs = (
        db.query(PipelineLog)
        .filter(PipelineLog.run_id == run.id)
        .order_by(PipelineLog.timestamp.asc())
        .all()
    )
    log_dicts = [
        {
            "timestamp": l.timestamp.isoformat(),
            "level": l.level.value,
            "source": l.source,
            "message": l.message,
        }
        for l in logs
    ]

    # ── Step 1: Build the error query text for RAG ──────────────────────────
    error_query = " ".join(filter(None, [
        run.error_message or "",
        pipe.name,
        connector.type.value,
        *[l["message"] for l in log_dicts if str(l.get("level","")).upper() in {"ERROR","CRITICAL"}],
    ]))[:2000]

    # ── Step 2: Search knowledge base (similar incidents + runbooks) ─────────
    context_block = ""
    kb_references: list[dict] = []
    _runbook_top_similarity: float | None = None  # passed to confidence_explainer later
    _similar_incidents: list = []
    _runbook_chunks: list = []
    try:
        from app.services.rag_service import rag_service          # noqa: PLC0415
        from app.services.graph_rag_service import combined_context_block  # noqa: PLC0415
        _similar_incidents = rag_service.search_similar_incidents(error_query, k=3)
        _runbook_chunks    = rag_service.search_runbooks(error_query, k=3)

        # ── FIX 1: Use combined_context_block so ArangoDB graph-ranked fixes
        #          are included alongside Chroma semantic results. ──────────
        chroma_block  = rag_service.build_context_block(_similar_incidents, _runbook_chunks)
        context_block = combined_context_block(
            chroma_context=chroma_block,
            error_text=error_query,
            component=connector.type.value,
        )

        # ── FIX 3 (prep): track top runbook similarity for confidence explainer
        if _runbook_chunks:
            _runbook_top_similarity = float(_runbook_chunks[0].get("similarity", 0.0))

        # Build lightweight reference list for the UI
        for hit in _similar_incidents:
            m = hit.get("metadata") or {}
            kb_references.append({
                "kind":        "historical_incident",
                "title":       m.get("pipeline_name", "Historical Incident"),
                "similarity":  round(float(hit.get("similarity", 0.0)), 3),
                "risk_tier":   m.get("risk_tier", ""),
                "incident_id": m.get("incident_id", ""),
            })
        for hit in _runbook_chunks:
            m = hit.get("metadata") or {}
            kb_references.append({
                "kind":       "runbook",
                "title":      m.get("title") or m.get("source_filename") or "Runbook",
                "similarity": round(float(hit.get("similarity", 0.0)), 3),
                "source":     m.get("source_filename", ""),
            })
    except Exception as _rag_err:
        import logging
        logging.getLogger(__name__).warning("RAG lookup failed (non-fatal): %s", _rag_err)

    # ── FIX 2: Pre-classify against the KB so we can inject the proven fix
    #          text into the LLM prompt BEFORE calling the model. ──────────
    _pre_cls = None
    _error_text_for_kb = (run.error_message or "") + "\n" + "\n".join(
        l["message"] for l in log_dicts
        if str(l.get("level", "")).upper() in {"ERROR", "CRITICAL"}
    )
    try:
        from app.services.solution_kb_service import solution_kb_service  # noqa: PLC0415
        _pre_cls = solution_kb_service.classify(
            db, error_text=_error_text_for_kb, component=connector.type.value,
            llm_confidence=0.0,
        )
        if _pre_cls.is_known and _pre_cls.pattern is not None:
            p = _pre_cls.pattern
            proven_fix_text = p.fix_summary or "\n".join(p.fix_steps or [])
            if proven_fix_text.strip():
                context_block = (
                    context_block + "\n\n" if context_block else ""
                ) + (
                    f"## Known accepted fix (seen {p.occurrence_count}x, "
                    f"{p.acceptance_count} accepted, confidence {(p.confidence or 0):.0%})\n"
                    f"{proven_fix_text.strip()[:2000]}"
                )
    except Exception as _kb_pre_err:
        import logging
        logging.getLogger(__name__).warning("KB pre-classify failed (non-fatal): %s", _kb_pre_err)

    # ── Step 3: Call LLM with enriched context ───────────────────────────────
    result = mistral_service.analyze_failure(
        pipe.name, connector.type.value, run.error_message, log_dicts,
        pipe.metadata_json or {},
        context_block=context_block or None,
    )

    # ── Enrich: classify against the KB and explain WHY the confidence is what
    #    it is, then fold the detail into raw_response (schema-safe — no new
    #    columns needed on error_analyses). ──────────────────────────────
    enriched_raw = dict(result.get("raw_response") or {})
    llm_conf = float(result.get("confidence") or 0.0)
    final_conf = llm_conf
    try:
        from app.services.solution_kb_service import solution_kb_service   # noqa: PLC0415
        from app.services import confidence_explainer                     # noqa: PLC0415
        # Re-use the pre-classification result (already computed above) and
        # update it with the real LLM confidence now that we have it.
        if _pre_cls is not None:
            cls = _pre_cls
            # Recompute with actual llm_conf (was 0.0 during pre-classify)
            if cls.pattern is not None:
                final_conf = max(llm_conf, float(cls.pattern.confidence or 0.0))
        else:
            cls = solution_kb_service.classify(
                db, error_text=_error_text_for_kb, component=connector.type.value,
                llm_confidence=llm_conf,
            )
            if cls.pattern is not None:
                final_conf = max(llm_conf, float(cls.pattern.confidence or 0.0))

        # ── FIX 3: Pass runbook_top_similarity so the confidence explainer
        #          renders the "Runbook coverage" factor. ──────────────────
        explanation = confidence_explainer.build(
            llm_confidence=llm_conf,
            final_confidence=final_conf,
            pattern=cls.pattern,
            is_known=cls.is_known,
            error_type=cls.error_type,
            runbook_top_similarity=_runbook_top_similarity,
            llm_rationale=result.get("confidence_rationale"),
        )
        enriched_raw["classification"] = {
            "is_known": cls.is_known,
            "auto_fix": cls.auto_fix,
            "error_type": cls.error_type,
            "signature": cls.signature,
            "reason": cls.reason,
        }
        enriched_raw["confidence_explanation"] = explanation.to_dict()
    except Exception:
        pass

    enriched_raw["root_cause_details"]  = result.get("root_cause_details") or []
    enriched_raw["validation_steps"]     = result.get("validation_steps") or []
    enriched_raw["confidence_rationale"] = result.get("confidence_rationale") or []
    enriched_raw["kb_references"]        = kb_references  # RAG-sourced references

    if run.analysis:
        analysis = run.analysis
        analysis.summary = result["summary"]
        analysis.root_cause = result["root_cause"]
        analysis.suggested_fix = result["suggested_fix"]
        analysis.fix_patch = result.get("fix_patch") or ""
        analysis.confidence = final_conf
        analysis.model = result["model"]
        analysis.raw_response = enriched_raw
        analysis.auto_fix_applied = False
        analysis.auto_fix_result = None
    else:
        analysis = ErrorAnalysis(
            run_id=run.id,
            summary=result["summary"],
            root_cause=result["root_cause"],
            suggested_fix=result["suggested_fix"],
            fix_patch=result.get("fix_patch") or "",
            confidence=final_conf,
            model=result["model"],
            raw_response=enriched_raw,
        )
        db.add(analysis)
    db.commit()
    db.refresh(analysis)
    _save_analysis_to_disk(analysis, run)

    # ── FIX 4 + 5: Post-analysis learning loop ──────────────────────────────
    # Store this diagnosis back into Chroma so future similar failures get
    # RAG context from it. Also record/upsert the error pattern in the
    # PostgreSQL KB so `is_known` becomes true on the second occurrence.
    try:
        from app.services.rag_service import rag_service as _rs         # noqa: PLC0415
        from app.services.solution_kb_service import solution_kb_service as _sks  # noqa: PLC0415

        # ── FIX 4: Store diagnosis into Chroma incidents collection ──────
        _rs.store_incident(
            incident_id=f"run-{run.id}",
            pipeline_name=pipe.name,
            error_log=(
                (run.error_message or "") + "\n"
                + "\n".join(l["message"] for l in log_dicts[-20:])
            )[:4000],
            root_cause=result.get("root_cause") or "",
            suggested_fix=result.get("suggested_fix") or "",
            confidence=final_conf,
            risk_tier="Low" if final_conf >= 0.7 else ("Medium" if final_conf >= 0.4 else "High"),
        )

        # ── FIX 5: Upsert error pattern into PostgreSQL KB ───────────────
        _sks.record_occurrence(
            db,
            error_text=_error_text_for_kb,
            component=connector.type.value,
            category=connector.type.value,
            root_cause=result.get("root_cause") or "",
            fix_summary=result.get("suggested_fix") or "",
            fix_steps=result.get("validation_steps") or [],
            llm_confidence=final_conf,
        )
    except Exception as _learn_err:
        import logging
        logging.getLogger(__name__).warning(
            "Post-analysis learning loop failed (non-fatal): %s", _learn_err
        )

    return analysis


@router.post("/runs/{run_id}/auto-fix")
def apply_auto_fix(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Ask the connector to apply the LLM-suggested fix back to the source."""
    run = _user_run(db, run_id, user)
    if not run.analysis or not run.analysis.fix_patch:
        raise HTTPException(400, "No fix patch available for this run")

    connector = run.pipeline.connector
    creds = json.loads(decrypt_secret(connector.encrypted_credentials))
    client = get_connector(connector.type, creds)
    if not client.supports_auto_fix():
        raise HTTPException(400, f"{connector.type.value} connector does not support auto-fix")

    ok, msg = client.apply_fix(run.pipeline.external_id, run.analysis.fix_patch)
    run.analysis.auto_fix_applied = ok
    run.analysis.auto_fix_result = msg
    db.commit()
    return {"success": ok, "message": msg}