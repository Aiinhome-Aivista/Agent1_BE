"""Pipeline-related read endpoints + analysis + auto-fix."""
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import traceback
import sys

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

    result = mistral_service.analyze_failure(
        pipe.name, connector.type.value, run.error_message, log_dicts,
        pipe.metadata_json or {},
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
        error_text = (run.error_message or "") + "\n" + "\n".join(
            l["message"] for l in log_dicts
            if str(l.get("level", "")).upper() in {"ERROR", "CRITICAL"}
        )
        cls = solution_kb_service.classify(
            db, error_text=error_text, component=connector.type.value,
            llm_confidence=llm_conf,
        )
        # If we have a known pattern, blend toward its reinforced confidence.
        if cls.pattern is not None:
            final_conf = max(llm_conf, float(cls.pattern.confidence or 0.0))
        explanation = confidence_explainer.build(
            llm_confidence=llm_conf,
            final_confidence=final_conf,
            pattern=cls.pattern,
            is_known=cls.is_known,
            error_type=cls.error_type,
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

    enriched_raw["root_cause_details"] = result.get("root_cause_details") or []
    enriched_raw["validation_steps"] = result.get("validation_steps") or []
    enriched_raw["confidence_rationale"] = result.get("confidence_rationale") or []

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