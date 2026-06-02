"""
Pipeline sync service.

This is the orchestrator that runs in the background:
  for each connected connector:
      pull pipelines        -> upsert into DB
      pull recent runs      -> upsert into DB
      for each newly-FAILED run:
          pull logs         -> insert
          run Mistral analysis
          broadcast to user via WebSocket

It is invoked both from APScheduler (periodic) and from the API (on-demand).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.connectors import get_connector
from app.core.security import decrypt_secret
from app.models import (
    Connector, ConnectorStatus, Pipeline, PipelineRun, PipelineLog,
    ErrorAnalysis, RunStatus, LogLevel,
)
from app.services.mistral_service import mistral_service
from app.websockets.manager import manager
from app.services.email_service import send_pipeline_error_email
from app.models import User

logger = logging.getLogger(__name__)

def _json_safe(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(i) for i in obj]
    return obj


def _decrypt_creds(connector: Connector) -> dict[str, Any]:
    return json.loads(decrypt_secret(connector.encrypted_credentials))


async def _broadcast(user_id: int | None, event: str, payload: dict) -> None:
    try:
        if user_id is None:
            await manager.broadcast({"event": event, "data": payload})
        else:
            await manager.send_to_user(user_id, {"event": event, "data": payload})
    except Exception as e:
        logger.debug("ws broadcast failed: %s", e)


def _serialize_run(run: PipelineRun) -> dict:
    return {
        "id": run.id,
        "pipeline_id": run.pipeline_id,
        "external_run_id": run.external_run_id,
        "status": run.status.value if run.status else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_seconds": run.duration_seconds,
        "error_message": run.error_message,
    }


# ---------------------------------------------------------------------------
# Single-connector sync
# ---------------------------------------------------------------------------
async def sync_connector(db: Session, connector: Connector) -> dict[str, Any]:
    """Sync one connector. Returns a small stats dict."""
    stats = {"pipelines": 0, "new_runs": 0, "newly_failed": 0, "errors": []}
    

    try:
        creds = _decrypt_creds(connector)
        client = get_connector(connector.type, creds)
    except Exception as exc:
        connector.status = ConnectorStatus.ERROR
        connector.last_error = f"Failed to instantiate connector: {exc}"
        db.commit()
        stats["errors"].append(connector.last_error)
        return stats

    # 1. Pipelines -----------------------------------------------------
    try:
        remote_pipelines = await asyncio.to_thread(client.list_pipelines)
    except Exception as exc:
        logger.exception("list_pipelines failed for connector %s", connector.id)
        connector.status = ConnectorStatus.ERROR
        connector.last_error = str(exc)
        db.commit()
        stats["errors"].append(str(exc))
        return stats

    existing = {p.external_id: p for p in connector.pipelines}
    for rp in remote_pipelines:
        pipe = existing.get(rp.external_id)
        if pipe is None:
            pipe = Pipeline(
                connector_id=connector.id,
                external_id=rp.external_id,
                name=rp.name,
                description=rp.description,
                metadata_json=rp.metadata,
            )
            db.add(pipe)
            db.flush()  # get pipe.id
            existing[rp.external_id] = pipe
            await _broadcast(None, "pipeline.created", {
                "id": pipe.id, "name": pipe.name, "connector_id": connector.id,
            })
        else:
            pipe.name = rp.name
            pipe.description = rp.description
            pipe.metadata_json = rp.metadata
        stats["pipelines"] += 1

    db.commit()

    # 2. Runs + logs + LLM analysis -----------------------------------
    for pipe in list(existing.values()):
        try:
            remote_runs = await asyncio.to_thread(client.list_runs, pipe.external_id, 25)
        except Exception as exc:
            logger.warning("list_runs failed for pipeline %s: %s", pipe.external_id, exc)
            stats["errors"].append(f"{pipe.name}: {exc}")
            
            # Fallback: Sync pipeline status from the latest known run in the database
            latest_run = (
                db.query(PipelineRun)
                .filter(PipelineRun.pipeline_id == pipe.id)
                .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
                .first()
            )
            if latest_run:
                pipe.last_run_status = latest_run.status
                run_ts = latest_run.started_at or latest_run.finished_at
                if run_ts:
                    pipe.last_run_at = run_ts
                db.commit()
                
            continue

        existing_runs = {r.external_run_id: r for r in pipe.runs}

        for rr in remote_runs:
            run = existing_runs.get(rr.external_run_id)
            new_run = run is None
            prev_status = run.status if run else None

            if new_run:
                run = PipelineRun(
                    pipeline_id=pipe.id,
                    external_run_id=rr.external_run_id,
                    status=RunStatus(rr.status),
                    started_at=rr.started_at,
                    finished_at=rr.finished_at,
                    duration_seconds=rr.duration_seconds,
                    error_message=rr.error_message,
                    raw_payload=_json_safe(rr.raw),
                )
                db.add(run)
                db.flush()
                existing_runs[rr.external_run_id] = run
                stats["new_runs"] += 1
            else:
                run.status = RunStatus(rr.status)
                run.started_at = rr.started_at or run.started_at
                run.finished_at = rr.finished_at or run.finished_at
                run.duration_seconds = rr.duration_seconds or run.duration_seconds
                run.error_message = rr.error_message or run.error_message
                run.raw_payload = _json_safe(rr.raw)

            db.commit()

            # Update pipeline's "latest run" pointers only if this run
            # is newer than what is already stored (prevents older runs
            # processed later in the loop from overwriting the value).
            run_ts = run.started_at or run.finished_at
            if run_ts and (pipe.last_run_at is None or run_ts >= pipe.last_run_at):
                pipe.last_run_status = run.status
                pipe.last_run_at = run_ts
            db.commit()

            await _broadcast(None, "run.updated", {
                "pipeline_id": pipe.id,
                "pipeline_name": pipe.name,
                "run": _serialize_run(run),
            })

        # --- AFTER PIPELINE RUN SYNC IS COMPLETE ---
        # Query the database for the single latest run of this pipeline
        latest_run = (
            db.query(PipelineRun)
            .filter(PipelineRun.pipeline_id == pipe.id)
            .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
            .first()
        )

        if latest_run and latest_run.status == RunStatus.FAILED:
            # Check if this specific failed run already has an active or closed incident record
            from app.models.agent_models import Incident
            existing_inc = db.query(Incident).filter(Incident.run_id == latest_run.id).first()

            if not existing_inc:
                logger.info(f"Pipeline '{pipe.name}' latest run #{latest_run.id} is FAILED. Initiating escalation and incident flow.")

                # Admin/L1 alerts are fully handled by the premium AI incident loop below
                stats["newly_failed"] += 1

                await _ingest_logs_and_analyze(db, client, connector, pipe, latest_run)
                # Kick off the agent-loop incident for this run
                try:
                    from app.services.incident_service import process_failed_run
                    await process_failed_run(db, latest_run, None)
                except Exception as _ie:
                    logger.exception("incident_service.process_failed_run failed: %s", _ie)

    connector.last_synced_at = datetime.utcnow()
    connector.status = ConnectorStatus.CONNECTED
    connector.last_error = None
    db.commit()

    await _broadcast(None, "connector.synced", {
        "id": connector.id, "stats": stats,
        "last_synced_at": connector.last_synced_at.isoformat(),
    })
    return stats


async def _ingest_logs_and_analyze(
    db: Session, client, connector: Connector, pipe: Pipeline, run: PipelineRun,
) -> None:
    """Pull logs for a failed run, then call Mistral to analyse them."""
    try:
        remote_logs = await asyncio.to_thread(
            client.get_logs, pipe.external_id, run.external_run_id,
        )
    except Exception as exc:
        logger.warning("get_logs failed: %s", exc)
        remote_logs = []

    # Wipe old logs for this run (idempotent re-sync) then insert fresh
    db.query(PipelineLog).filter(PipelineLog.run_id == run.id).delete()
    log_dicts: list[dict] = []
    for nl in remote_logs:
        plog = PipelineLog(
            run_id=run.id,
            timestamp=nl.timestamp or datetime.utcnow(),
            level=LogLevel(nl.level) if nl.level in LogLevel.__members__ else LogLevel.INFO,
            source=nl.source,
            message=nl.message,
        )
        db.add(plog)
        log_dicts.append({
            "timestamp": plog.timestamp.isoformat(),
            "level": plog.level.value,
            "source": plog.source,
            "message": plog.message,
        })
    db.commit()

    await _broadcast(None, "logs.updated", {
        "pipeline_id": pipe.id, "run_id": run.id, "log_count": len(log_dicts),
    })

    # Run Mistral analysis (skip if already analysed)
    existing = db.query(ErrorAnalysis).filter(ErrorAnalysis.run_id == run.id).first()
    if existing is not None:
        return

    try:
        result = await asyncio.to_thread(
            mistral_service.analyze_failure,
            pipe.name,
            connector.type.value,
            run.error_message,
            log_dicts,
            pipe.metadata_json or {},
        )
    except Exception as exc:
        logger.exception("Mistral analysis failed")
        result = {
            "summary": "Analysis failed",
            "root_cause": str(exc),
            "suggested_fix": "",
            "fix_patch": "",
            "confidence": 0.0,
            "model": "",
            "raw_response": {"error": str(exc)},
        }

    analysis = ErrorAnalysis(
        run_id=run.id,
        summary=result["summary"],
        root_cause=result["root_cause"],
        suggested_fix=result["suggested_fix"],
        fix_patch=result.get("fix_patch") or "",
        confidence=result["confidence"],
        model=result["model"],
        raw_response=result["raw_response"],
    )
    db.add(analysis)
    db.commit()

    await _broadcast(None, "analysis.ready", {
        "pipeline_id": pipe.id,
        "pipeline_name": pipe.name,
        "run_id": run.id,
        "summary": analysis.summary,
        "confidence": analysis.confidence,
    })


# ---------------------------------------------------------------------------
# All-connectors sweep (called by APScheduler)
# ---------------------------------------------------------------------------
async def sync_all_connectors() -> None:
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        connectors = db.query(Connector).filter(
            Connector.status.in_([ConnectorStatus.CONNECTED, ConnectorStatus.PENDING])
        ).all()
        logger.info("Periodic sync: %d connector(s)", len(connectors))
        for c in connectors:

            connector_id = getattr(c, "id", "unknown")

            try:
                await sync_connector(db, c)

            except Exception:

                db.rollback()

                logger.exception(
                    "Sync failed for connector %s",
                    connector_id,
                )
    finally:
        db.close()