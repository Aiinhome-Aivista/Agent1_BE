"""
Databricks connector.

Uses the Jobs 2.1 REST API with a Personal Access Token. The workspace_url is
the host root, e.g. https://adb-1234567890.12.azuredatabricks.net
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.connectors.base import (
    BaseConnector, NormalizedPipeline, NormalizedRun, NormalizedLog,
)

logger = logging.getLogger(__name__)


_LIFECYCLE_TO_STATUS = {
    "PENDING": "QUEUED",
    "RUNNING": "RUNNING",
    "TERMINATING": "RUNNING",
    "TERMINATED": "SUCCEEDED",  # refined below using result_state
    "SKIPPED": "CANCELLED",
    "INTERNAL_ERROR": "FAILED",
}

_RESULT_TO_STATUS = {
    "SUCCESS": "SUCCEEDED",
    "FAILED": "FAILED",
    "TIMEDOUT": "FAILED",
    "CANCELED": "CANCELLED",
}


def _ts(ms: int | None) -> datetime | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class DatabricksConnector(BaseConnector):
    type_name = "DATABRICKS"

    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self.base_url = credentials["workspace_url"].rstrip("/")
        self.token = credentials["personal_access_token"]

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    def test_connection(self) -> tuple[bool, str]:
        try:
            data = self._get("/api/2.0/preview/scim/v2/Me")
            return True, f"Connected as {data.get('userName', 'unknown user')}"
        except httpx.HTTPStatusError as e:
            return False, f"Databricks auth failed: HTTP {e.response.status_code}"
        except Exception as e:
            logger.exception("Databricks test failed")
            return False, f"Databricks connection failed: {e}"

    def list_pipelines(self) -> list[NormalizedPipeline]:
        out: list[NormalizedPipeline] = []
        offset = 0
        while True:
            data = self._get("/api/2.1/jobs/list",
                             params={"limit": 25, "offset": offset, "expand_tasks": True})
            for job in data.get("jobs", []):
                settings = job.get("settings", {})
                out.append(NormalizedPipeline(
                    external_id=str(job["job_id"]),
                    name=settings.get("name", f"Job {job['job_id']}"),
                    description=settings.get("description"),
                    metadata={
                        "creator_user_name": job.get("creator_user_name"),
                        "tasks": [t.get("task_key") for t in settings.get("tasks", [])],
                    },
                ))
            if not data.get("has_more"):
                break
            offset += 25
            if offset > 500:  # safety
                break
        return out

    def list_runs(self, pipeline_external_id: str, limit: int = 25) -> list[NormalizedRun]:
        data = self._get("/api/2.1/jobs/runs/list", params={
            "job_id": int(pipeline_external_id),
            "limit": min(limit, 25),
        })
        runs: list[NormalizedRun] = []
        for r in data.get("runs", []):
            state = r.get("state", {})
            lifecycle = state.get("life_cycle_state")
            result = state.get("result_state")
            if lifecycle == "TERMINATED" and result:
                status = _RESULT_TO_STATUS.get(result, "UNKNOWN")
            else:
                status = _LIFECYCLE_TO_STATUS.get(lifecycle, "UNKNOWN")

            started = _ts(r.get("start_time"))
            ended = _ts(r.get("end_time"))
            duration = None
            if r.get("execution_duration") is not None:
                duration = r["execution_duration"] / 1000.0
            elif started and ended:
                duration = (ended - started).total_seconds()

            runs.append(NormalizedRun(
                external_run_id=str(r["run_id"]),
                status=status,
                started_at=started,
                finished_at=ended,
                duration_seconds=duration,
                error_message=state.get("state_message") if status == "FAILED" else None,
                raw=r,
            ))
        return runs

    def get_logs(self, pipeline_external_id: str, run_external_id: str) -> list[NormalizedLog]:
        try:
            data = self._get("/api/2.1/jobs/runs/get",
                             params={"run_id": int(run_external_id), "include_history": True})
        except Exception as e:
            logger.warning("Databricks runs/get failed: %s", e)
            return []

        logs: list[NormalizedLog] = []
        ts = _ts(data.get("start_time")) or datetime.now(timezone.utc)

        for task in data.get("tasks", []):
            task_state = task.get("state", {})
            task_status = task_state.get("result_state") or task_state.get("life_cycle_state")
            level = "ERROR" if task_status in ("FAILED", "TIMEDOUT") else "INFO"
            msg = f"[task] {task.get('task_key')} -> {task_status}"
            if task_state.get("state_message"):
                msg += f"\n{task_state['state_message']}"
            logs.append(NormalizedLog(
                timestamp=_ts(task.get("start_time")) or ts,
                level=level, message=msg, source=task.get("task_key"),
            ))

        # Try stdout/stderr for the run output
        try:
            output = self._get("/api/2.1/jobs/runs/get-output",
                               params={"run_id": int(run_external_id)})
            for stream_name in ("logs", "error", "error_trace"):
                content = output.get(stream_name)
                if content:
                    level = "ERROR" if stream_name != "logs" else "INFO"
                    logs.append(NormalizedLog(
                        timestamp=ts, level=level,
                        message=content[:8000], source=stream_name,
                    ))
        except Exception:
            pass

        return logs
