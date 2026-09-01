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
            "expand_tasks": True,
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

    # ------------------------------------------------------------------
    @staticmethod
    def _is_generic_wrapper(text: str | None) -> bool:
        """Return True if the error text is only a generic Databricks wrapper message."""
        if not text:
            return True
        t = str(text).lower().strip().rstrip(".")
        return (
            t == "workload failed, see run output for details"
            or t == "workload failed"
            or (len(t) < 120 and "workload failed" in t and "see run output" in t)
        )

    def _get_output_with_progressive_retry(
        self,
        run_id: int | str,
        *,
        max_attempts: int = 5,
        delays: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Progressive, content-aware retry for Databricks Jobs API get-output.

        Continues retrying if:
        1. API returns 404 (task terminating / output not yet flushed) or 5xx.
        2. API returns HTTP 200, but output contains only the generic wrapper
           ('Workload failed, see run output for details.') or is completely empty.

        Stops immediately as soon as a non-wrapper exception, error_trace, or
        notebook output is retrieved.
        """
        import time

        if delays is None:
            delays = [0.0, 2.0, 4.0, 8.0, 12.0]

        last_output: dict[str, Any] = {}
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            delay = delays[attempt] if attempt < len(delays) else 10.0
            if delay > 0:
                time.sleep(delay)

            try:
                output = self._get(
                    "/api/2.1/jobs/runs/get-output",
                    params={"run_id": int(run_id)},
                )
                last_output = output or {}

                err_text = str(last_output.get("error") or "").strip()
                trace_text = str(last_output.get("error_trace") or "").strip()
                nb_output = last_output.get("notebook_output") or {}

                # Check if we have actionable, non-wrapper content
                has_trace = bool(trace_text and len(trace_text) > 20)
                has_real_error = bool(err_text and not self._is_generic_wrapper(err_text))
                has_nb_res = bool(isinstance(nb_output, dict) and (nb_output.get("result") or nb_output.get("error")))

                if has_real_error or has_trace or has_nb_res:
                    # Successfully retrieved deep failure telemetry
                    return last_output

            except httpx.HTTPStatusError as e:
                # 404 = output not ready yet; 5xx = transient server error
                if e.response.status_code in (404, 500, 502, 503, 504):
                    last_exc = e
                    continue
                # 400 Bad Request (e.g. parent run on multi-task) or 401/403: don't retry
                raise
            except Exception as e:
                last_exc = e
                continue

        # If loop exhausts, return the best available output
        if last_output:
            return last_output
        if last_exc:
            raise last_exc
        return {}

    def check_logs_completeness(self, logs: list[NormalizedLog]) -> dict:
        """
        Assess whether a list of NormalizedLog entries has useful error detail.
        """
        error_lines = [
            l for l in logs
            if l.level in ("ERROR", "CRITICAL")
            and not self._is_generic_wrapper(l.message)
            and str(l.source) != "_investigation"
        ]
        detail_chars = sum(len(l.message or "") for l in error_lines)
        return {
            "is_complete": len(error_lines) > 0 and detail_chars > 50,
            "has_error_detail": len(error_lines) > 0,
            "error_lines": len(error_lines),
            "detail_chars": detail_chars,
        }

    def get_logs(self, pipeline_external_id: str, run_external_id: str) -> list[NormalizedLog]:
        try:
            data = self._get(
                "/api/2.1/jobs/runs/get",
                params={"run_id": int(run_external_id), "include_history": True},
            )
        except Exception as e:
            logger.warning("Databricks runs/get failed for run %s: %s", run_external_id, e)
            return []

        logs: list[NormalizedLog] = []
        ts = _ts(data.get("start_time")) or datetime.now(timezone.utc)
        tasks = data.get("tasks", [])

        # ------------------------------------------------------------------
        # Case A: MULTI-TASK Job (tasks[] is present and non-empty)
        # ------------------------------------------------------------------
        if tasks:
            for task in tasks:
                task_key = task.get("task_key") or "task"
                task_state = task.get("state", {})
                task_status = task_state.get("result_state") or task_state.get("life_cycle_state")
                task_run_id = task.get("run_id")
                level = "ERROR" if task_status in ("FAILED", "TIMEDOUT", "INTERNAL_ERROR") else "INFO"

                task_msg = f"[task] {task_key} -> {task_status}"
                if task_state.get("state_message"):
                    task_msg += f"\n{task_state['state_message']}"

                logs.append(NormalizedLog(
                    timestamp=_ts(task.get("start_time")) or ts,
                    level=level,
                    message=task_msg,
                    source=task_key,
                ))

                # For failed/terminated tasks, query the task-level get-output endpoint
                if task_run_id and task_status in ("FAILED", "TIMEDOUT", "INTERNAL_ERROR", "TERMINATED"):
                    try:
                        task_output = self._get_output_with_progressive_retry(task_run_id)

                        # 1. Error text (RuntimeError, Exception, etc.)
                        err_content = task_output.get("error")
                        if err_content:
                            logs.append(NormalizedLog(
                                timestamp=_ts(task.get("end_time")) or ts,
                                level="ERROR",
                                message=str(err_content).strip()[:10000],
                                source=task_key,
                            ))

                        # 2. Error traceback
                        trace_content = task_output.get("error_trace")
                        if trace_content:
                            logs.append(NormalizedLog(
                                timestamp=_ts(task.get("end_time")) or ts,
                                level="ERROR",
                                message=str(trace_content).strip()[:10000],
                                source=f"{task_key}_trace",
                            ))

                        # 3. Notebook output
                        nb_output = task_output.get("notebook_output")
                        if isinstance(nb_output, dict):
                            res_str = nb_output.get("result") or nb_output.get("error")
                            if res_str:
                                logs.append(NormalizedLog(
                                    timestamp=ts,
                                    level="ERROR" if "error" in nb_output else "INFO",
                                    message=str(res_str).strip()[:8000],
                                    source=f"{task_key}_output",
                                ))
                        elif nb_output:
                            logs.append(NormalizedLog(
                                timestamp=ts,
                                level="INFO",
                                message=str(nb_output).strip()[:8000],
                                source=f"{task_key}_output",
                            ))

                        # 4. Standard task logs
                        log_content = task_output.get("logs")
                        if log_content:
                            logs.append(NormalizedLog(
                                timestamp=ts,
                                level="INFO",
                                message=str(log_content).strip()[:8000],
                                source=f"{task_key}_logs",
                            ))

                        if err_content or trace_content:
                            logs.append(NormalizedLog(
                                timestamp=ts,
                                level="INFO",
                                message=f"[investigation] Deep failure output successfully retrieved for task {task_key} (task_run_id={task_run_id}).",
                                source="_investigation",
                            ))

                    except Exception as task_out_err:
                        logger.warning(
                            "Could not fetch get-output for task %s (run_id=%s): %s",
                            task_key, task_run_id, task_out_err,
                        )
                        logs.append(NormalizedLog(
                            timestamp=ts,
                            level="INFO",
                            message=f"[investigation] Task output retrieval failed for {task_key} (run_id={task_run_id}): {str(task_out_err)[:200]}",
                            source="_investigation",
                        ))

        # ------------------------------------------------------------------
        # Case B: SINGLE-TASK Job (legacy single-task container)
        # ------------------------------------------------------------------
        else:
            try:
                output = self._get_output_with_progressive_retry(run_external_id)
                for stream_name in ("error", "error_trace", "logs"):
                    content = output.get(stream_name)
                    if content:
                        level = "ERROR" if stream_name != "logs" else "INFO"
                        logs.append(NormalizedLog(
                            timestamp=ts,
                            level=level,
                            message=str(content).strip()[:10000],
                            source=stream_name,
                        ))

                nb_out = output.get("notebook_output")
                if isinstance(nb_out, dict) and nb_out.get("result"):
                    logs.append(NormalizedLog(
                        timestamp=ts,
                        level="INFO",
                        message=str(nb_out["result"]).strip()[:8000],
                        source="notebook_output",
                    ))
            except Exception as single_err:
                logger.debug("Single-task get-output skipped/failed: %s", single_err)

        return logs
