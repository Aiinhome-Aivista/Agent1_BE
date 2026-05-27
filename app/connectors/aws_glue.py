"""
AWS Glue connector.

Authenticates via AWS Access Key ID + Secret Access Key.
Credentials shape: {aws_access_key_id, aws_secret_access_key, region_name}
"""
from __future__ import annotations

import logging
from typing import Any

import boto3

from app.connectors.base import (
    BaseConnector, NormalizedPipeline, NormalizedRun, NormalizedLog,
)

logger = logging.getLogger(__name__)


class AWSGlueConnector(BaseConnector):
    """Connector for AWS Glue jobs."""

    type_name = "AWS_GLUE"

    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self.glue = boto3.client(
            "glue",
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"],
            region_name=credentials.get("region_name", "ap-south-1"),
        )

    def test_connection(self) -> tuple[bool, str]:
        try:
            jobs = self.glue.get_jobs(MaxResults=1)
            count = len(jobs.get("Jobs", []))
            return True, f"AWS Glue connected successfully ({count} job(s) found)"
        except Exception as exc:
            logger.warning("AWS Glue connection test failed: %s", exc)
            return False, str(exc)

    def list_pipelines(self) -> list[NormalizedPipeline]:
        response = self.glue.get_jobs()
        pipelines = []
        for job in response.get("Jobs", []):
            pipelines.append(NormalizedPipeline(
                external_id=job["Name"],
                name=job["Name"],
                description=job.get("Description"),
                metadata={
                    "glue_version": job.get("GlueVersion"),
                    "worker_type": job.get("WorkerType"),
                },
            ))
        return pipelines

    def list_runs(self, pipeline_external_id: str, limit: int = 25) -> list[NormalizedRun]:
        response = self.glue.get_job_runs(JobName=pipeline_external_id, MaxResults=limit)
        runs = []
        for run in response.get("JobRuns", []):
            state = run.get("JobRunState", "UNKNOWN")
            status = _glue_state_to_status(state)
            runs.append(NormalizedRun(
                external_run_id=run["Id"],
                status=status,
                started_at=run.get("StartedOn"),
                finished_at=run.get("CompletedOn"),
                duration_seconds=run.get("ExecutionTime"),
                error_message=run.get("ErrorMessage"),
                raw=run,
            ))
        return runs

    def get_logs(self, pipeline_external_id: str, run_external_id: str) -> list[NormalizedLog]:
        # Glue logs live in CloudWatch; returning empty list until that is wired up
        return []

    # Legacy helpers used by the /aws-glue/* REST endpoints
    def get_jobs(self):
        response = self.glue.get_jobs()
        return [
            {
                "name": job.get("Name"),
                "glue_version": job.get("GlueVersion"),
                "worker_type": job.get("WorkerType"),
            }
            for job in response.get("Jobs", [])
        ]

    def get_job_runs(self, job_name: str):
        response = self.glue.get_job_runs(JobName=job_name, MaxResults=10)
        return [
            {
                "run_id": run.get("Id"),
                "status": run.get("JobRunState"),
                "started_on": str(run.get("StartedOn")),
                "completed_on": str(run.get("CompletedOn")),
                "execution_time": run.get("ExecutionTime"),
                "error_message": run.get("ErrorMessage"),
            }
            for run in response.get("JobRuns", [])
        ]


def _glue_state_to_status(state: str) -> str:
    return {
        "STARTING": "QUEUED",
        "RUNNING": "RUNNING",
        "STOPPING": "RUNNING",
        "STOPPED": "CANCELLED",
        "SUCCEEDED": "SUCCEEDED",
        "FAILED": "FAILED",
        "TIMEOUT": "FAILED",
        "ERROR": "FAILED",
        "WAITING": "QUEUED",
    }.get(state, "UNKNOWN")