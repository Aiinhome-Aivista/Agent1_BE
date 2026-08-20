"""
Azure Data Factory connector.

Auth: Azure AD service principal (tenant_id + client_id + client_secret).
Uses azure-mgmt-datafactory SDK.

Required Azure RBAC: the service principal needs at least the
"Data Factory Contributor" role on the target factory (or read at minimum,
"Reader" + "Data Factory Contributor" if we want to also re-trigger runs).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
try:
    # pyrefly: ignore [missing-import]
    from azure.identity import ClientSecretCredential
    # pyrefly: ignore [missing-import]
    from azure.mgmt.datafactory import DataFactoryManagementClient
    # pyrefly: ignore [missing-import]
    from azure.mgmt.datafactory.models import RunFilterParameters
except ImportError:
    ClientSecretCredential = None  # type: ignore
    DataFactoryManagementClient = None  # type: ignore
    RunFilterParameters = None  # type: ignore

from app.connectors.base import (
    BaseConnector, NormalizedPipeline, NormalizedRun, NormalizedLog,
)

logger = logging.getLogger(__name__)


_STATUS_MAP = {
    "Queued": "QUEUED",
    "InProgress": "RUNNING",
    "Succeeded": "SUCCEEDED",
    "Failed": "FAILED",
    "Cancelling": "RUNNING",
    "Cancelled": "CANCELLED",
}


class ADFConnector(BaseConnector):
    type_name = "ADF"

    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self._client: DataFactoryManagementClient | None = None

    def _client_or_create(self) -> DataFactoryManagementClient:
        if self._client is None:
            cred = ClientSecretCredential(
                tenant_id=self.credentials["tenant_id"],
                client_id=self.credentials["client_id"],
                client_secret=self.credentials["client_secret"],
            )
            self._client = DataFactoryManagementClient(
                credential=cred,
                subscription_id=self.credentials["subscription_id"],
            )
        return self._client

    # ------------------------------------------------------------------
    def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._client_or_create()
            factory = client.factories.get(
                resource_group_name=self.credentials["resource_group"],
                factory_name=self.credentials["factory_name"],
            )
            return True, f"Connected to Data Factory '{factory.name}' in {factory.location}"
        except Exception as exc:
            logger.exception("ADF connection test failed")
            return False, f"ADF connection failed: {exc}"

    def list_pipelines(self) -> list[NormalizedPipeline]:
        client = self._client_or_create()
        rg = self.credentials["resource_group"]
        fn = self.credentials["factory_name"]

        result: list[NormalizedPipeline] = []
        for p in client.pipelines.list_by_factory(rg, fn):
            result.append(NormalizedPipeline(
                external_id=p.name,
                name=p.name,
                description=p.description,
                metadata={
                    "activities": [a.name for a in (p.activities or [])],
                    "folder": p.folder.name if p.folder else None,
                },
            ))
        return result

    def list_runs(self, pipeline_external_id: str, limit: int = 25) -> list[NormalizedRun]:
        client = self._client_or_create()
        rg = self.credentials["resource_group"]
        fn = self.credentials["factory_name"]

        # ADF requires a time window - fetch the last 7 days
        now = datetime.now(timezone.utc)
        flt = RunFilterParameters(
            last_updated_after=now - timedelta(days=7),
            last_updated_before=now,
            filters=[{"operand": "PipelineName", "operator": "Equals",
                      "values": [pipeline_external_id]}],
        )
        page = client.pipeline_runs.query_by_factory(rg, fn, flt)

        runs: list[NormalizedRun] = []
        for r in (page.value or [])[:limit]:
            duration = None
            if r.duration_in_ms:
                duration = r.duration_in_ms / 1000.0
            runs.append(NormalizedRun(
                external_run_id=r.run_id,
                status=_STATUS_MAP.get(r.status, "UNKNOWN"),
                started_at=r.run_start,
                finished_at=r.run_end,
                duration_seconds=duration,
                error_message=r.message if r.status == "Failed" else None,
                raw={
                    "pipeline_name": r.pipeline_name,
                    "invoked_by": r.invoked_by.name if r.invoked_by else None,
                    "parameters": r.parameters,
                },
            ))
        return runs

    def get_logs(self, pipeline_external_id: str, run_external_id: str) -> list[NormalizedLog]:
        """ADF doesn't have line-by-line logs through the management API; the
        closest equivalent is per-activity status + error messages."""
        client = self._client_or_create()
        rg = self.credentials["resource_group"]
        fn = self.credentials["factory_name"]

        now = datetime.now(timezone.utc)
        flt = RunFilterParameters(
            last_updated_after=now - timedelta(days=7),
            last_updated_before=now,
        )
        try:
            page = client.activity_runs.query_by_pipeline_run(rg, fn, run_external_id, flt)
        except Exception as exc:
            logger.warning("Failed to fetch ADF activity runs: %s", exc)
            return []

        logs: list[NormalizedLog] = []
        for a in (page.value or []):
            ts = a.activity_run_start or datetime.now(timezone.utc)
            level = "ERROR" if a.status == "Failed" else "INFO"
            msg = f"[{a.activity_type}] {a.activity_name} -> {a.status}"
            if a.error and isinstance(a.error, dict):
                err = a.error.get("message") or a.error.get("Message")
                if err:
                    msg += f"\n{err}"
                    level = "ERROR"
            logs.append(NormalizedLog(
                timestamp=ts, level=level, message=msg, source=a.activity_name,
            ))
        return logs
