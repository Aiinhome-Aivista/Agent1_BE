"""
Real integration tests for the ADF connector.

These tests authenticate to Azure with a service principal and read from a real
Data Factory. They are read-only: list pipelines, list runs, fetch logs.

Requires env vars (see .env.test.example):
  AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET,
  AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_FACTORY_NAME

Run with:
  pytest -m integration tests/integration/test_adf_real.py
"""
import pytest

from app.connectors.base import NormalizedPipeline, NormalizedRun, NormalizedLog


pytestmark = [pytest.mark.integration]


class TestADFConnection:
    def test_test_connection_succeeds(self, adf_connector):
        """Service principal can authenticate and the factory exists."""
        ok, msg = adf_connector.test_connection()
        assert ok, f"ADF connection failed: {msg}"
        assert "Connected to Data Factory" in msg

    def test_test_connection_with_bad_secret_fails_cleanly(self, adf_credentials):
        """Wrong secret returns (False, message), doesn't raise."""
        from app.connectors import ADFConnector
        bad = {**adf_credentials, "client_secret": "definitely-wrong"}
        ok, msg = ADFConnector(bad).test_connection()
        assert ok is False
        assert msg  # non-empty human message


class TestADFListPipelines:
    def test_list_pipelines_returns_normalized_shape(self, adf_connector):
        pipelines = adf_connector.list_pipelines()
        assert isinstance(pipelines, list)
        # The factory may legitimately be empty; accept that and skip deeper
        # assertions if so.
        if not pipelines:
            pytest.skip("Target factory has no pipelines")
        for p in pipelines:
            assert isinstance(p, NormalizedPipeline)
            assert p.external_id, "pipeline must have an external_id"
            assert p.name, "pipeline must have a name"
            assert isinstance(p.metadata, dict)


class TestADFRunsAndLogs:
    """List runs & logs for the first pipeline that has any."""

    @pytest.fixture
    def pipeline_with_runs(self, adf_connector):
        for p in adf_connector.list_pipelines():
            runs = adf_connector.list_runs(p.external_id, limit=5)
            if runs:
                return p, runs
        pytest.skip("No pipeline in the factory has runs in the last 7 days")

    def test_list_runs_returns_normalized_shape(self, pipeline_with_runs):
        _, runs = pipeline_with_runs
        for r in runs:
            assert isinstance(r, NormalizedRun)
            assert r.external_run_id
            assert r.status in {
                "QUEUED", "RUNNING", "SUCCEEDED", "FAILED",
                "CANCELLED", "UNKNOWN",
            }

    def test_get_logs_returns_list(self, adf_connector, pipeline_with_runs):
        pipeline, runs = pipeline_with_runs
        logs = adf_connector.get_logs(pipeline.external_id, runs[0].external_run_id)
        # ADF activity-run logs may legitimately be empty for very old runs;
        # the shape contract still applies.
        assert isinstance(logs, list)
        for log in logs:
            assert isinstance(log, NormalizedLog)
            assert log.message
            assert log.level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
