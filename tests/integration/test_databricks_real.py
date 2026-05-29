"""
Real integration tests for the Databricks connector.

Hits the real Jobs 2.1 API on the configured workspace. Read-only.

Requires:
  DATABRICKS_WORKSPACE_URL, DATABRICKS_TOKEN

Run with:
  pytest -m integration tests/integration/test_databricks_real.py
"""
import pytest

from app.connectors.base import NormalizedPipeline, NormalizedRun, NormalizedLog


pytestmark = [pytest.mark.integration]


class TestDatabricksConnection:
    def test_test_connection_succeeds(self, databricks_connector):
        ok, msg = databricks_connector.test_connection()
        assert ok, f"Databricks connection failed: {msg}"
        assert "Connected as" in msg

    def test_test_connection_with_bad_token_fails(self, databricks_credentials):
        from app.connectors import DatabricksConnector
        bad = {**databricks_credentials, "personal_access_token": "dapi-bad"}
        ok, msg = DatabricksConnector(bad).test_connection()
        assert ok is False


class TestDatabricksListJobs:
    def test_list_pipelines_returns_jobs(self, databricks_connector):
        pipelines = databricks_connector.list_pipelines()
        assert isinstance(pipelines, list)
        if not pipelines:
            pytest.skip("Workspace has no Databricks jobs")
        for p in pipelines:
            assert isinstance(p, NormalizedPipeline)
            # Databricks job_id is numeric in the API; we stringify it
            assert p.external_id
            assert p.external_id.lstrip("-").isdigit()
            assert p.name


class TestDatabricksRunsAndLogs:
    @pytest.fixture
    def job_with_runs(self, databricks_connector):
        for p in databricks_connector.list_pipelines():
            runs = databricks_connector.list_runs(p.external_id, limit=5)
            if runs:
                return p, runs
        pytest.skip("No Databricks job has runs")

    def test_list_runs_shape(self, job_with_runs):
        _, runs = job_with_runs
        for r in runs:
            assert isinstance(r, NormalizedRun)
            assert r.external_run_id
            assert r.status in {
                "QUEUED", "RUNNING", "SUCCEEDED", "FAILED",
                "CANCELLED", "UNKNOWN",
            }
            # Successful runs should have a started_at
            if r.status == "SUCCEEDED":
                assert r.started_at is not None

    def test_get_logs_returns_list(self, databricks_connector, job_with_runs):
        pipeline, runs = job_with_runs
        logs = databricks_connector.get_logs(pipeline.external_id, runs[0].external_run_id)
        assert isinstance(logs, list)
        for log in logs:
            assert isinstance(log, NormalizedLog)
            assert log.message
