"""
Real integration tests for the Git (GitHub) connector.

Hits the real GitHub API. Read-only by default; the auto-fix test is gated
behind RUN_DESTRUCTIVE_TESTS=1 because it files a real issue.

Requires:
  GITHUB_TOKEN (PAT with `repo` + `actions:read`)
  GITHUB_OWNER
  GITHUB_REPO     (optional; required for the destructive test)

Run with:
  pytest -m integration tests/integration/test_git_real.py
  # to also run the destructive auto-fix:
  RUN_DESTRUCTIVE_TESTS=1 pytest -m "integration or destructive" tests/integration/test_git_real.py
"""
import os

import pytest

from app.connectors.base import NormalizedPipeline, NormalizedRun, NormalizedLog


pytestmark = [pytest.mark.integration]


class TestGitConnection:
    def test_test_connection_succeeds(self, git_connector):
        ok, msg = git_connector.test_connection()
        assert ok, f"GitHub connection failed: {msg}"
        assert "Authenticated as" in msg

    def test_test_connection_with_bad_token_fails(self, git_credentials):
        from app.connectors import GitConnector
        bad = {**git_credentials, "token": "ghp_clearly_invalid_token_value"}
        ok, msg = GitConnector(bad).test_connection()
        assert ok is False


class TestGitListWorkflows:
    def test_list_pipelines_returns_workflows(self, git_connector):
        pipelines = git_connector.list_pipelines()
        assert isinstance(pipelines, list)
        if not pipelines:
            pytest.skip(
                "No GitHub Actions workflows found under the owner/repo"
            )
        for p in pipelines:
            assert isinstance(p, NormalizedPipeline)
            # external_id is "<repo>:<workflow_id>"
            assert ":" in p.external_id
            repo, wf_id = p.external_id.split(":", 1)
            assert repo
            assert wf_id.isdigit()
            assert p.metadata.get("workflow_id") is not None


class TestGitRunsAndLogs:
    @pytest.fixture
    def workflow_with_runs(self, git_connector):
        for p in git_connector.list_pipelines():
            runs = git_connector.list_runs(p.external_id, limit=5)
            if runs:
                return p, runs
        pytest.skip("No workflow has runs")

    def test_list_runs_shape(self, workflow_with_runs):
        _, runs = workflow_with_runs
        for r in runs:
            assert isinstance(r, NormalizedRun)
            assert r.external_run_id
            assert r.status in {
                "QUEUED", "RUNNING", "SUCCEEDED", "FAILED",
                "CANCELLED", "UNKNOWN",
            }

    def test_get_logs_returns_list(self, git_connector, workflow_with_runs):
        pipeline, runs = workflow_with_runs
        logs = git_connector.get_logs(pipeline.external_id, runs[0].external_run_id)
        assert isinstance(logs, list)
        for log in logs:
            assert isinstance(log, NormalizedLog)


# ---------------------------------------------------------------------------
# Destructive: actually files a GitHub issue. Off by default.
# ---------------------------------------------------------------------------
class TestGitAutoFix:
    @pytest.mark.destructive
    def test_apply_fix_files_a_real_issue(self, git_connector, git_credentials):
        if not git_credentials.get("repo"):
            pytest.skip("GITHUB_REPO required for auto-fix test")
        assert git_connector.supports_auto_fix()

        # Pick any pipeline external_id - apply_fix doesn't actually need a
        # valid one because the GitHub connector files an issue, it doesn't
        # touch the workflow file.
        pipelines = git_connector.list_pipelines()
        pid = pipelines[0].external_id if pipelines else "dummy:0"

        sample_patch = (
            "# Sample fix patch from pipeline-monitor integration test\n"
            "# (this issue was filed automatically; safe to close)\n"
            "- name: Fix env var\n"
            "  run: echo 'PATH=/usr/local/bin:$PATH' >> $GITHUB_ENV\n"
        )
        ok, msg = git_connector.apply_fix(pid, sample_patch)
        assert ok, f"apply_fix failed: {msg}"
        assert "issue" in msg.lower() or "https://" in msg
