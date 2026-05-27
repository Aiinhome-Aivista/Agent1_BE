"""
Git connector.

Currently supports GitHub Actions workflow runs (the most common 'pipeline' on
git providers). Each GitHub Actions workflow becomes a Pipeline; each workflow
run becomes a PipelineRun; the run logs become PipelineLogs.

Auth: Personal Access Token with `repo` + `actions:read` scopes.
"""
from __future__ import annotations

import logging
import zipfile
import io
from datetime import datetime, timezone
from typing import Any

import httpx

from app.connectors.base import (
    BaseConnector, NormalizedPipeline, NormalizedRun, NormalizedLog,
)

logger = logging.getLogger(__name__)


_GH_STATUS_MAP = {
    ("queued", None): "QUEUED",
    ("in_progress", None): "RUNNING",
    ("completed", "success"): "SUCCEEDED",
    ("completed", "failure"): "FAILED",
    ("completed", "timed_out"): "FAILED",
    ("completed", "cancelled"): "CANCELLED",
    ("completed", "skipped"): "CANCELLED",
}


def _gh_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class GitConnector(BaseConnector):
    type_name = "GIT"

    API_ROOT = "https://api.github.com"

    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self.token = credentials["token"]
        self.owner = credentials["owner"]
        self.repo = credentials.get("repo")  # may be None -> all repos

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _get(self, path: str, params: dict | None = None) -> Any:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(f"{self.API_ROOT}{path}", headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    def _repos(self) -> list[str]:
        if self.repo:
            return [self.repo]
        # list all repos accessible to the token within the owner
        try:
            data = self._get(f"/users/{self.owner}/repos", params={"per_page": 50})
            return [r["name"] for r in data]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def test_connection(self) -> tuple[bool, str]:
        try:
            user = self._get("/user")
            if self.repo:
                repo = self._get(f"/repos/{self.owner}/{self.repo}")
                return True, f"Authenticated as {user.get('login')}; repo {repo.get('full_name')} accessible."
            return True, f"Authenticated as {user.get('login')}"
        except httpx.HTTPStatusError as e:
            return False, f"GitHub auth failed: HTTP {e.response.status_code}"
        except Exception as e:
            return False, f"GitHub connection failed: {e}"

    def list_pipelines(self) -> list[NormalizedPipeline]:
        out: list[NormalizedPipeline] = []
        for repo in self._repos():
            try:
                data = self._get(f"/repos/{self.owner}/{repo}/actions/workflows")
            except Exception as e:
                logger.warning("Skipping workflow list for %s/%s: %s", self.owner, repo, e)
                continue
            for wf in data.get("workflows", []):
                # external_id encodes the repo too so list_runs knows where to look
                eid = f"{repo}:{wf['id']}"
                out.append(NormalizedPipeline(
                    external_id=eid,
                    name=f"{repo} / {wf['name']}",
                    description=wf.get("path"),
                    metadata={
                        "repo": repo,
                        "workflow_id": wf["id"],
                        "state": wf.get("state"),
                        "html_url": wf.get("html_url"),
                    },
                ))
        return out

    def _split_eid(self, external_id: str) -> tuple[str, str]:
        repo, wf_id = external_id.split(":", 1)
        return repo, wf_id

    def list_runs(self, pipeline_external_id: str, limit: int = 25) -> list[NormalizedRun]:
        repo, wf_id = self._split_eid(pipeline_external_id)
        data = self._get(
            f"/repos/{self.owner}/{repo}/actions/workflows/{wf_id}/runs",
            params={"per_page": min(limit, 50)},
        )
        runs: list[NormalizedRun] = []
        for r in data.get("workflow_runs", []):
            status = _GH_STATUS_MAP.get((r.get("status"), r.get("conclusion")), "UNKNOWN")
            started = _gh_iso(r.get("run_started_at") or r.get("created_at"))
            ended = _gh_iso(r.get("updated_at")) if r.get("status") == "completed" else None
            duration = (ended - started).total_seconds() if started and ended else None
            runs.append(NormalizedRun(
                external_run_id=str(r["id"]),
                status=status,
                started_at=started,
                finished_at=ended,
                duration_seconds=duration,
                error_message=None,
                raw={
                    "name": r.get("name"),
                    "event": r.get("event"),
                    "head_branch": r.get("head_branch"),
                    "head_sha": r.get("head_sha"),
                    "html_url": r.get("html_url"),
                    "actor": (r.get("actor") or {}).get("login"),
                    "repo": repo,
                },
            ))
        return runs

    def get_logs(self, pipeline_external_id: str, run_external_id: str) -> list[NormalizedLog]:
        repo, _ = self._split_eid(pipeline_external_id)
        logs: list[NormalizedLog] = []

        # Per-job statuses
        try:
            jobs_data = self._get(
                f"/repos/{self.owner}/{repo}/actions/runs/{run_external_id}/jobs",
            )
            for job in jobs_data.get("jobs", []):
                started = _gh_iso(job.get("started_at"))
                level = "ERROR" if job.get("conclusion") in ("failure", "timed_out") else "INFO"
                logs.append(NormalizedLog(
                    timestamp=started or datetime.now(timezone.utc),
                    level=level,
                    message=f"[job] {job.get('name')} -> "
                            f"{job.get('status')} / {job.get('conclusion')}",
                    source=job.get("name"),
                ))
                for step in job.get("steps", []) or []:
                    if step.get("conclusion") in ("failure", "timed_out"):
                        ts = _gh_iso(step.get("started_at")) or started or datetime.now(timezone.utc)
                        logs.append(NormalizedLog(
                            timestamp=ts, level="ERROR",
                            message=f"step '{step.get('name')}' failed",
                            source=f"{job.get('name')}/{step.get('name')}",
                        ))
        except Exception as e:
            logger.warning("GitHub jobs fetch failed: %s", e)

        # Raw text logs (zip archive of per-step .txt)
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                r = client.get(
                    f"{self.API_ROOT}/repos/{self.owner}/{repo}/actions/runs/{run_external_id}/logs",
                    headers=self._headers(),
                )
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/zip"):
                    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                        # Take the first ~5 files to avoid huge payloads
                        for name in zf.namelist()[:5]:
                            try:
                                content = zf.read(name).decode("utf-8", errors="replace")
                                logs.append(NormalizedLog(
                                    timestamp=datetime.now(timezone.utc),
                                    level="INFO",
                                    message=content[-8000:],  # tail
                                    source=name,
                                ))
                            except Exception:
                                continue
        except Exception as e:
            logger.warning("GitHub raw logs fetch failed: %s", e)

        return logs

    # --- auto-fix support -------------------------------------------------
    def supports_auto_fix(self) -> bool:
        return bool(self.repo)  # need a target repo

    def apply_fix(self, pipeline_external_id: str, patch: str) -> tuple[bool, str]:
        """Open a PR with the suggested patch.

        For safety we never push directly to the default branch. Instead we
        create a fix branch and open a PR. The `patch` is expected to be a
        unified diff or a fenced ```yaml block; we store it as-is in the PR
        body so a human can review.
        """
        # NOTE: applying a unified diff programmatically is non-trivial across
        # all edge cases; we deliberately keep this conservative: open a PR
        # describing the suggested fix so a human reviews and merges.
        try:
            with httpx.Client(timeout=30.0) as client:
                # Get default branch (used for context in the issue body)
                repo_info = client.get(
                    f"{self.API_ROOT}/repos/{self.owner}/{self.repo}",
                    headers=self._headers(),
                ).json()
                default_branch = repo_info.get("default_branch", "main")
                title = "[pipeline-monitor] Suggested fix from LLM analysis"
                body = (
                    f"An LLM analysis of a recent failed run produced this "
                    f"suggested fix (target branch: `{default_branch}`).\n\n"
                    "Please review carefully before merging.\n\n"
                    "```\n" + patch[:50000] + "\n```"
                )
                # Create an issue (PR creation requires a branch with commits;
                # we choose an issue as the safe default)
                r = client.post(
                    f"{self.API_ROOT}/repos/{self.owner}/{self.repo}/issues",
                    headers=self._headers(),
                    json={"title": title, "body": body, "labels": ["pipeline-monitor"]},
                )
                r.raise_for_status()
                issue_url = r.json().get("html_url")
                return True, f"Filed issue with suggested fix: {issue_url}"
        except Exception as e:
            return False, f"Failed to file fix issue: {e}"
