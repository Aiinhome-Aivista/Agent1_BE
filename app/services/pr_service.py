"""
PR orchestration service.

Given an incident that the agent has classified as a KNOWN, auto-fixable error,
this service:

  1. builds the source connector for the incident's pipeline,
  2. asks code_fix_service for a concrete change (reusing an accepted fix when
     one exists, else asking the LLM),
  3. opens a real PR via the Git connector (branch + commit + PR), and
  4. records the generated fix on the SolutionPattern.

It degrades gracefully: if the connector isn't a Git repo, or credentials are
missing, it returns a structured "skipped" result and the incident pipeline
falls back to the notify-only path.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.connectors import get_connector
from app.connectors.git import GitConnector
from app.core.security import decrypt_secret
from app.models import Pipeline
from app.models.agent_models import Incident
from app.models.solution_models import SolutionPattern
from app.services import code_fix_service
from app.services.solution_kb_service import solution_kb_service

logger = logging.getLogger(__name__)


def _decrypt_creds(connector) -> dict[str, Any]:
    return json.loads(decrypt_secret(connector.encrypted_credentials))


def raise_pr_for_incident(
    db: Session,
    incident: Incident,
    *,
    pattern: SolutionPattern | None = None,
) -> dict[str, Any]:
    pipe: Pipeline | None = (
        db.query(Pipeline).filter(Pipeline.id == incident.pipeline_id).first()
        if incident.pipeline_id else None
    )
    if pipe is None or pipe.connector is None:
        return {"ok": False, "skipped": True,
                "reason": "incident has no linked pipeline/connector"}

    connector_model = pipe.connector
    try:
        creds = _decrypt_creds(connector_model)
        client = get_connector(connector_model.type, creds)
    except Exception as e:
        logger.warning("PR: connector build failed: %s", e)
        return {"ok": False, "skipped": True, "reason": f"connector error: {e}"}

    if not isinstance(client, GitConnector) or not client.supports_auto_fix():
        return {"ok": False, "skipped": True,
                "reason": "auto-PR only supported for Git connectors with a target repo"}

    # Try to give the LLM the current contents of the most likely file.
    existing_content = None
    repo_hint = getattr(client, "repo", None)

    fix = code_fix_service.generate_fix(
        pattern=pattern,
        error_text=incident.error_log or "",
        root_cause=incident.root_cause or "",
        fix_steps=incident.remediation_plan or [],
        category=(pattern.category if pattern else "General"),
        repo_hint=repo_hint,
        existing_file_content=existing_content,
    )

    if not fix.is_committable:
        # We have a diagnosis but no committable file change → file an issue
        # so a human can act (and later ingest their PR back into the KB).
        patch_text = fix.diff or fix.explanation or "\n".join(incident.remediation_plan or [])
        ok, msg = client.apply_fix(
            pipeline_external_id=pipe.external_id or "",
            patch=patch_text,
        )
        return {"ok": ok, "skipped": False, "mode": "issue",
                "message": msg, "fix_source": fix.source}

    title = f"[dataops-agent] Fix: {(pattern.title if pattern else incident.pipeline_name)[:80]}"
    body = (
        f"Automated fix proposed by the DataOps agent for incident "
        f"#{incident.id} on **{incident.pipeline_name}**.\n\n"
        f"**Root cause**\n{incident.root_cause or '(see incident)'}\n\n"
        f"**What changed**\n{fix.explanation}\n\n"
        f"**Fix source**: `{fix.source}`  ·  **Agent confidence**: "
        f"{(pattern.confidence if pattern else 0):.0%}\n\n"
        f"> Please review carefully before merging. After merge, the agent can "
        f"ingest this PR back into its knowledge base so the same error is "
        f"auto-fixed next time."
    )

    ok, url_or_msg = client.open_fix_pr(
        file_path=fix.file_path,        # type: ignore[arg-type]
        new_content=fix.new_content,    # type: ignore[arg-type]
        title=title,
        body=body,
    )

    # Persist the generated fix on the pattern for reuse.
    if pattern is not None and fix.source == "llm":
        try:
            solution_kb_service.attach_llm_fix(
                db,
                pattern_id=pattern.id,
                file_path=fix.file_path,
                new_content=fix.new_content,
                diff=fix.diff,
                explanation=fix.explanation,
                language=fix.language,
            )
        except Exception:
            logger.debug("attach_llm_fix failed", exc_info=True)

    return {
        "ok": ok, "skipped": False, "mode": "pr",
        "pr_url": url_or_msg if ok else None,
        "message": url_or_msg,
        "fix_source": fix.source,
        "file_path": fix.file_path,
    }
