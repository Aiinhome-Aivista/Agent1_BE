"""
LLM-assisted runbook metadata extraction.

When a user uploads a runbook (PDF / DOCX / MD / TXT) we send the extracted
text to Mistral and ask it to suggest:

    title         -- a short, human-friendly headline
    category      -- one of: ADF, Databricks, Git, AWS Glue
    description   -- 1-3 sentence summary of when this runbook applies
    steps         -- ordered list of remediation steps (max ~10)
    risk_level    -- Low / Medium / High
    tags          -- a few comma-friendly keywords

The frontend pre-fills its form from this. The user reviews / edits and
then commits, at which point the file goes through the existing chunk +
embed + Chroma upsert pipeline.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.mistral_service import mistral_service

logger = logging.getLogger(__name__)


# These are the only categories the UI accepts — keep the LLM honest by
# constraining its output too.
ALLOWED_CATEGORIES: list[str] = ["ADF", "Databricks", "Git", "AWS Glue"]

# Cap how much text we send to Mistral. all-MiniLM-L6-v2 can chunk the rest
# at index time; for the suggestion step we just need a representative slice
# of the document so token cost stays bounded.
MAX_INPUT_CHARS = 12_000


_SYSTEM_PROMPT = """You are a senior SRE who curates a library of operational
runbooks for a data-platform team. The user has just uploaded a document.
Read it, then return a JSON object that proposes how this runbook should
be catalogued. Do NOT invent steps that are not in the document; if a
field cannot be inferred, return an empty string or empty list for it.

Respond with a SINGLE JSON object (no markdown fences) matching exactly:

{
  "title":       string,           // <= 80 chars, human-friendly
  "category":    string,           // EXACTLY one of: "ADF", "Databricks", "Git", "AWS Glue"
  "description": string,           // 1-3 sentences, what symptoms this remediates
  "steps":       [string, ...],    // ordered remediation steps, plain text, max 10
  "risk_level":  string,           // "Low" | "Medium" | "High"
  "tags":           [string, ...],    // 3-6 short keywords, lowercase, no spaces inside a tag
  "relevance_score": integer          // 0-100: how relevant this runbook is to a DataOps/pipeline
                                      // orchestration system. 90-100 = perfect fit (pipeline SOPs,
                                      // Airflow/Dagster/ADF/Databricks runbooks). 50-70 = loosely
                                      // related. Below 40 = not relevant (e.g. HR policy, legal docs).
}

Category rules:
- "ADF"        → Azure Data Factory pipelines, integration runtimes, linked services
- "Databricks" → Databricks jobs, clusters, notebooks, Spark on Databricks
- "Git"        → Git / GitHub / GitLab / CI workflows, repo permissions, merge issues
- "AWS Glue"   → AWS Glue jobs, crawlers, Glue catalog, EMR-on-Glue

If the document does not clearly fit any of these four, pick the closest one
and lower the risk_level to "Low" to flag uncertainty."""


def _truncate_for_prompt(text: str) -> str:
    """Take a representative slice: head + tail. Heads usually contain
    title + intro, tails often contain the verification/rollback steps."""
    if len(text) <= MAX_INPUT_CHARS:
        return text
    head = text[: MAX_INPUT_CHARS - 2000]
    tail = text[-2000:]
    return f"{head}\n\n[... document truncated ...]\n\n{tail}"


def _strip_fences(raw: str) -> str:
    """Mistral sometimes wraps the JSON in ```json fences anyway."""
    s = raw.strip()
    if s.startswith("```json"):
        s = s[len("```json"):].lstrip()
    elif s.startswith("```"):
        s = s[3:].lstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    return s.strip()


def _coerce_category(value: Any) -> str:
    """Normalise to one of the allowed categories, with fuzzy fallback."""
    if not isinstance(value, str):
        return "ADF"

    v = value.strip()
    for allowed in ALLOWED_CATEGORIES:
        if v.lower() == allowed.lower():
            return allowed

    # fuzzy: substring match
    lower = v.lower()
    if "adf" in lower or "azure data factory" in lower or "data factory" in lower:
        return "ADF"
    if "databricks" in lower or "spark" in lower:
        return "Databricks"
    if "glue" in lower or "aws" in lower:
        return "AWS Glue"
    if "git" in lower or "github" in lower or "gitlab" in lower:
        return "Git"

    return "ADF"


def _coerce_risk(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().capitalize()
        if v in {"Low", "Medium", "High"}:
            return v
    return "Medium"


def _coerce_steps(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:10]:
        if isinstance(item, str):
            cleaned = re.sub(r"^\s*\d+[.)\s]\s*", "", item).strip()
            if cleaned:
                out.append(cleaned[:400])
    return out


def _coerce_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:8]:
        if isinstance(item, str):
            tag = item.strip().lower().replace(" ", "-")
            if tag:
                out.append(tag[:40])
    return out


def _coerce_relevance_score(value: Any) -> int:
    """Clamp LLM-returned score to 0-100 integer."""
    try:
        score = int(value)
        return max(0, min(100, score))
    except (TypeError, ValueError):
        return 50  # neutral default if LLM returns garbage

def _fallback_from_filename(filename: str | None) -> dict[str, Any]:
    """If Mistral is unreachable, do something reasonable so upload still works."""
    stem = (filename or "Untitled runbook").rsplit(".", 1)[0]
    pretty = re.sub(r"[_\-]+", " ", stem).strip().title() or "Untitled runbook"
    return {
        "title":       pretty[:80],
        "category":    "ADF",
        "description": "",
        "steps":       [],
        "risk_level":  "Medium",
        "tags":        [],
        "llm_used":    False,
    }


def suggest_metadata(
    text: str,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Send the extracted document text to Mistral and return suggested
    runbook metadata. Always returns a valid dict — falls back to a
    filename-derived stub if the LLM call fails or returns garbage.
    """
    if not text or not text.strip():
        return _fallback_from_filename(filename)

    snippet = _truncate_for_prompt(text.strip())

    # We reuse mistral_service.analyze_failure because it already has retries,
    # JSON-fence stripping, latency tracking, etc. Pretend the "logs" are the
    # document body. The system prompt above redirects the model away from
    # the diagnosis schema and onto the metadata schema.
    try:
        result = mistral_service.analyze_failure(
            pipeline_name="runbook-ingestion",
            connector_type="runbook",
            error_message="Suggest catalogue metadata for the uploaded runbook below.",
            logs=[{
                "timestamp": "",
                "level":     "INFO",
                "source":    filename or "uploaded_runbook",
                "message":   snippet,
            }],
            metadata={"task": "runbook_metadata_suggestion"},
            context_block=_SYSTEM_PROMPT,
        )
    except Exception:
        logger.exception("Mistral call failed for runbook metadata; using fallback")
        return _fallback_from_filename(filename)

    raw = result.get("raw_response") or {}

    # `mistral_service.analyze_failure` is built for the diagnose-and-fix
    # JSON schema, but it just forwards whatever JSON the model emits as
    # raw_response. So our keys live there directly.
    if not isinstance(raw, dict):
        return _fallback_from_filename(filename)

    # Defensive: some calls return the diagnose-schema, so fall back on those
    # fields if our prompt-format wasn't honoured. Worst case we still get
    # something usable.
    title = (
        raw.get("title")
        or raw.get("summary")
        or (filename.rsplit(".", 1)[0] if filename else "Untitled runbook")
    )

    description = (
        raw.get("description")
        or raw.get("root_cause")
        or ""
    )

    # Some models emit steps as a single string with newlines instead of an
    # actual list — accept both.
    steps_field = raw.get("steps") or raw.get("suggested_fix") or []
    if isinstance(steps_field, str):
        steps_field = [
            re.sub(r"^\s*\d+[.)\s]\s*", "", ln).strip()
            for ln in steps_field.splitlines()
            if ln.strip()
        ]

    suggestion = {
        "title":       str(title).strip()[:80] or _fallback_from_filename(filename)["title"],
        "category":    _coerce_category(raw.get("category")),
        "description": str(description).strip()[:600],
        "steps":       _coerce_steps(steps_field),
        "risk_level":  _coerce_risk(raw.get("risk_level") or raw.get("risk")),
        "tags":        _coerce_tags(raw.get("tags") or []),
        "llm_used":    True,
        "model":       result.get("model"),
        "latency_ms":  result.get("latency_ms", 0),
        "relevance_score": _coerce_relevance_score(raw.get("relevance_score")),
    }
    return suggestion
