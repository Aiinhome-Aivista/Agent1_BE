"""
Code-fix generation service.

When the agent classifies an incident as a KNOWN error that is trusted enough
to auto-remediate, it needs an actual code change to put in a PR. Two paths:

  1. The pattern already carries an accepted code fix (e.g. ingested from a
     previously-merged human PR). We reuse it verbatim — this is the whole
     point of the learning loop.

  2. Otherwise we ask the LLM to produce a concrete change, constrained to a
     strict JSON contract so we can commit it without guessing.

The LLM is asked for a *full-file replacement* (file_path + new_content) when
it can, because that commits cleanly via the GitHub contents API. It may also
return a unified diff for human review.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.llm_service import llm_service
from app.models.solution_models import SolutionPattern

logger = logging.getLogger(__name__)


CODEGEN_SYSTEM_PROMPT = """You are a senior data engineer who writes minimal,
surgical code fixes for data-pipeline failures. You are given a known error,
its root cause, and the remediation steps. Produce the smallest change that
fixes the root cause.

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "file_path":    string,   // repo-relative path to change, e.g. "pipeline/transform.py"
  "new_content":  string,   // the COMPLETE new contents of that file
  "diff":         string,   // optional unified diff for human review
  "language":     string,   // e.g. "python", "yaml"
  "explanation":  string,   // 1-3 sentences on what changed and why
  "confidence":   number    // 0..1, your confidence the change compiles & fixes it
}
If you cannot safely produce a full file, set new_content to "" and put a
unified diff in "diff" instead."""


@dataclass
class GeneratedFix:
    file_path: str | None
    new_content: str | None
    diff: str | None
    language: str | None
    explanation: str
    confidence: float
    source: str  # "reused-known-fix" | "llm" | "none"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_committable(self) -> bool:
        return bool(self.file_path and self.new_content)


def _from_existing(pattern: SolutionPattern) -> GeneratedFix | None:
    for f in pattern.fixes or []:
        if f.has_code:
            return GeneratedFix(
                file_path=f.file_path,
                new_content=f.new_content,
                diff=f.diff,
                language=f.language,
                explanation=f.explanation or "Reused previously-accepted fix.",
                confidence=pattern.confidence or 0.8,
                source="reused-known-fix",
            )
    return None


def generate_fix(
    *,
    pattern: SolutionPattern | None,
    error_text: str,
    root_cause: str,
    fix_steps: list[str],
    category: str,
    repo_hint: str | None = None,
    existing_file_content: str | None = None,
) -> GeneratedFix:
    # 1. Reuse an accepted fix when we have one — the learning-loop payoff.
    if pattern is not None:
        reused = _from_existing(pattern)
        if reused is not None:
            logger.info("code_fix: reusing accepted fix for pattern #%s", pattern.id)
            return reused

    # 2. Ask the LLM for a concrete change.
    user_parts = [
        f"Category: {category}",
        f"Root cause: {root_cause or '(unknown)'}",
        "Remediation steps:",
        *[f"- {s}" for s in (fix_steps or [])],
        "",
        "Error / logs:",
        (error_text or "")[:3000],
    ]
    if repo_hint:
        user_parts.append(f"\nTarget repository: {repo_hint}")
    if existing_file_content:
        user_parts.append(
            "\nCurrent contents of the most likely file to change:\n"
            + existing_file_content[:6000]
        )

    messages = [
        {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    try:
        parsed = llm_service.chat_json(messages, temperature=0.1, max_tokens=2500)
    except Exception as e:
        logger.exception("code_fix LLM call failed: %s", e)
        parsed = {}

    if not parsed:
        return GeneratedFix(
            file_path=None, new_content=None, diff=None, language=None,
            explanation="LLM did not return a usable fix.",
            confidence=0.0, source="none", raw={},
        )

    return GeneratedFix(
        file_path=(parsed.get("file_path") or None),
        new_content=(parsed.get("new_content") or None),
        diff=(parsed.get("diff") or None),
        language=(parsed.get("language") or None),
        explanation=str(parsed.get("explanation") or "")[:600],
        confidence=float(parsed.get("confidence") or 0.0),
        source="llm",
        raw=parsed,
    )
