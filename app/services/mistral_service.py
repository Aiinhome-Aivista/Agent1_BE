"""
Mistral LLM service.

Updated flow:
    logs ─► RAG (vector DB) ─► similar incidents + runbook snippets
                          ─► assembled context_block
                          ─► Mistral

The caller (incident_service / pipelines API) is responsible for retrieval –
this service just takes a `context_block` string and stitches it into the
prompt. That keeps Mistral decoupled from Chroma.

Also reports latency / success/failure into MetricsService.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any

from mistralai import Mistral

from app.core.config import settings
from app.services.metrics_service import metrics_service

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are an expert SRE / data engineer who diagnoses
CI/CD and data-pipeline failures.

You will be given:
  1. Metadata + the failing run's logs
  2. (Optional) RAG context: similar past incidents and excerpts from
     uploaded operational runbooks

If the RAG context contains a clearly applicable fix, prefer it: cite the
runbook or past incident by name when you do.

Respond with a SINGLE JSON object (no markdown fences) matching:
{
  "summary":       string,    // one-line headline
  "root_cause":    string,    // 1-3 sentences
  "suggested_fix": string,    // numbered remediation steps, plain text
  "fix_patch":     string,    // optional unified diff / code snippet, may be empty
  "confidence":    number,    // 0.0 to 1.0
  "used_context":  boolean    // true if you leaned on the RAG context above
}
"""


class MistralService:
    def __init__(self) -> None:
        self.api_key = (settings.MISTRAL_API_KEY or "").strip()
        self.model   = settings.MISTRAL_MODEL
        self._client: Mistral | None = None

        logger.info("Mistral service init – model=%s, key_present=%s",
                    self.model, bool(self.api_key))

    def _client_or_create(self) -> Mistral:
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is missing in .env")
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)
        return self._client

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def analyze_failure(
        self,
        pipeline_name: str,
        connector_type: str,
        error_message: str | None,
        logs: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        context_block: str | None = None,        # ← NEW: RAG snippets
    ) -> dict[str, Any]:
        """
        Diagnose-and-fix. The optional `context_block` is rendered above
        the user prompt; if absent the prompt looks the same as the old one,
        so older callers don't break.
        """
        log_text = "\n".join(
            f"[{l.get('timestamp')}] [{l.get('level')}] "
            f"{(l.get('source') or '').strip()} | {l.get('message','')[:2000]}"
            for l in logs[-80:]
        )

        user_msg_parts: list[str] = []
        if context_block and context_block.strip():
            user_msg_parts.append(
                "Below is retrieved context from prior incidents and runbooks.\n"
                "Use it if relevant; ignore it if not.\n\n"
                f"{context_block.strip()}\n"
            )

        user_msg_parts.append(
            f"--- CURRENT FAILURE ---\n"
            f"Pipeline: {pipeline_name}\n"
            f"Source: {connector_type}\n"
            f"Top-level error: {error_message or '(none reported)'}\n"
            f"Metadata: {json.dumps(metadata or {}, default=str)[:1500]}\n\n"
            f"--- LOGS ---\n{log_text}\n--- END LOGS ---"
        )
        user_msg = "\n".join(user_msg_parts)

        t0 = time.perf_counter()
        try:
            client = self._client_or_create()

            resp = client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            content = resp.choices[0].message.content if resp.choices else "{}"
            if isinstance(content, list):
                content = "".join(getattr(c, "text", "") for c in content)

            parsed = self._safe_parse(str(content))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics_service.record_llm_call(
                model=self.model,
                latency_ms=elapsed_ms,
                success=True,
                prompt_chars=len(user_msg),
                response_chars=len(str(content)),
            )

            confidence = self._clamp_confidence(parsed.get("confidence", 0.0))

            return {
                "summary":       (parsed.get("summary", "") or "")[:1000],
                "root_cause":    (parsed.get("root_cause", "") or "")[:4000],
                "suggested_fix": (parsed.get("suggested_fix", "") or "")[:8000],
                "fix_patch":     (parsed.get("fix_patch", "") or "")[:20000],
                "confidence":    confidence,
                "used_context":  bool(parsed.get("used_context", False)),
                "model":         self.model,
                "latency_ms":    round(elapsed_ms, 2),
                "raw_response":  parsed,
            }

        except Exception as exc:
            traceback.print_exc()
            logger.exception("Mistral call failed")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics_service.record_llm_call(
                model=self.model,
                latency_ms=elapsed_ms,
                success=False,
                prompt_chars=len(user_msg),
            )
            return {
                "summary":       "LLM analysis unavailable",
                "root_cause":    f"Mistral request failed: {exc}",
                "suggested_fix": "Verify MISTRAL_API_KEY and model permissions, then retry.",
                "fix_patch":     "",
                "confidence":    0.0,
                "used_context":  False,
                "model":         self.model,
                "latency_ms":    round(elapsed_ms, 2),
                "raw_response":  {"error": str(exc)},
            }

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_parse(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        # Strip the occasional ```json fences Mistral throws in
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception as e:
            logger.warning("Mistral JSON parse failed: %s", e)
            return {
                "summary":       "LLM response parsing failed",
                "root_cause":    cleaned[:4000],
                "suggested_fix": "",
                "fix_patch":     "",
                "confidence":    0.0,
            }

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(v, 1.0))


mistral_service = MistralService()
