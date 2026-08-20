"""
Unified LLM service with runtime Cloud ↔ Local ↔ Gemini switching.

Three backends are supported behind one interface:

  - "Cloud" : the official `mistralai` SDK against api.mistral.ai
              (controlled by MISTRAL_API_KEY + MODEL_NAME)
  - "Local" : an Ollama-compatible HTTP endpoint
              (controlled by MISTRAL_LOCAL_URL + MISTRAL_LOCAL_MODEL)
  - "Gemini": the official `google-genai` SDK against Google Gemini
              (controlled by GEMINI_API_KEY + GEMINI_MODEL)

Both honour the same temperature / max-tokens knobs (LLM_TEMPERATURE,
LLM_MAX_TOKENS) and expose the same `chat_json(...)` and
`analyze_failure(...)` methods, so every existing caller
(mistral_service.analyze_failure, llm_runbook_service.suggest_metadata,
etc.) keeps working untouched.

The active mode is held in-process and can be flipped at runtime via
LLMService.set_mode("Cloud" | "Local") — used by the /api/v1/llm/mode
endpoint and the header pill in the UI. The change is also written back
to the .env file so it survives restarts (best-effort).

Latency + success/failure are reported into metrics_service if it's
importable, so the existing Metrics page keeps lighting up regardless
of which backend is active.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import httpx

from app.core.config import settings
from app.services.diagnosis_normalizer import (
    extract_verified_facts,
    normalize_diagnosis,
    normalize_known_fix,
    build_suggested_fix_text,
)

logger = logging.getLogger(__name__)

# Optional metrics hook — don't hard-fail if the module isn't present.
try:
    from app.services.metrics_service import metrics_service  # type: ignore
except Exception:  # pragma: no cover
    metrics_service = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_json_fences(s: str) -> str:
    """Mistral (esp. local Ollama) often wraps JSON in ```json ... ``` fences."""
    return _JSON_FENCE_RE.sub("", s or "").strip()


def _safe_json_loads(s: str) -> dict[str, Any] | None:
    if not s or not s.strip():
        return None
    cleaned = s.strip()

    # 1. Direct parse attempt
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Strip Markdown code fences (```json ... ``` or ``` ... ```)
    fenceless = _strip_json_fences(cleaned)
    try:
        data = json.loads(fenceless)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 3. Find outermost curly braces { ... }
    i = fenceless.find("{")
    j = fenceless.rfind("}")
    if i >= 0 and j > i:
        candidate = fenceless[i : j + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            # 4. Repair trailing commas before closing braces/brackets
            repaired = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    return None


def _normalize_url(url: str) -> str:
    """Trim trailing slashes; tolerate copy-paste with embedded brackets/whitespace."""
    if not url:
        return ""
    url = url.strip().strip("[]").rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    for suffix in ("/api/generate", "/api/chat", "/v1/chat/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def _normalize_string_or_list(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        items = []
        for i, item in enumerate(val, start=1):
            if isinstance(item, dict):
                step = item.get("step") or i
                details = str(item.get("details") or item.get("detail") or "").strip()
                desc = str(item.get("description") or item.get("step_description") or item.get("text") or "").strip()
                title = str(item.get("title") or item.get("action") or item.get("name") or "").strip()
                outcome = str(item.get("expected_outcome") or item.get("outcome") or "").strip()

                # If description is a concise title and details has the body
                if not title and desc and details:
                    title = desc
                    desc = details
                elif not desc and details:
                    desc = details

                step_lines = []
                if title and desc and title != desc:
                    step_lines.append(f"{step}. {title}:\n{desc}")
                elif desc:
                    step_lines.append(f"{step}. {desc}")
                elif title:
                    step_lines.append(f"{step}. {title}")
                else:
                    step_lines.append(f"{step}. {item}")

                if outcome:
                    step_lines.append(f"Expected outcome: {outcome}")

                items.append("\n".join(step_lines))
            elif item and str(item).strip():
                items.append(str(item).strip())
        return "\n\n".join(items)
    if isinstance(val, dict):
        details = str(val.get("details") or val.get("detail") or "").strip()
        desc = str(val.get("description") or val.get("text") or "").strip()
        title = str(val.get("title") or val.get("action") or "").strip()
        outcome = str(val.get("expected_outcome") or val.get("outcome") or "").strip()

        if not title and desc and details:
            title = desc
            desc = details
        elif not desc and details:
            desc = details

        parts = []
        if title and desc and title != desc:
            parts.append(f"{title}:\n{desc}")
        elif desc:
            parts.append(desc)
        elif title:
            parts.append(title)
        if outcome:
            parts.append(f"Expected outcome: {outcome}")
        return "\n".join(parts) if parts else str(val).strip()
    return str(val).strip()


def _normalize_action_list(val: Any) -> list[str]:
    if not val:
        return []
    if isinstance(val, str):
        lines = [line.strip() for line in val.split("\n") if line.strip()]
        return lines if lines else [val.strip()]
    if isinstance(val, list):
        res = []
        for item in val:
            if isinstance(item, dict):
                action = (
                    item.get("action")
                    or item.get("description")
                    or item.get("title")
                    or item.get("step_description")
                    or item.get("text")
                )
                if action:
                    clean_action = str(action).strip()
                    clean_action = re.sub(r"^\d+\.\s*", "", clean_action)
                    res.append(clean_action)
            elif isinstance(item, str) and item.strip():
                clean_str = item.strip()
                if clean_str.startswith("{") and ("'action':" in clean_str or '"action":' in clean_str):
                    try:
                        import ast
                        d = ast.literal_eval(clean_str)
                        if isinstance(d, dict) and ("action" in d or "description" in d):
                            act = d.get("action") or d.get("description")
                            if act:
                                res.append(str(act).strip())
                                continue
                    except Exception:
                        pass
                res.append(clean_str)
        return res
    return []


# ──────────────────────────────────────────────────────────────────────
# Built-in system prompt
# ──────────────────────────────────────────────────────────────────────

DIAGNOSIS_SYSTEM_PROMPT = """You are an SRE and Data Engineering incident diagnosis synthesizer.

You receive VERIFIED_FACTS and HISTORICAL_KB_CONTEXT.

VERIFIED_FACTS are authoritative, verified, and immutable.
Never modify, reinterpret, recalculate, contradict, or invent them.

Use historical KB context only as supporting evidence. Do not assume historical remediation is required for the current incident unless proven by current telemetry.

Produce a concise, pinpointed diagnosis and remediation plan.

Separate:
1. required recovery actions (actions strictly required to recover THIS failed pipeline run)
2. optional runbook improvements (audit improvements, quarantine Delta tables, long-term enhancements)
3. long-term prevention (pre-ingestion contract checks, early warning alerts)

CRITICAL RULES:
1. Do NOT generate confidence or claim a fix was accepted.
2. Do NOT invent team names ("Data Engineering Team"), Slack channels, URLs, Jira tickets, or notebook names. Use neutral ownership ("Identify the owner of the upstream source responsible for the failed batch").
3. Do NOT invent RuntimeError unless present in verified logs.
4. Quarantine does NOT alter the failed batch's invalid percentage or bypass the threshold. The source batch must be corrected or replaced.
5. Every required action must have a measurable expected outcome and validation condition.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED JSON OUTPUT SCHEMA (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a single valid JSON object. No prose or markdown outside the JSON.

{
  "summary": "Pipeline <name> failed during <stage> because <metric> exceeded configured <threshold> threshold.",
  "root_cause": "The incoming batch contained <invalid_count> unique invalid records out of <total_count> records, producing a <rate>% rate exceeding configured <threshold>%.",
  "failure_mechanism": "The <stage> stage detected an invalid-record rate of <rate>%, exceeding <threshold>% and triggering <error_code>, terminating processing before downstream layers.",
  "impact": "Pipeline execution was terminated during <stage>, preventing unvalidated records from reaching downstream layers.",
  "immediate_fix": [
    {
      "title": "Correct or Replace Failed Source Batch",
      "action": "Identify the owner of the upstream source responsible for the failed batch and coordinate correction of the invalid records.",
      "expected_outcome": "The corrected source batch achieves an invalid-record rate at or below the configured threshold.",
      "validation": "Re-run validation checks and confirm invalid percentage <= threshold."
    }
  ],
  "optional_improvements": [
    {
      "title": "Evaluate Quarantine Handling for Auditing",
      "action": "Consider routing invalid records to a quarantine Delta table for investigation without bypassing the threshold.",
      "expected_outcome": "Invalid records are preserved for auditing while pipeline data quality enforcement remains intact."
    }
  ],
  "long_term_prevention": [
    "Implement pre-ingestion schema validation and contract checks at the upstream source boundary.",
    "Configure early-warning alert thresholds before reaching the hard pipeline failure limit."
  ]
}
"""


# ──────────────────────────────────────────────────────────────────────
# Backend implementations
# ──────────────────────────────────────────────────────────────────────


class _CloudBackend:
    """Mistral Cloud via the official SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = (api_key or "").strip()
        self.model = (model or "mistral-small-latest").strip()
        self._client = None  # lazy

    def _client_or_create(self):
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is missing — cannot use Cloud mode")
        if self._client is None:
            try:
                from mistralai import Mistral
            except ImportError as e:
                raise RuntimeError(
                    "mistralai SDK not installed. Run: pip install mistralai"
                ) from e
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict[str, str]], *, temperature: float,
             max_tokens: int) -> str:
        client = self._client_or_create()
        resp = client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # mistralai >=1.x returns ChatCompletionResponse with .choices[0].message.content
        try:
            return resp.choices[0].message.content or ""
        except Exception:  # pragma: no cover
            return str(resp)

    def health(self) -> dict[str, Any]:
        try:
            txt = self.chat(
                [{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=4,
            )
            return {"ok": True, "model": self.model, "sample": (txt or "")[:32]}
        except Exception as e:
            return {"ok": False, "model": self.model, "error": str(e)}


class _LocalBackend:
    """
    Local LLM via an Ollama-compatible HTTP endpoint.

    Ollama exposes two compatible surfaces:
      1. Native:   POST {base}/api/chat   {"model": ..., "messages": [...], "stream": false}
      2. OpenAI:   POST {base}/v1/chat/completions

    We try Native first (richer), fall back to OpenAI shape so this also
    works against vLLM / LM Studio / any OpenAI-compatible local server.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 600.0) -> None:
        self.base_url = _normalize_url(base_url)
        self.model = (model or "mistral:latest").strip()
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}{path}"
        return httpx.post(url, json=payload, timeout=self.timeout)

    def chat(self, messages: list[dict[str, str]], *, temperature: float,
             max_tokens: int) -> str:
        if not self.base_url:
            raise RuntimeError("MISTRAL_LOCAL_URL is missing — cannot use Local mode")

        # 1. Native Ollama chat shape with format="json"
        native_payload = {
            "model": self.model,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            r = self._post("/api/chat", native_payload)
            if r.status_code == 200:
                data = r.json()
                # Ollama: {"message": {"role": "assistant", "content": "..."}}
                msg = data.get("message") or {}
                content = msg.get("content")
                if content is not None:
                    return content
                if "response" in data:
                    return data["response"] or ""
        except Exception as e:
            logger.debug("Local /api/chat failed, will try /api/generate or OpenAI shape: %s", e)

        # 2. Ollama /api/generate shape with format="json"
        try:
            prompt_text = "\n\n".join(
                f"{m.get('role', 'user').upper()}:\n{m.get('content', '')}"
                for m in messages
            )
            gen_payload = {
                "model": self.model,
                "prompt": prompt_text,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            r = self._post("/api/generate", gen_payload)
            if r.status_code == 200:
                data = r.json()
                if "response" in data and data["response"]:
                    return data["response"]
        except Exception as e:
            logger.debug("Local /api/generate failed, will try OpenAI shape: %s", e)

        # 3. OpenAI-compatible fallback with response_format={"type": "json_object"}
        openai_payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            r = self._post("/v1/chat/completions", openai_payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            # Fallback without response_format if not supported by local inference server
            openai_payload.pop("response_format", None)
            r = self._post("/v1/chat/completions", openai_payload)
            r.raise_for_status()
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"Unexpected local LLM response: {data}") from e

    def health(self) -> dict[str, Any]:
        if not self.base_url:
            return {"ok": False, "model": self.model, "error": "MISTRAL_LOCAL_URL is empty"}
        # Try /api/tags (Ollama) → /v1/models (OpenAI) → /
        for path in ("/api/tags", "/v1/models", "/"):
            try:
                r = httpx.get(f"{self.base_url}{path}", timeout=5.0)
                if r.status_code < 500:
                    return {
                        "ok": True,
                        "model": self.model,
                        "endpoint": f"{self.base_url}{path}",
                        "status": r.status_code,
                    }
            except Exception:
                continue
        return {"ok": False, "model": self.model, "error": f"unreachable at {self.base_url}"}


class _GeminiBackend:
    """Google Gemini via the official SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = (api_key or "").strip()
        self.model = (model or "gemini-3.6-flash").strip()
        self._client = None  # lazy

    def _client_or_create(self):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is missing — cannot use Gemini mode")
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise RuntimeError(
                    "google-genai SDK not installed. Run: pip install google-genai"
                ) from e
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict[str, str]], *, temperature: float,
             max_tokens: int) -> str:
        client = self._client_or_create()
        # Translate role names: 'user' -> 'user', 'assistant' -> 'model', 'system' -> 'user' or system config
        # Google genai prefers single system instruction in config, or we just keep it simple.
        # But for backward compatibility with the generic messages list, let's format it to a simple prompt string for now
        # OR format it properly as Contents:
        # A simple approach: concat history into text or use the SDK's structured contents.
        contents = []
        system_instruction = None
        for m in messages:
            role = m["role"]
            if role == "system":
                system_instruction = m["content"]
            else:
                r = "model" if role == "assistant" else "user"
                contents.append({"role": r, "parts": [{"text": m["content"]}]})
        
        # pyrefly: ignore [missing-import]
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return resp.text or ""

    def health(self) -> dict[str, Any]:
        try:
            txt = self.chat(
                [{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=4,
            )
            return {"ok": True, "model": self.model, "sample": (txt or "")[:32]}
        except Exception as e:
            return {"ok": False, "model": self.model, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────
# Public service
# ──────────────────────────────────────────────────────────────────────


class LLMService:
    """One service, three backends, hot-swappable."""

    VALID_MODES = ("Cloud", "Local", "Gemini")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mode: str = self._normalize_mode(
            getattr(settings, "MISTRAL_MODE", "Cloud")
        )
        self._cloud = _CloudBackend(
            api_key=getattr(settings, "MISTRAL_API_KEY", "") or "",
            model=getattr(settings, "MODEL_NAME", None)
            or getattr(settings, "MISTRAL_MODEL", "mistral-small-latest"),
        )
        self._local = _LocalBackend(
            base_url=getattr(settings, "MISTRAL_LOCAL_URL", "") or "",
            model=getattr(settings, "MISTRAL_LOCAL_MODEL", "mistral:latest"),
        )
        self._gemini = _GeminiBackend(
            api_key=getattr(settings, "GEMINI_API_KEY", "") or "",
            model=getattr(settings, "GEMINI_MODEL", "gemini-3.6-flash"),
        )
        logger.info(
            "LLMService init – mode=%s, cloud_model=%s, local_url=%s, local_model=%s, gemini_model=%s",
            self._mode, self._cloud.model, self._local.base_url, self._local.model, self._gemini.model
        )

    # ── mode management ──────────────────────────────────────────────

    @staticmethod
    def _normalize_mode(m: str | None) -> str:
        if not m:
            return "Cloud"
        m = m.strip().capitalize()
        return m if m in LLMService.VALID_MODES else "Cloud"

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str, *, persist: bool = True) -> str:
        mode = self._normalize_mode(mode)
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        with self._lock:
            old = self._mode
            self._mode = mode
            logger.info("LLM mode switched %s → %s", old, mode)
            if persist:
                try:
                    self._persist_mode_to_env(mode)
                except Exception as e:
                    logger.warning("Could not persist MISTRAL_MODE to .env: %s", e)
        return self._mode

    @staticmethod
    def _persist_mode_to_env(mode: str) -> None:
        """Best-effort: rewrite MISTRAL_MODE=... in the project .env."""
        # Look for .env next to the working directory or one level up
        candidates = [Path.cwd() / ".env", Path.cwd().parent / ".env"]
        env_path = next((p for p in candidates if p.exists()), None)
        if env_path is None:
            return
        lines = env_path.read_text(encoding="utf-8").splitlines()
        found = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("MISTRAL_MODE="):
                lines[i] = f"MISTRAL_MODE={mode}"
                found = True
                break
        if not found:
            lines.append(f"MISTRAL_MODE={mode}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── core chat ────────────────────────────────────────────────────

    def _backend(self):
        if self._mode == "Gemini":
            return self._gemini
        return self._cloud if self._mode == "Cloud" else self._local

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Raw chat call. Returns the assistant's text content."""
        t = float(temperature if temperature is not None
                  else getattr(settings, "LLM_TEMPERATURE", 0.2))
        n = int(max_tokens if max_tokens is not None
                else getattr(settings, "LLM_MAX_TOKENS", 2000))
        backend = self._backend()
        start = time.perf_counter()
        try:
            out = backend.chat(messages, temperature=t, max_tokens=n)
            self._record_latency(start, ok=True)
            return out
        except Exception:
            self._record_latency(start, ok=False)
            raise

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Same as chat() but tries hard to return a parsed JSON dict.
        On failure returns {} so callers can fall back gracefully."""
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return _safe_json_loads(raw) or {}

    # ── high-level: diagnose-and-fix (back-compat with mistral_service) ──

    def analyze_failure(
        self,
        pipeline_name: str,
        connector_type: str,
        error_message: str | None,
        logs: Iterable[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        context_block: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Diagnose a pipeline failure using fact-locked normalization and dual-engine synthesis."""
        logs_list = list(logs or [])
        meta = dict(metadata or {})

        # ── 1. PARSER OWNS FACTS: Deterministic fact extraction ───────────────
        verified_facts = extract_verified_facts(
            pipe_name=pipeline_name,
            connector_type=connector_type,
            error_message=error_message,
            logs=logs_list,
            metadata=meta,
        )

        canonical_pipeline = verified_facts["pipeline_name"]
        canonical_stage = verified_facts["failed_stage"]
        error_code = verified_facts["error_code"]
        total_records = verified_facts["total_records"]
        invalid_records = verified_facts["invalid_records"]
        invalid_pct = verified_facts["invalid_percentage"]
        allowed_threshold = verified_facts["allowed_threshold"]

        # Compose prompt
        log_block = "\n".join(
            f"[{l.get('timestamp','')}] {l.get('level','')} {l.get('source','')}: {l.get('message','')}"
            for l in logs_list[-30:]
        )
        meta_block = json.dumps(meta, default=str)[:1000]

        facts_lines = []
        if canonical_pipeline:
            facts_lines.append(f"Pipeline Name: {canonical_pipeline}")
        if canonical_stage:
            facts_lines.append(f"Failed Stage: {canonical_stage}")
        if error_code:
            facts_lines.append(f"Error Code: {error_code}")
        if total_records is not None and invalid_records is not None:
            facts_lines.append(f"Total Records: {total_records}")
            facts_lines.append(f"Invalid Records: {invalid_records}")
        if invalid_pct is not None:
            facts_lines.append(f"Invalid Percentage: {float(invalid_pct):.2f}%")
        if allowed_threshold is not None:
            facts_lines.append(f"Allowed Threshold: {float(allowed_threshold):.1f}%")

        # Format user prompt with clear separation: VERIFIED_FACTS, HISTORICAL_KB_CONTEXT, TASK
        user_parts = [
            "### VERIFIED_FACTS (AUTHORITATIVE & IMMUTABLE)",
            f"- Pipeline Name: {canonical_pipeline}",
            f"- Failed Stage: {canonical_stage}",
            f"- Connector Type: {connector_type}",
            f"- Primary Error: {error_message or '(none)'}",
        ]
        if error_code:
            user_parts.append(f"- Error Code: {error_code}")
        if total_records is not None and invalid_records is not None:
            user_parts.append(f"- Total Records: {total_records}")
            user_parts.append(f"- Unique Invalid Records: {invalid_records}")
        if invalid_pct is not None:
            user_parts.append(f"- Measured Invalid Rate: {float(invalid_pct):.2f}%")
        if allowed_threshold is not None:
            user_parts.append(f"- Allowed Maximum Threshold: {float(allowed_threshold):.1f}%")
        val_failures = verified_facts.get("validation_failures")
        if val_failures:
            fails_str = ", ".join(f"{k}: {v}" for k, v in val_failures.items())
            user_parts.append(f"- Validation Category Violations: {fails_str}")
        user_parts.extend([
            f"- Metadata: {meta_block}",
            "- Recent Execution Logs:",
            log_block,
        ])

        if context_block:
            user_parts.extend([
                "",
                "### HISTORICAL_KB_CONTEXT (REFERENCE ONLY)",
                context_block,
            ])

        user_parts.extend([
            "",
            "### TASK",
            "Synthesize the root cause, failure mechanism, impact, and remediation plan into the requested JSON schema.",
            "Do NOT recalculate or contradict VERIFIED_FACTS. Separate required recovery steps from optional improvements.",
        ])

        if self._mode == "Local":
            user_parts.append(
                "\nSTRICT OUTPUT REQUIREMENT:\n"
                "Return ONLY a single valid JSON object containing reasoning fields. "
                "Do not include any prose or markdown fences outside the JSON. "
                "Output must start with '{' and end with '}'."
            )

        user_msg = "\n".join(user_parts)
        messages = [
            {"role": "system", "content": system_prompt or DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        t0 = time.perf_counter()
        raw_text = ""
        try:
            raw_text = self.chat(messages)
        except Exception as e:
            logger.exception("LLM analyze_failure failed: %s", e)
            unavail_output = {
                "summary": "AI diagnosis temporarily unavailable",
                "root_cause": "",
                "failure_mechanism": "",
                "impact": "",
                "suggested_fix": "",
                "fix_patch": "",
                "confidence": 0.0,
                "used_context": False,
                "diagnosis_status": "failed",
                "diagnosis_error": str(e),
                "error_details": f"AI service error: {e}",
            }
            normalized = normalize_diagnosis(verified_facts, unavail_output)
            normalized.update({
                "mode": self._mode,
                "model": self._backend().model,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "raw_text": "",
                "raw_response": normalized,
                "error": str(e),
            })
            return normalized

        parsed = _safe_json_loads(raw_text)
        if not parsed:
            logger.warning(
                "LLM response could not be parsed as structured JSON schema (len=%d). "
                "raw_text[:200]=%r",
                len(raw_text), raw_text[:200],
            )
            parse_fail_output = {
                "summary": "AI diagnosis response could not be structured",
                "root_cause": "Not determinable because the diagnosis model returned an invalid structured response.",
                "failure_mechanism": "Not determinable from the available logs and metadata.",
                "impact": "Not determinable from the available logs and metadata.",
                "suggested_fix": "Retry the diagnosis using the Re-analyze button, or switch to Gemini mode for higher-capacity structured output.",
                "fix_patch": "",
                "confidence": 0.0,
                "used_context": False,
                "diagnosis_status": "parse_failed",
                "diagnosis_error": "The model response did not conform to the required JSON schema.",
                "error_details": "Model formatting error: response was not valid JSON.",
            }
            normalized = normalize_diagnosis(verified_facts, parse_fail_output)
            normalized.update({
                "mode": self._mode,
                "model": self._backend().model,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "raw_text": raw_text,
                "raw_response": normalized,
                "error": "JSON parse failed",
            })
            return normalized

        # ── 2. BACKEND OWNS NORMALIZATION & FACT LOCKING ─────────────────────
        normalized = normalize_diagnosis(verified_facts, parsed)
        normalized.update({
            "mode": self._mode,
            "model": self._backend().model,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "raw_text": raw_text,
            "raw_response": dict(normalized),
        })
        return normalized

    # ── health ───────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        backend = self._backend()
        return {
            "mode": self._mode,
            "backend": backend.health(),
            "config": {
                "cloud_model": self._cloud.model,
                "local_url": self._local.base_url,
                "local_model": self._local.model,
                "temperature": float(getattr(settings, "LLM_TEMPERATURE", 0.2)),
                "max_tokens": int(getattr(settings, "LLM_MAX_TOKENS", 2000)),
            },
        }

    def health_both(self) -> dict[str, Any]:
        """Probe ALL backends — used by the UI mode-switch pill."""
        return {
            "active_mode": self._mode,
            "Cloud": self._cloud.health(),
            "Local": self._local.health(),
            "Gemini": self._gemini.health(),
        }

    # ── metrics ──────────────────────────────────────────────────────

    def _record_latency(self, t0: float, *, ok: bool) -> None:
        if metrics_service is None:
            return
        try:
            ms = (time.perf_counter() - t0) * 1000
            # Try a few likely method names so this stays compatible
            for attr in ("record_llm_latency", "record_llm", "observe_llm"):
                fn = getattr(metrics_service, attr, None)
                if callable(fn):
                    fn(ms, ok=ok, mode=self._mode)  # type: ignore[misc]
                    return
        except Exception:  # pragma: no cover
            pass


# Module-level singleton — import this everywhere.
llm_service = LLMService()