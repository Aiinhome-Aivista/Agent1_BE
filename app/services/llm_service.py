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
    if not s:
        return None
    s = _strip_json_fences(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Last-ditch: try to slice the outermost { ... }
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                return None
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


# ──────────────────────────────────────────────────────────────────────
# Built-in system prompt
# ──────────────────────────────────────────────────────────────────────

DIAGNOSIS_SYSTEM_PROMPT = """You are an expert SRE / data engineer who diagnoses
CI/CD and data-pipeline failures with precision. You must explain your findings in a deeply human-oriented, conversational, and highly elaborate manner so that even non-experts can understand what went wrong and how to fix it.

You will be given:
  1. Metadata + the failing run's logs
  2. (Optional) RAG context: similar past incidents and excerpts from
     uploaded operational runbooks
  3. (Optional) GRAPH context: structured fix recommendations ranked by
     historical success rate
  4. (Optional) KB context: a known error pattern with its accepted fix and
     how many times humans have accepted it

If the GRAPH / RAG / KB context contains a clearly applicable fix, PREFER it and
cite the runbook, past incident, or known pattern by name. When a known
accepted fix exists, reuse its wording so the operator sees continuity.

Be extremely specific, tracing the exact origin of the error from the logs. Eliminate conversational fluff, but provide a DEEPLY detailed, beginner-friendly explanation. Your root cause MUST clearly explain exactly where the error came from, what it means in simple terms, and its impact. Your suggested fix MUST be an extremely detailed, foolproof, step-by-step tutorial that a complete beginner can easily follow to fix the issue.

CRITICAL ADVANCED INSTRUCTIONS:
1. DO NOT GUESS. Your root cause and exact fix MUST be derived solely and explicitly from the provided logs and data pipeline context.
2. Be as advanced as Databricks Assistant: Extract the EXACT filename, line number, SQL query, or function name that triggered the failure.
3. For the suggested fix, provide explicit, copy-pasteable code patches, exact SQL queries, or exact CLI commands based strictly on the pipeline's exact data.

Respond with a SINGLE JSON object (no markdown fences) matching:
{
  "summary":          string,           // one-line headline
  "root_cause":       string,           // Deep, beginner-friendly explanation of exactly where the error originated, why it happened, and what it means
  "root_cause_details": [string, ...],  // 2-5 bullet points pinpointing the exact log lines, code lines, or components
  "suggested_fix":    string,           // Highly detailed, foolproof, step-by-step tutorial including exact code changes or SQL queries to fix the issue
  "validation_steps": [string, ...],    // 1-4 checks to confirm the fix worked
  "fix_patch":        string,           // optional unified diff / code snippet, may be empty
  "confidence":       number,           // 0.0 to 1.0
  "confidence_rationale": [string, ...],// 2-4 short reasons WHY this confidence
  "used_context":     boolean           // true if you leaned on the RAG/GRAPH/KB context
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

        # Native Ollama shape first
        native_payload = {
            "model": self.model,
            "messages": messages,
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
                # Some versions wrap inside "messages" or "response"
                if "response" in data:
                    return data["response"] or ""
        except Exception as e:
            logger.debug("Local /api/chat failed, will try OpenAI shape: %s", e)

        # OpenAI-compatible fallback
        openai_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
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
        """Diagnose a pipeline failure. Returns:
        {
            "summary": str, "root_cause": str, "suggested_fix": str,
            "fix_patch": str, "confidence": float, "used_context": bool,
            "raw_response": dict,        # the parsed JSON (or {} on parse failure)
            "raw_text": str,             # the model's literal text
            "mode": "Cloud" | "Local",
            "model": str,
            "latency_ms": int,
        }
        """
        logs = list(logs or [])
        # Compose the user prompt
        log_block = "\n".join(
            f"[{l.get('timestamp','')}] {l.get('level','')} {l.get('source','')}: {l.get('message','')}"
            for l in logs[-30:]  # cap
        )
        meta_block = json.dumps(metadata or {}, default=str)[:1000]

        user_parts = [
            f"Pipeline: {pipeline_name}",
            f"Connector: {connector_type}",
            f"Error: {error_message or '(none)'}",
            f"Metadata: {meta_block}",
            "Logs (most recent last):",
            log_block,
        ]
        if context_block:
            user_parts.append("\n=== Retrieved context ===\n" + context_block)
        user_msg = "\n".join(user_parts)

        messages = [
            {"role": "system", "content": system_prompt or DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        t0 = time.perf_counter()
        try:
            raw_text = self.chat(messages)
        except Exception as e:
            logger.exception("LLM analyze_failure failed: %s", e)
            return {
                "summary": "LLM call failed",
                "root_cause": str(e),
                "suggested_fix": "",
                "fix_patch": "",
                "confidence": 0.0,
                "used_context": False,
                "raw_response": {},
                "raw_text": "",
                "mode": self._mode,
                "model": self._backend().model,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": str(e),
            }

        parsed = _safe_json_loads(raw_text) or {}
        def _as_list(v: Any) -> list[str]:
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, str) and v.strip():
                return [v.strip()]
            return []
        return {
            "summary":       str(parsed.get("summary") or "")[:500],
            "root_cause":    "\n\n".join(_as_list(parsed.get("root_cause"))),
            "root_cause_details": _as_list(parsed.get("root_cause_details")),
            "suggested_fix": "\n\n".join(_as_list(parsed.get("suggested_fix"))),
            "validation_steps": _as_list(parsed.get("validation_steps")),
            "fix_patch":     str(parsed.get("fix_patch") or ""),
            "confidence":    float(parsed.get("confidence") or 0.0),
            "confidence_rationale": _as_list(parsed.get("confidence_rationale")),
            "used_context":  bool(parsed.get("used_context")),
            "raw_response":  parsed,
            "raw_text":      raw_text,
            "mode":          self._mode,
            "model":         self._backend().model,
            "latency_ms":    int((time.perf_counter() - t0) * 1000),
        }

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