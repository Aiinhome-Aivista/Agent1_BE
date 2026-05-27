"""
Back-compat shim.

The application previously imported `mistral_service` and called
`mistral_service.analyze_failure(...)` directly. We've replaced the
real implementation with the unified, mode-switchable `llm_service`,
but every existing caller keeps working because this module re-exports
the same surface area.

If you have spare time, migrate callers to `from app.services.llm_service
import llm_service` — but it's not required.
"""
from __future__ import annotations

from app.services.llm_service import llm_service as _impl


class _MistralServiceProxy:
    """Forward attribute access to the real LLMService singleton."""

    def __getattr__(self, name: str):
        return getattr(_impl, name)

    # Explicit pass-throughs for the most-used method, so IDEs autocomplete:
    def analyze_failure(self, *args, **kwargs):
        return _impl.analyze_failure(*args, **kwargs)

    @property
    def model(self) -> str:
        return _impl._backend().model

    @property
    def mode(self) -> str:
        return _impl.mode


mistral_service = _MistralServiceProxy()
