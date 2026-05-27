"""
LLM control endpoints.

Mount in app/main.py (or wherever you register routers):

    from app.api import llm as llm_router
    app.include_router(llm_router.router, prefix="/api/v1")

──────────────────────────────────────────────────────────────────────
Endpoints
──────────────────────────────────────────────────────────────────────
  GET  /api/v1/llm/mode              -> current active mode + config
  POST /api/v1/llm/mode  {"mode": "Local"|"Cloud"}
                                     -> switches and (best-effort) writes
                                        the change back to .env
  GET  /api/v1/llm/health            -> health of the active backend
  GET  /api/v1/llm/health/both       -> probes BOTH backends; used by
                                        the header pill to colour Cloud/Local
                                        green or red simultaneously
  POST /api/v1/llm/chat   {"prompt": "..."}
                                     -> raw chat completion (handy for the
                                        UI's "Test LLM" panel)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llm_service import llm_service


router = APIRouter(prefix="/llm", tags=["llm"])


class ModeRequest(BaseModel):
    mode: str = Field(..., description="One of 'Cloud' or 'Local'")
    persist: bool = True


class ChatRequest(BaseModel):
    prompt: str
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@router.get("/mode")
def get_mode() -> dict[str, Any]:
    return {"mode": llm_service.mode, **llm_service.health()["config"]}


@router.post("/mode")
def set_mode(req: ModeRequest) -> dict[str, Any]:
    if req.mode not in llm_service.VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {llm_service.VALID_MODES}, got {req.mode!r}",
        )
    new_mode = llm_service.set_mode(req.mode, persist=req.persist)
    return {"mode": new_mode, "ok": True, "health": llm_service.health()}


@router.get("/health")
def health() -> dict[str, Any]:
    return llm_service.health()


@router.get("/health/both")
def health_both() -> dict[str, Any]:
    return llm_service.health_both()


@router.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})
    try:
        text = llm_service.chat(
            messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return {"ok": True, "mode": llm_service.mode, "response": text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
