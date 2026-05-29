from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.connectors import router as connectors_router
from app.api.pipelines import router as pipelines_router
from app.api.websocket import router as websocket_router
from app.api.aws_glue import router as aws_glue_router
from app.api.agent_api import router as agent_router
from app.api.runbooks import router as runbooks_router      # NEW
from app.api.metrics import router as metrics_router        # NEW


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth_router)
api_router.include_router(connectors_router)
api_router.include_router(pipelines_router)
api_router.include_router(aws_glue_router)
api_router.include_router(agent_router)
api_router.include_router(runbooks_router)       # NEW: /runbooks/*
api_router.include_router(metrics_router)        # NEW: /metrics/*

# WebSocket route mounted at root (no /api/v1) so the URL is just /ws
__all__ = ["api_router", "websocket_router"]
