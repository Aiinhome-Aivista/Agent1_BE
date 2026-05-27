"""Real-time WebSocket endpoint."""
import logging
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.security import decode_token
from app.models import User
from app.websockets.manager import manager
from app.api.aws_glue import aws_sessions

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT token (same as REST API)"),
):
    """Authenticate via ?token=... query param then stream events."""
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        await websocket.close(code=4401, reason="Invalid token")
        return

    user_id = int(payload["sub"])

    # Verify user still exists and is active
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            await websocket.close(code=4403, reason="Inactive user")
            return
    finally:
        db.close()

    # await manager.connect(user_id, websocket)
    # try:
    #     # Send a hello so the client knows it's live
    #     await websocket.send_json({"event": "hello", "data": {"user_id": user_id}})
    #     while True:
    #         # We don't expect messages from the client, but receive_text keeps
    #         # the connection alive and lets us detect disconnects.
    #         msg = await websocket.receive_text()
    #         if msg == "ping":
    #             await websocket.send_json({"event": "pong", "data": {}})
    # except WebSocketDisconnect:
    #     pass
    # except Exception as e:
    #     logger.warning("WS error: %s", e)
    # finally:
    #     await manager.disconnect(user_id, websocket)

    await manager.connect(user_id, websocket)

    try:
        # Build and send initial snapshot so the client store hydrates immediately
        snap_db: Session = SessionLocal()

        try:
            from app.models.agent_models import Incident as _Inc  # noqa: PLC0415
            from app.services.incident_service import _incident_to_dict  # noqa: PLC0415
            from app.api.agent_api import (
                _AGENT_DEFINITIONS,
                _agent_states,
            )  # noqa: PLC0415

            incidents_raw = (
                snap_db.query(_Inc)
                .order_by(_Inc.detected_at.desc())
                .limit(50)
                .all()
            )

            incidents = [_incident_to_dict(i) for i in incidents_raw]

            agents = [
                {
                    "role": d["role"],
                    "name": d["name"],
                    "description": d["description"],
                    "color": d["color"],
                    **_agent_states.get(d["role"], {}),
                }
                for d in _AGENT_DEFINITIONS
            ]

        finally:
            snap_db.close()

        await websocket.send_json(
            {
                "event": "snapshot",
                "payload": {
                    "incidents": incidents,
                    "agents": agents,
                    "pipelines": [],  # pipelines come from REST /pipelines
                    "simulating": False,
                },
            }
        )

        while True:
            msg = await websocket.receive_text()

            if msg == "ping":
                await websocket.send_json(
                    {
                        "event": "pong",
                        "payload": {},
                    }
                )

    except WebSocketDisconnect:
        pass

    finally:
        await manager.disconnect(user_id, websocket)


@router.websocket("/ws/aws-glue/{job_name}")
async def aws_glue_ws(websocket: WebSocket, job_name: str):
    await websocket.accept()

    connector = aws_sessions.get("default")

    if not connector:
        await websocket.send_json({"error": "AWS not connected"})
        await websocket.close()
        return

    try:
        while True:
            runs = connector.get_job_runs(job_name)

            if runs:
                await websocket.send_json(runs[0])

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        print("AWS Glue websocket disconnected")