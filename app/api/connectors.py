"""Connector endpoints."""
import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.connectors import get_connector
from app.core.database import get_db
from app.core.security import encrypt_secret, decrypt_secret
from app.models import User, Connector, ConnectorStatus
from app.schemas import (
    ConnectorCreate, ConnectorUpdate, ConnectorOut, ConnectorTestResult,
)
from app.services.sync_service import sync_connector
from app.services.email_service import send_pipeline_error_email


router = APIRouter(prefix="/connectors", tags=["connectors"])


@router.get("", response_model=list[ConnectorOut])
def list_connectors(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Connector]:
    return db.query(Connector).all()


@router.post("", response_model=ConnectorOut, status_code=status.HTTP_201_CREATED)
def create_connector(
    payload: ConnectorCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Connector:
    # The Pydantic discriminator already enforced credentials.type matches
    # ConnectorType conceptually, but also enforce literally:
    if payload.credentials.type != payload.type.value:
        raise HTTPException(400, "credentials.type must match connector type")

    cred_dict = payload.credentials.model_dump()
    cred_dict.pop("type", None)
    encrypted = encrypt_secret(json.dumps(cred_dict))

    connector = Connector(
        name=payload.name,
        type=payload.type,
        encrypted_credentials=encrypted,
        status=ConnectorStatus.PENDING,
    )
    db.add(connector)
    db.commit()
    db.refresh(connector)
    return connector


@router.get("/{connector_id}", response_model=ConnectorOut)
def get_connector_by_id(
    connector_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Connector:
    c = db.query(Connector).filter(
        Connector.id == connector_id).first()
    if not c:
        raise HTTPException(404, "Connector not found")
    return c


@router.patch("/{connector_id}", response_model=ConnectorOut)
def update_connector(
    connector_id: int,
    payload: ConnectorUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Connector:
    c = db.query(Connector).filter(
        Connector.id == connector_id).first()
    if not c:
        raise HTTPException(404, "Connector not found")

    if payload.name is not None:
        c.name = payload.name
    if payload.credentials is not None:
        if payload.credentials.type != c.type.value:
            raise HTTPException(400, "credentials.type cannot change")
        cred_dict = payload.credentials.model_dump()
        cred_dict.pop("type", None)
        c.encrypted_credentials = encrypt_secret(json.dumps(cred_dict))
        c.status = ConnectorStatus.PENDING
    db.commit()
    db.refresh(c)
    return c


@router.delete("/{connector_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connector(
    connector_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    c = db.query(Connector).filter(
        Connector.id == connector_id).first()
    if not c:
        raise HTTPException(404, "Connector not found")
    db.delete(c)
    db.commit()


@router.post("/{connector_id}/test", response_model=ConnectorTestResult)
def test_connector(
    connector_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConnectorTestResult:
    c = db.query(Connector).filter(
        Connector.id == connector_id).first()
    if not c:
        raise HTTPException(404, "Connector not found")

    creds = json.loads(decrypt_secret(c.encrypted_credentials))
    client = get_connector(c.type, creds)
    ok, msg = client.test_connection()
    detected = None
    if ok:
        try:
            detected = len(client.list_pipelines())
        except Exception:
            detected = None
        c.status = ConnectorStatus.CONNECTED
        c.last_error = None
    else:
        c.status = ConnectorStatus.ERROR
        c.last_error = msg
    db.commit()
    return ConnectorTestResult(success=ok, message=msg, detected_pipelines=detected)


@router.post("/{connector_id}/sync", status_code=status.HTTP_202_ACCEPTED)
async def trigger_sync(
    connector_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    c = db.query(Connector).filter(
        Connector.id == connector_id).first()

    if not c:
        raise HTTPException(404, "Connector not found")

    try:

        stats = await sync_connector(db, c)

        return {
            "queued": True,
            "stats": stats
        }

    except Exception as e:

        # Send email to logged in user
        send_pipeline_error_email(
            to_email=user.email,
            pipeline_name=c.name,
            connector_name=c.type.value,
            error_message=str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )
