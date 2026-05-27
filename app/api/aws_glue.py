"""AWS Glue API endpoints."""
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.connectors.aws_glue import AWSGlueConnector
from app.core.database import get_db
from app.core.security import encrypt_secret
from app.models import User, Connector, ConnectorType, ConnectorStatus

router = APIRouter(prefix="/aws-glue", tags=["AWS Glue"])


class AWSGlueConfig(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str = "ap-south-1"
    name: str = "aws-glue"   # user-friendly name shown in the connector table


aws_sessions = {}


@router.post("/connect")
async def connect_aws(
    config: AWSGlueConfig,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    credentials = {
        "aws_access_key_id": config.aws_access_key_id,
        "aws_secret_access_key": config.aws_secret_access_key,
        "region_name": config.region_name,
    }

    try:
        connector_client = AWSGlueConnector(credentials)
        ok, message = connector_client.test_connection()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not ok:
        raise HTTPException(status_code=400, detail=message)

    # Save (or update) the connector row in the database
    encrypted = encrypt_secret(json.dumps(credentials))

    existing = (
        db.query(Connector)
        .filter(
            Connector.type == ConnectorType.AWS_GLUE,
            Connector.name == config.name,
        )
        .first()
    )

    if existing:
        existing.encrypted_credentials = encrypted
        existing.status = ConnectorStatus.CONNECTED
        existing.last_error = None
        db.commit()
        db.refresh(existing)
        db_connector = existing
    else:
        db_connector = Connector(
            name=config.name,
            type=ConnectorType.AWS_GLUE,
            encrypted_credentials=encrypted,
            status=ConnectorStatus.CONNECTED,
        )
        db.add(db_connector)
        db.commit()
        db.refresh(db_connector)

    aws_sessions["default"] = connector_client

    return {
        "success": True,
        "message": message,
        "connector_id": db_connector.id,
        "connector_name": db_connector.name,
        "status": db_connector.status,
    }


@router.get("/jobs")
async def get_jobs():
    connector = aws_sessions.get("default")
    if not connector:
        raise HTTPException(status_code=400, detail="AWS not connected")
    return connector.get_jobs()


@router.get("/jobs/{job_name}")
async def get_job_runs(job_name: str):
    connector = aws_sessions.get("default")
    if not connector:
        raise HTTPException(status_code=400, detail="AWS not connected")
    return connector.get_job_runs(job_name)