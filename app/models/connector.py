"""
Connector model.

A connector represents a connection to a real third-party account:
- ADF: Azure Data Factory (auth via Azure AD service principal)
- DATABRICKS: Databricks workspace (PAT-based auth)
- GIT: GitHub / GitLab repositories (PAT-based auth)
"""
import enum
from datetime import datetime

# pyrefly: ignore [missing-import]
from sqlalchemy import (
    Column, Integer, String, DateTime, Enum, Text, ForeignKey, JSON
)
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import relationship

from app.core.database import Base


class ConnectorType(str, enum.Enum):
    ADF = "ADF"
    DATABRICKS = "DATABRICKS"
    GIT = "GIT"
    AWS_GLUE = "AWS_GLUE"


class ConnectorStatus(str, enum.Enum):
    PENDING = "PENDING"          # just created, not yet validated
    CONNECTED = "CONNECTED"      # auth verified, syncing
    ERROR = "ERROR"              # auth failed or API error
    DISABLED = "DISABLED"        # user paused it


class Connector(Base):
    __tablename__ = "connectors"

    id = Column(Integer, primary_key=True, autoincrement=True)

    name = Column(String(255), nullable=False)              # user-friendly name
    type = Column(Enum(ConnectorType), nullable=False)
    status = Column(Enum(ConnectorStatus), default=ConnectorStatus.PENDING)

    # Encrypted JSON blob - schema depends on connector type. Always decrypted via
    # the security helper before use; never returned to the API.
    # ADF:        {tenant_id, client_id, client_secret, subscription_id,
    #              resource_group, factory_name}
    # DATABRICKS: {workspace_url, personal_access_token}
    # GIT:        {provider: github|gitlab, token, owner, repo (optional)}
    encrypted_credentials = Column(Text, nullable=False)

    # Non-secret config / metadata
    config = Column(JSON, nullable=True, default=dict)

    last_synced_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    pipelines = relationship("Pipeline", back_populates="connector", cascade="all, delete-orphan")
