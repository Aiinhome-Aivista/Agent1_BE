"""Pydantic schemas for Connectors.

Credentials are accepted as strongly-typed payloads per connector type so the
frontend can render the right form, and so we never silently store the wrong
shape.
"""
from datetime import datetime
from typing import Literal, Annotated, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator

from app.models.connector import ConnectorType, ConnectorStatus


# --- Credential payloads ---------------------------------------------------

class ADFCredentials(BaseModel):
    type: Literal["ADF"] = "ADF"
    tenant_id: str
    client_id: str
    client_secret: str
    subscription_id: str
    resource_group: str
    factory_name: str


class DatabricksCredentials(BaseModel):
    type: Literal["DATABRICKS"] = "DATABRICKS"
    workspace_url: str = Field(..., description="e.g. https://adb-xxx.azuredatabricks.net")
    personal_access_token: str


class GitCredentials(BaseModel):
    type: Literal["GIT"] = "GIT"
    provider: Literal["github", "gitlab"] = "github"
    token: str
    owner: str = Field(..., description="GitHub user/org or GitLab namespace")
    repo: str | None = None  # if None, sync all repos owner has access to

    @field_validator("owner", "repo")
    @classmethod
    def prevent_urls(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if "http://" in v or "https://" in v or "github.com" in v:
            raise ValueError("Please provide only the name, not a full URL")
        if " " in v:
            raise ValueError("Spaces are not allowed in the owner or repository name")
        return v.strip()


CredentialsPayload = Annotated[
    Union[ADFCredentials, DatabricksCredentials, GitCredentials],
    Field(discriminator="type"),
]


# --- Connector CRUD --------------------------------------------------------

class ConnectorCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    type: ConnectorType
    credentials: CredentialsPayload


class ConnectorUpdate(BaseModel):
    name: str | None = None
    credentials: CredentialsPayload | None = None


class ConnectorOut(BaseModel):
    id: int
    name: str
    type: ConnectorType
    status: ConnectorStatus
    last_synced_at: datetime | None
    last_error: str | None
    config: dict | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConnectorTestResult(BaseModel):
    success: bool
    message: str
    detected_pipelines: int | None = None
