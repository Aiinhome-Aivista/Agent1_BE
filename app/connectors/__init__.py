"""Connector registry & factory."""
from app.connectors.base import (
    BaseConnector, NormalizedPipeline, NormalizedRun, NormalizedLog,
)
from app.connectors.aws_glue import AWSGlueConnector

from app.connectors.adf import ADFConnector
from app.connectors.databricks import DatabricksConnector
from app.connectors.git import GitConnector
from app.models.connector import ConnectorType


_REGISTRY: dict[ConnectorType, type[BaseConnector]] = {
    ConnectorType.ADF: ADFConnector,
    ConnectorType.DATABRICKS: DatabricksConnector,
    ConnectorType.GIT: GitConnector,
    ConnectorType.AWS_GLUE: AWSGlueConnector,
}


def get_connector(type_: ConnectorType, credentials: dict) -> BaseConnector:
    cls = _REGISTRY.get(type_)
    if cls is None:
        raise ValueError(f"Unsupported connector type: {type_}")
    return cls(credentials)


__all__ = [
    "BaseConnector",
    "NormalizedPipeline",
    "NormalizedRun",
    "NormalizedLog",
    "ADFConnector",
    "DatabricksConnector",
    "GitConnector",
    "get_connector",
    "AWSGlueConnector", 
]
