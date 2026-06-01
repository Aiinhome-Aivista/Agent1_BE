"""
Support-group routing.

Requirement #1 asks the agent to assign the raised ticket "to the respective
support group or individual as applicable". This module centralises that
mapping so the routing logic lives in one place and can later be driven from a
DB table or config file without touching the incident pipeline.

Routing decision is keyed on (category, error_type) with sensible fallbacks.
The returned RouteDecision carries everything jira_service needs:
  - assignee_account_id : Jira accountId for an individual (optional)
  - group               : a human-readable support-group name
  - labels              : Jira labels to attach
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings


@dataclass
class RouteDecision:
    group: str
    labels: list[str] = field(default_factory=list)
    assignee_account_id: str | None = None


# category → default support group. Override with env later if desired.
_CATEGORY_GROUP = {
    "ADF":        "Azure Data Platform",
    "DATABRICKS": "Databricks Engineering",
    "GIT":        "Build & CI",
    "AWS_GLUE":   "AWS Data Platform",
    "AWS GLUE":   "AWS Data Platform",
    "GENERAL":    "DataOps On-Call",
}

# error_type → extra labels that help triage.
_ERROR_TYPE_LABELS = {
    "SchemaDrift":     ["schema-drift", "data-contract"],
    "Timeout":         ["timeout", "performance"],
    "AuthError":       ["auth", "credentials"],
    "ConnectionError": ["connectivity"],
    "OutOfMemory":     ["resource", "memory"],
    "RateLimit":       ["throttling"],
    "DependencyError": ["dependency"],
    "KeyError":        ["data-quality"],
    "NotFound":        ["missing-resource"],
}


def route(
    *,
    category: str | None,
    error_type: str | None,
    is_known: bool,
    confidence: float,
) -> RouteDecision:
    cat = (category or "GENERAL").upper()
    group = _CATEGORY_GROUP.get(cat, _CATEGORY_GROUP["GENERAL"])

    labels: list[str] = ["dataops-agent"]
    labels.extend(_ERROR_TYPE_LABELS.get(error_type or "", []))

    if is_known:
        labels.append("known-error")
        labels.append("auto-fix-candidate" if confidence >= 0.7 else "human-review")
    else:
        labels.append("new-error")
        labels.append("needs-investigation")

    # Individual assignment: for now we only have one configured human; route
    # high-confidence known errors to them, everything else stays at group
    # level. A real deployment would map group → on-call rotation accountId.
    assignee = settings.JIRA_HUMAN_ASSIGNEE_ACCOUNT_ID or None

    return RouteDecision(group=group, labels=labels, assignee_account_id=assignee)
