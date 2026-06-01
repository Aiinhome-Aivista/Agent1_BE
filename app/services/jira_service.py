import logging
import requests
from requests.auth import HTTPBasicAuth
from app.core.config import settings

logger = logging.getLogger(__name__)

def create_jira_ticket(
    summary: str,
    description: str,
    assign_to_human: bool = True,
    labels: list[str] = None,
    assignee_account_id: str | None = None,
    support_group: str | None = None,
):
    if not settings.JIRA_BASE_URL or not settings.JIRA_USER_EMAIL or not settings.JIRA_API_TOKEN:
        logger.warning("Jira config missing, returning mock ticket.")
        return {"key": "MOCK-123", "id": "1", "self": "", "group": support_group}

    url = f"{settings.JIRA_BASE_URL.rstrip('/')}/rest/api/3/issue"
    auth = HTTPBasicAuth(settings.JIRA_USER_EMAIL, settings.JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # Prepend the routed support group to the description so it's visible even
    # if the Jira project has no custom "team" field.
    full_description = description
    if support_group:
        full_description = f"Support group: {support_group}\n\n{description}"

    description_adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": full_description
                    }
                ]
            }
        ]
    }

    payload = {
        "fields": {
            "project": {
                "key": settings.JIRA_PROJECT_KEY
            },
            "summary": summary,
            "description": description_adf,
            "issuetype": {
                "name": "Bug"
            }
        }
    }

    if labels:
        payload["fields"]["labels"] = labels

    # Explicit individual assignment takes precedence over the configured
    # default human assignee.
    chosen_assignee = assignee_account_id or (
        settings.JIRA_HUMAN_ASSIGNEE_ACCOUNT_ID if assign_to_human else None
    )
    if chosen_assignee:
        payload["fields"]["assignee"] = {"accountId": chosen_assignee}

    try:
        response = requests.post(url, json=payload, headers=headers, auth=auth, timeout=10)
        response.raise_for_status()
        out = response.json()
        out["group"] = support_group
        return out
    except Exception as e:
        logger.error(f"Failed to create Jira ticket: {e}")
        return {"key": "MOCK-123", "id": "1", "self": "", "group": support_group}
