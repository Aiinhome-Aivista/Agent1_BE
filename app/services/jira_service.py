import logging
import requests
from requests.auth import HTTPBasicAuth
from app.core.config import settings

logger = logging.getLogger(__name__)

def create_jira_ticket(summary: str, description: str, assign_to_human: bool = True, labels: list[str] = None):
    if not settings.JIRA_BASE_URL or not settings.JIRA_USER_EMAIL or not settings.JIRA_API_TOKEN:
        logger.warning("Jira config missing, returning mock ticket.")
        return {"key": "MOCK-123", "id": "1", "self": ""}

    url = f"{settings.JIRA_BASE_URL.rstrip('/')}/rest/api/3/issue"
    auth = HTTPBasicAuth(settings.JIRA_USER_EMAIL, settings.JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    description_adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": description
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

    if assign_to_human and settings.JIRA_HUMAN_ASSIGNEE_ACCOUNT_ID:
        payload["fields"]["assignee"] = {
            "accountId": settings.JIRA_HUMAN_ASSIGNEE_ACCOUNT_ID
        }
        
    try:
        response = requests.post(url, json=payload, headers=headers, auth=auth, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to create Jira ticket: {e}")
        return {"key": "MOCK-123", "id": "1", "self": ""}
