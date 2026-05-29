import os
import requests
import json
from dotenv import load_dotenv

# Load the environment variables from .env
load_dotenv()

def test_jira_connection():
    base_url = os.getenv('JIRA_BASE_URL', '').rstrip('/')
    project_key = os.getenv('JIRA_PROJECT_KEY')
    email = os.getenv('JIRA_USER_EMAIL')
    token = os.getenv('JIRA_API_TOKEN')
    account_id = os.getenv('JIRA_HUMAN_ASSIGNEE_ACCOUNT_ID')
    
    print("--- Testing Jira Connection ---")
    print(f"JIRA_BASE_URL: {base_url}")
    print(f"JIRA_PROJECT_KEY: {project_key}")
    print(f"JIRA_USER_EMAIL: {email}")
    print(f"ASSIGNEE ACCOUNT_ID: {account_id}")
    print("-------------------------------\n")
    
    if not all([base_url, project_key, email, token]):
        print("[FAILED:] Missing Jira credentials in .env file.")
        return
        
    # First, list all accessible projects to debug
    print("Fetching accessible Jira projects...")
    try:
        proj_url = f"{base_url}/rest/api/3/project"
        proj_resp = requests.get(
            proj_url,
            headers={"Accept": "application/json"},
            auth=(email, token),
            timeout=10
        )
        proj_resp.raise_for_status()
        projects = proj_resp.json()
        print(f"Found {len(projects)} accessible projects:")
        for p in projects:
            print(f" - {p.get('name')} (Key: {p.get('key')})")
        print("\n")
    except Exception as e:
        print("Failed to fetch projects.")
        print(e)
        
    print("Attempting to create a test ticket in Jira...")
    
    url = f"{base_url}/rest/api/3/issue"
    
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": "[TEST] Jira API Connection Verification",
            "issuetype": {"name": "Bug"},
            "labels": ["test-ticket"],
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "text": "This is an automated test to verify that the AIOps pipeline monitor can successfully communicate with the Jira REST API.",
                                "type": "text"
                            }
                        ]
                    }
                ]
            }
        }
    }
    
    if account_id and account_id != "john.doe@example.com":
        payload["fields"]["assignee"] = {"accountId": account_id}
        
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            auth=(email, token),
            timeout=10
        )
        
        response.raise_for_status()
        data = response.json()
        
        print("\n[SUCCESS!]")
        print(f"Ticket Created: {data.get('key')}")
        print(f"Ticket URL: {base_url}/browse/{data.get('key')}")
        print("Please check your Jira board to confirm it appears.")
        
    except requests.exceptions.HTTPError as e:
        print("\n[FAILED!]")
        print(f"Status Code: {e.response.status_code}")
        print(f"Response from Jira: {e.response.text}")
    except Exception as e:
        print("\n[EXCEPTION OCCURRED!]")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_jira_connection()
