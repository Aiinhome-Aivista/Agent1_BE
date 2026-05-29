import requests

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMSIsImV4cCI6MTc3OTEyMDYwMH0.7_9xFHRoZ6C_xQUOju5O67hjtD3JsWttHi2CcYenmpc"
r = requests.get(
    "http://localhost:3002/api/v1/incidents?tab=open&limit=2",
    headers={"Authorization": f"Bearer {token}"},
)
data = r.json()
print(f"Status: {r.status_code}, Count: {len(data)}")
if data:
    inc = data[0]
    keys = ["id", "pipeline_name", "is_active", "pipeline_id", "escalation_count"]
    for k in keys:
        print(f"  {k}: {inc.get(k)}")

# Test events endpoint
r2 = requests.get(
    f"http://localhost:3002/api/v1/incidents/{data[0]['id']}/events",
    headers={"Authorization": f"Bearer {token}"},
)
print(f"\nEvents endpoint: {r2.status_code}, events: {len(r2.json())}")
