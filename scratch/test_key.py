import httpx

keys = [
    "IotlgX9OC7gWRj0WqHuT5xdhTNne",
    "IotlgX9OC7gWRj0WqHuT5xdhT1LNkNne"
]

for key in keys:
    print(f"Testing key: {key}")
    try:
        r = httpx.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5
            },
            timeout=10.0
        )
        print(f"Status code: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"Error: {e}")
