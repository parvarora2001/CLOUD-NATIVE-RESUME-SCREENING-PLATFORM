import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "gemma2:2b",
    "prompt": "What is AI?",
    "stream": False
})

print(response.json())
