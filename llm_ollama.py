# llm_ollama.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"  # or another model you have pulled via `ollama pull mistral`

def query_llm(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        raise Exception(f"Error from Ollama: {response.status_code} - {response.text}")
