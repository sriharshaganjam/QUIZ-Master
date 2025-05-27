# mistral_api.py

import requests
import os

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Add your key in Streamlit secrets or .env

def call_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small",  # Use correct model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    res_json = response.json()

    try:
        return res_json["choices"][0]["message"]["content"].strip()
    except KeyError:
        return res_json["choices"][0]["text"].strip()
