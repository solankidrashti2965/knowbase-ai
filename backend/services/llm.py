"""
LLM service — wraps Groq API
Provides async generate_response() used by the RAG pipeline.
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)
from groq import AsyncGroq

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        _client = AsyncGroq(api_key=api_key)
    return _client

async def generate_response(prompt: str) -> str:
    """Send a prompt to Groq and return the text response."""
    client = _get_client()
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content
