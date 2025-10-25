# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Pull config from .env
MODEL_NAME = os.getenv("QWEN_TRANSLATION_MODEL", "Qwen3-32B-thinking-Hackathon")
BASE_URL   = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
API_KEY    = os.getenv("BOSON_API_KEY", "")

# Optional generation controls
EXTRA_BODY = {
    "repetition_penalty": 1.1,
}

def boson_openai_response(messages):
    """
    OpenAI-compatible chat call against Boson Hackathon endpoint.
    Uses model from QWEN_TRANSLATION_MODEL.
    """
    if not API_KEY:
        raise RuntimeError("BOSON_API_KEY is not set in environment/.env")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        timeout=240,
        extra_body=EXTRA_BODY,
    )
    return resp.choices[0].message.content

# --- Backwards-compatible alias so existing imports keep working ---
def openai_response(messages):
    """Alias kept for compatibility with step030_translation.py."""
    return boson_openai_response(messages)

if __name__ == "__main__":
    test_message = [{"role": "user", "content": "Briefly introduce yourself."}]
    logger.info(f"Using Boson model: {MODEL_NAME} @ {BASE_URL}")
    print(openai_response(test_message))
