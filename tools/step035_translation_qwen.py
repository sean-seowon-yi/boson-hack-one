# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env
load_dotenv()

# Extra inference controls (kept from original)
extra_body = {
    "repetition_penalty": 1.1,
}

# Use your .env model + Boson OpenAI-compatible endpoint/key
MODEL_NAME = os.getenv("QWEN_TRANSLATION_MODEL", "Qwen3-32B-thinking-Hackathon")
BASE_URL = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
API_KEY = os.getenv("BOSON_API_KEY")  # .env: BOSON_API_KEY=seansean

def qwen_response(messages):
    """
    Thin wrapper around OpenAI-compatible chat completions.
    Reads model/endpoint/key from .env.
    """
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        timeout=240,
        extra_body=extra_body,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    test_message = [{"role": "user", "content": "Briefly introduce yourself."}]
    logger.info(f"Using model: {MODEL_NAME} @ {BASE_URL}")
    response = qwen_response(test_message)
    print(response)
