# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# Load environment variables
load_dotenv()

# Pull from .env (see your provided .env)
MODEL_NAME = os.getenv("QWEN_TRANSLATION_MODEL", "Qwen3-32B-thinking-Hackathon")
BOSON_API_KEY = os.getenv("BOSON_API_KEY")  # seansean
BOSON_BASE_URL = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")

# Initialize OpenAI-compatible client once
client = OpenAI(api_key=BOSON_API_KEY, base_url=BOSON_BASE_URL)

# Keep function signature for compatibility with existing imports/callers
def init_llm_model(model_name: str = None):
    """
    Kept for backwards compatibility. No-op now that we use a hosted model.
    """
    chosen = model_name or MODEL_NAME
    logger.info(f"[LLM init] Using hosted model: {chosen} @ {BOSON_BASE_URL}")

def llm_response(messages, device: str = "auto"):
    """
    Hosted LLM response via Boson OpenAI-compatible API.
    Returns string content (first choice), mirroring previous behavior.
    """
    # ensure model is "initialized" (log only)
    init_llm_model(MODEL_NAME)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        timeout=240,
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    test_message = [{"role": "user", "content": "Briefly introduce yourself."}]
    logger.info(f"Using model: {MODEL_NAME} @ {BOSON_BASE_URL}")
    print(llm_response(test_message))
