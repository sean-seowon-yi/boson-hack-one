# -*- coding: utf-8 -*-
"""
tools.step032_translation_llm
OpenAI-compatible client wrapper for Boson Hackathon endpoint (Qwen models).

Public API (kept stable):
    init_llm_model(model_name: str | None = None) -> None
    llm_response(messages: list[dict], device: str = "auto") -> str

Back-compat alias:
    openai_response(messages: list[dict]) -> str

Env (.env):
    QWEN_TRANSLATION_MODEL=Qwen3-32B-thinking-Hackathon
    BOSON_API_KEY=...
    BOSON_BASE_URL=https://hackathon.boson.ai/v1
    QWEN_MAX_RETRIES=4
    QWEN_INITIAL_BACKOFF=0.8
    QWEN_TIMEOUT=240
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from openai import APIError

# -----------------------
# Environment & constants
# -----------------------

load_dotenv()

MODEL_NAME: str = os.getenv("QWEN_TRANSLATION_MODEL", "Qwen3-32B-thinking-Hackathon")
BOSON_API_KEY: str = os.getenv("BOSON_API_KEY", "")
BOSON_BASE_URL: str = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")

# Tunables (do not alter prompts/semantics)
DEFAULT_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "240"))           # seconds; server still enforces limits
MAX_RETRIES = int(os.getenv("QWEN_MAX_RETRIES", "4"))             # total attempts = 1 + MAX_RETRIES
INITIAL_BACKOFF = float(os.getenv("QWEN_INITIAL_BACKOFF", "0.8")) # seconds

if not BOSON_API_KEY:
    logger.warning("BOSON_API_KEY is not set; calls will fail until provided.")

# -----------------------
# Client initialization
# -----------------------

_client: Optional[OpenAI] = None
_model_logged: Optional[str] = None

def _get_client() -> OpenAI:
    """Lazy, singleton client; avoids re-instantiation overhead."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=BOSON_API_KEY, base_url=BOSON_BASE_URL)
    return _client

def init_llm_model(model_name: Optional[str] = None) -> None:
    """
    Backwards-compat stub: logs the model/base URL selection (once per model).
    """
    global _model_logged
    chosen = model_name or MODEL_NAME
    if _model_logged != chosen:
        _model_logged = chosen
        logger.info(f"[LLM init] Using hosted model: {chosen} @ {BOSON_BASE_URL}")

# -----------------------
# Helpers
# -----------------------

def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Trim Nones/empties without changing prompt content.
    Keeps order; removes messages with falsy 'content' to prevent server rejections.
    """
    cleaned: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned or messages  # fall back to original if everything was stripped

def _should_retry(e: Exception) -> Tuple[bool, str]:
    """
    Decide whether to retry on a given exception.
    Retries on 429, >=500, and transient network errors from the OpenAI client.
    """
    # OpenAI APIError has .status_code for HTTP
    if isinstance(e, APIError):
        sc = getattr(e, "status_code", None)
        if sc in (429, 408):   # rate limit / timeout
            return True, f"HTTP {sc}"
        if sc is not None and sc >= 500:
            return True, f"HTTP {sc}"
        return False, f"HTTP {sc}"
    # Generic transient errors (connection resets, etc.)
    etxt = str(e).lower()
    transient = any(k in etxt for k in [
        "temporarily", "timeout", "timed out", "connection reset",
        "connection aborted", "server disconnected", "remote end closed",
        "read error", "write error", "unreachable", "rate limit"
    ])
    return (True, "transient") if transient else (False, "non-transient")

def _backoff_sleep(attempt: int) -> None:
    """
    Exponential backoff with jitter: t = INITIAL_BACKOFF * 2^attempt + U[0, 0.25]
    """
    delay = INITIAL_BACKOFF * (2 ** attempt) + random.random() * 0.25
    time.sleep(delay)

# -----------------------
# Core chat call
# -----------------------

def _chat_completion(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Low-level wrapper for a single chat.completions.create call with retries.
    Returns the first message's content as a stripped string.
    """
    if not BOSON_API_KEY:
        raise RuntimeError("BOSON_API_KEY is missing. Set it in your .env")

    chosen_model = model or MODEL_NAME
    init_llm_model(chosen_model)

    msgs = _sanitize_messages(messages)
    kwargs: Dict[str, Any] = {"model": chosen_model, "messages": msgs, "timeout": timeout}
    if extra_body:
        kwargs["extra_body"] = extra_body

    client = _get_client()
    last_err: Optional[Exception] = None

    # Attempt loop (1 + MAX_RETRIES total tries)
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            # Defensive: support both string and list segments if present
            content = getattr(resp.choices[0].message, "content", "") or ""
            if isinstance(content, list):
                # Some providers may return a list of segments; join their 'text' fields if any
                joined = " ".join(
                    seg.get("text", "") if isinstance(seg, dict) else str(seg)
                    for seg in content
                ).strip()
                return joined
            return str(content).strip()
        except Exception as e:
            last_err = e
            do_retry, reason = _should_retry(e)
            if attempt < MAX_RETRIES and do_retry:
                logger.debug(f"[LLM] retry {attempt+1}/{MAX_RETRIES} due to {reason}: {e}")
                _backoff_sleep(attempt)
                continue
            # No more retries or not retryable
            logger.warning(f"[LLM] final failure after {attempt+1} attempt(s): {e}")
            break

    # Graceful fallback: return empty string rather than raising,
    # to preserve old behavior patterns in upstream callers that already handle empties.
    return ""

# -----------------------
# Public APIs
# -----------------------

def llm_response(messages: List[Dict[str, Any]], device: str = "auto") -> str:
    """
    Hosted LLM response via Boson OpenAI-compatible API.
    Mirrors previous behavior: returns the first choice content as a string.
    """
    # 'device' retained for compatibility; intentionally unused.
    return _chat_completion(messages)

def openai_response(messages: List[Dict[str, Any]]) -> str:
    """Alias kept for compatibility with older modules."""
    return _chat_completion(messages)

# -----------------------
# CLI quick test
# -----------------------

if __name__ == "__main__":
    test = [{"role": "user", "content": "Briefly introduce yourself in one short sentence."}]
    logger.info(f"Using model: {MODEL_NAME} @ {BOSON_BASE_URL}")
    print(llm_response(test))
