# -*- coding: utf-8 -*-
"""
tools.step032_translation_llm
OpenAI-compatible client wrapper for Boson Hackathon endpoint (Qwen models).

Public API (kept stable):
    init_llm_model(model_name: str | None = None) -> None
    llm_response(messages: list[dict], device: str = "auto") -> str

Back-compat alias:
    openai_response(messages: list[dict]) -> str
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

load_dotenv()

# ------------------------------------------------------------------------------------
# Language normalization (restricted to 5 langs; optional helpers — NOT used implicitly)
# ------------------------------------------------------------------------------------
# These helpers are included for consistency with other steps; they do not alter
# existing behavior unless explicitly called by the caller constructing messages.
_LANG_ALIASES: Dict[str, str] = {
    # Simplified Chinese
    "zh-cn": "zh-cn", "zh_cn": "zh-cn", "cn": "zh-cn",
    "chinese (中文)": "zh-cn", "chinese": "zh-cn", "中文": "zh-cn",
    "simplified chinese (简体中文)": "zh-cn", "simplified chinese": "zh-cn", "简体中文": "zh-cn",

    # English
    "en": "en", "english": "en",

    # Korean
    "ko": "ko", "korean": "ko", "한국어": "ko",

    # Spanish
    "es": "es", "spanish": "es", "español": "es",

    # French
    "fr": "fr", "french": "fr", "français": "fr",
}
_ALLOWED_LANGS = {"zh-cn", "en", "ko", "es", "fr"}

def normalize_target_lang(lang: Optional[str]) -> str:
    """
    Optional utility for callers to normalize a target language into a canonical code.
    Does NOT affect llm_response behavior unless used by the caller when crafting prompts.
    """
    key = (lang or "").strip().lower()
    code = _LANG_ALIASES.get(key, key)
    if code not in _ALLOWED_LANGS:
        raise ValueError(f"Unsupported target language: {lang}")
    return code

# ------------------------------------------------------------------------------------
# Model/env configuration (unchanged)
# ------------------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("QWEN_TRANSLATION_MODEL", "Qwen3-32B-thinking-Hackathon")
BOSON_API_KEY: str = os.getenv("BOSON_API_KEY", "")
BOSON_BASE_URL: str = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")

DEFAULT_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "240"))
MAX_RETRIES = int(os.getenv("QWEN_MAX_RETRIES", "4"))
INITIAL_BACKOFF = float(os.getenv("QWEN_INITIAL_BACKOFF", "0.7"))

if not BOSON_API_KEY:
    logger.warning("BOSON_API_KEY is not set; calls will fail until provided.")

_client: Optional[OpenAI] = None
_model_logged: Optional[str] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=BOSON_API_KEY, base_url=BOSON_BASE_URL)
    return _client

def init_llm_model(model_name: Optional[str] = None) -> None:
    global _model_logged
    chosen = model_name or MODEL_NAME
    if _model_logged != chosen:
        _model_logged = chosen
        logger.info(f"[LLM init] Using hosted model: {chosen} @ {BOSON_BASE_URL}")

def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
        elif role and isinstance(content, list) and content:
            # Allow for multimodal segments (e.g., input_audio) if caller passes them
            cleaned.append({"role": role, "content": content})
    return cleaned or messages

def _should_retry(e: Exception) -> Tuple[bool, str]:
    if isinstance(e, APIError):
        sc = getattr(e, "status_code", None)
        if sc in (429, 408) or (sc is not None and sc >= 500):
            return True, f"HTTP {sc}"
        return False, f"HTTP {sc}"
    etxt = str(e).lower()
    transient = any(k in etxt for k in [
        "temporarily", "timeout", "timed out", "connection reset",
        "connection aborted", "server disconnected", "remote end closed",
        "read error", "write error", "unreachable", "rate limit"
    ])
    return (True, "transient") if transient else (False, "non-transient")

def _backoff_sleep(attempt: int) -> None:
    delay = INITIAL_BACKOFF * (2 ** attempt) + random.random() * 0.25
    time.sleep(delay)

def _chat_completion(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    if not BOSON_API_KEY:
        raise RuntimeError("BOSON_API_KEY is missing. Set it in your .env")

    chosen_model = model or MODEL_NAME
    init_llm_model(chosen_model)

    msgs = _sanitize_messages(messages)
    kwargs: Dict[str, Any] = {
        "model": chosen_model,
        "messages": msgs,
        "timeout": timeout,
        "extra_body": {
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": int(os.getenv("QWEN_MAX_TOKENS", "256")),
        }
    }
    if extra_body:
        kwargs["extra_body"].update(extra_body)

    client = _get_client()
    last_err: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = getattr(resp.choices[0].message, "content", "") or ""
            # Qwen responses may be either string or list segments
            if isinstance(content, list):
                # Concatenate text parts; ignore non-text here since callers expect text
                content = " ".join(
                    seg.get("text", "") if isinstance(seg, dict) else str(seg)
                    for seg in content
                )
            return str(content).strip()
        except Exception as e:
            last_err = e
            do_retry, reason = _should_retry(e)
            if attempt < MAX_RETRIES and do_retry:
                logger.debug(f"[LLM] retry {attempt+1}/{MAX_RETRIES} due to {reason}: {e}")
                _backoff_sleep(attempt)
                continue
            logger.warning(f"[LLM] final failure after {attempt+1} attempt(s): {e}")
            break

    return ""

def llm_response(messages: List[Dict[str, Any]], device: str = "auto") -> str:
    """
    Thin wrapper that returns model text content.
    Callers are responsible for crafting prompts and enforcing schemas.
    """
    return _chat_completion(messages)

def openai_response(messages: List[Dict[str, Any]]) -> str:
    """
    Back-compat alias.
    """
    return _chat_completion(messages)

if __name__ == "__main__":
    test = [{"role": "user", "content": "Reply with a single word: ok"}]
    logger.info(f"Using model: {MODEL_NAME} @ {BOSON_BASE_URL}")
    print(llm_response(test))
