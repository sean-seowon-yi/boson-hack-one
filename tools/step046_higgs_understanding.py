# -*- coding: utf-8 -*-
"""
tools/step046_higgs_understanding.py

Emotion scoring via Boson/Higgs using OpenAI-compatible APIs.
Primary path: /v1/chat/completions with {"type": "input_audio"} (mirrors your ASR usage).
Fallback: /v1/responses if supported.

Env (.env):
  BOSON_API_KEY=sk-...
  BOSON_BASE_URL=https://hackathon.boson.ai/v1
  HIGGS_U_CHAT_MODEL=Qwen3-32B-non-thinking-Hackathon    # default text LLM
  HIGGS_U_MODEL=higgs-audio-understanding-Hackathon      # optional responses model
"""

from __future__ import annotations
import io
import os
import re
import json
import time
import base64
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
from loguru import logger
from openai import OpenAI

DEFAULT_BASE = "https://hackathon.boson.ai/v1"
_TARGET_SR = 16000

@dataclass(frozen=True)
class EmotionScore:
    valence: float     # [-1, 1]
    arousal: float     # [-1, 1]
    label: str         # "happy"/"sad"/"angry"/"neutral"/"other"
    confidence: float  # [0,1]

# ---------------- utils ---------------- #

def _np_audio_to_b64_wav(y: np.ndarray, sr: int) -> str:
    """Encode float32 mono → 16k PCM16 WAV → base64."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != _TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_TARGET_SR)
        sr = _TARGET_SR
    y = np.clip(y, -1.0, 1.0)
    y16 = (y * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, y16, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _audio_cache_key(y: np.ndarray, sr: int) -> str:
    h = hashlib.sha1()
    h.update(y.tobytes())
    h.update(str(sr).encode())
    return h.hexdigest()

def _chat_model_name() -> str:
    return os.getenv("HIGGS_U_CHAT_MODEL", "Qwen3-32B-non-thinking-Hackathon")

def _responses_model_name() -> str:
    return os.getenv("HIGGS_U_MODEL", "higgs-audio-understanding-Hackathon")

def _client() -> OpenAI:
    api_key = os.getenv("BOSON_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BOSON_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE
    if not api_key:
        raise RuntimeError("Missing API key: set BOSON_API_KEY or OPENAI_API_KEY.")
    base_url = base_url.rstrip("/")
    logger.debug(f"[HiggsU] base_url={base_url}, key_present={bool(api_key)}")
    return OpenAI(api_key=api_key, base_url=base_url)

def _with_retries(call, *, retries=3, base_delay=0.4, max_delay=1.6):
    last = None
    for i in range(retries):
        try:
            return call()
        except Exception as e:
            last = e
            msg = str(e)
            if any(x in msg for x in ("429", "Rate limit", "timeout", "temporar", "5")) and i < retries - 1:
                time.sleep(min(max_delay, base_delay * (2 ** i)))
                continue
            raise last

# ---------------- primary path: chat.completions ---------------- #

def _score_via_chat_completions(client: OpenAI, b64wav: str) -> str:
    sys_msg = {
        "role": "system",
        "content": (
            "You are an emotion classifier for short speech clips. "
            "Return ONLY compact JSON with keys: "
            '{"valence": float(-1..1), "arousal": float(-1..1), '
            '"label": "happy|sad|angry|neutral|other", "confidence": float(0..1)}'
        ),
    }
    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this 16k PCM16 WAV (base64). Return the JSON only."},
            {"type": "input_audio", "input_audio": {"data": b64wav, "format": "wav"}},
        ],
    }

    def _call():
        r = client.chat.completions.create(
            model=_chat_model_name(),
            messages=[sys_msg, user_msg],
            temperature=0,
            top_p=1.0,
            max_tokens=256,
        )
        return (r.choices[0].message.content or "").strip()

    return _with_retries(_call)

# ---------------- fallback: responses (optional) ---------------- #

def _try_via_responses(client: OpenAI, b64wav: str) -> Optional[str]:
    def _call():
        r = client.responses.create(
            model=_responses_model_name(),
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64wav, "format": "wav"}},
                    {"type": "text",
                     "text": ('Analyze the emotion and return JSON: '
                              '{"valence": float(-1..1), "arousal": float(-1..1), '
                              '"label": "happy|sad|angry|neutral|other", "confidence": float(0..1)}')},
                ],
            }],
        )
        # Try standard fields
        try:
            return r.output[0].content[0].text.strip()
        except Exception:
            return getattr(r, "output_text", "").strip()

    try:
        return _with_retries(_call)
    except Exception as e:
        msg = str(e)
        if "404" in msg or "Invalid URL" in msg or "/v1/responses" in msg:
            logger.info("[HiggsU] /v1/responses not supported; keeping chat.completions path")
            return None
        logger.error(f"[HiggsU] responses.create failed: {e}")
        return None

# ---------------- cached scoring wrapper ---------------- #

@lru_cache(maxsize=256)
def _score_cached(key: str, b64wav: str) -> EmotionScore:
    client = _client()
    try:
        txt = _score_via_chat_completions(client, b64wav)
    except Exception as e_chat:
        logger.warning(f"[HiggsU] chat.completions failed, trying responses: {e_chat}")
        txt = _try_via_responses(client, b64wav)
        if txt is None:
            return EmotionScore(0.0, 0.0, "neutral", 0.0)

    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        logger.warning(f"[HiggsU] Could not parse JSON from: {txt[:160] if txt else 'EMPTY'}")
        return EmotionScore(0.0, 0.0, "neutral", 0.0)
    try:
        j = json.loads(m.group(0))
        v = max(-1.0, min(1.0, float(j.get("valence", 0.0))))
        a = max(-1.0, min(1.0, float(j.get("arousal", 0.0))))
        lab = str(j.get("label", "neutral")).lower()
        conf = max(0.0, min(1.0, float(j.get("confidence", 0.0))))
        return EmotionScore(v, a, lab, conf)
    except Exception as e:
        logger.warning(f"[HiggsU] JSON parse error: {e} from {txt[:160]}...")
        return EmotionScore(0.0, 0.0, "neutral", 0.0)

# ---------------- public API ---------------- #

def score_emotion(y: np.ndarray, sr: int, force: bool = False) -> EmotionScore:
    """
    Score emotion of an audio clip (float32 mono).
    Uses memoization for identical audio buffers.
    """
    y = np.asarray(y, dtype=np.float32, copy=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    key = _audio_cache_key(y, sr)
    b64 = _np_audio_to_b64_wav(y, sr)
    if force:
        _score_cached.cache_clear()
    return _score_cached(key, b64)
