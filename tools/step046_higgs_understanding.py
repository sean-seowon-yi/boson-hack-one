# -*- coding: utf-8 -*-
"""
tools/step046_higgs_understanding.py
Emotion scoring via Boson/Higgs using OpenAI-compatible APIs.

Env (.env):
  BOSON_API_KEY=sk-...
  BOSON_BASE_URL=https://hackathon.boson.ai/v1
  HIGGS_EMO_MODEL=higgs-audio-understanding-Hackathon

Public API:
    score_emotion(y: np.ndarray, sr: int, force: bool = False) -> EmotionScore
    EmotionScore dataclass
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

import numpy as np
import librosa
import soundfile as sf
from dotenv import load_dotenv
from loguru import logger

# Load env
load_dotenv()

# ---- Boson / Higgs config ----
BOSON_API_KEY   = os.getenv("BOSON_API_KEY")
BOSON_BASE_URL  = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
HIGGS_EMO_MODEL = os.getenv("HIGGS_EMO_MODEL", "higgs-audio-understanding-Hackathon")

# OpenAI-compatible client (lazy)
_client = None
def _get_boson_client():
    global _client
    if _client is None:
        if not BOSON_API_KEY:
            raise RuntimeError("BOSON_API_KEY is not set. Put it in your .env.")
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package is required for Boson API. pip install openai") from e
        _client = OpenAI(api_key=BOSON_API_KEY, base_url=BOSON_BASE_URL)
        logger.info(f"[HIGGS-U] Using Boson endpoint: {BOSON_BASE_URL}")
    return _client

_TARGET_SR = 16000

@dataclass(frozen=True)
class EmotionScore:
    valence: float     # [-1, 1]
    arousal: float     # [-1, 1]
    label: str         # "happy"/"sad"/"angry"/"neutral"/"other"
    confidence: float  # [0,1]

# ---------------- utils ---------------- #

def _to_16k_pcm16_b64(y: np.ndarray, sr: int) -> tuple[str, str]:
    """
    Convert float32 mono array to 16 kHz PCM16 WAV base64 string.
    Returns (b64, "wav").
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != _TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_TARGET_SR)
        sr = _TARGET_SR
    y = np.clip(y, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64, "wav"

def _audio_cache_key(y: np.ndarray, sr: int) -> str:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    h = hashlib.sha1()
    h.update(y.tobytes()); h.update(str(sr).encode())
    return h.hexdigest()

# ---------------- low-level call ---------------- #

_SYSTEM_PROMPT = (
    "You are an emotion classifier for short speech clips. "
    "Return ONLY compact JSON with keys: "
    "{\"valence\": float(-1..1), \"arousal\": float(-1..1), "
    "\"label\": \"happy|sad|angry|neutral|other\", \"confidence\": float(0..1)}"
)

def _score_clip_boson(y: np.ndarray, sr: int, retry: int = 2) -> str:
    """
    Score one short audio clip via Boson with deterministic settings.
    Returns raw text (expected JSON or text containing JSON).
    """
    client = _get_boson_client()
    b64, fmt = _to_16k_pcm16_b64(y, sr)

    last_err = None
    for _ in range(max(1, retry + 1)):
        try:
            resp = client.chat.completions.create(
                model=HIGGS_EMO_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": [
                        {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                        {"type": "text", "text": "Analyze and return the JSON only."},
                    ]},
                ],
                modalities=["text", "audio"],
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_completion_tokens=512,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            logger.warning(f"[HIGGS-U] emotion scoring retry due to: {e}")
            time.sleep(0.5)
    raise RuntimeError(f"Boson emotion scoring failed after retries: {last_err}")

# ---------------- cached scoring wrapper ---------------- #

def score_emotion(y: np.ndarray, sr: int, force: bool = False) -> EmotionScore:
    """
    Score emotion of an audio clip (float32 mono). Uses memoization for identical buffers.
    Returns EmotionScore; on failure/parse-miss returns confidence==0.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)

    key = _audio_cache_key(y, sr)

    @lru_cache(maxsize=256)
    def _runner(cache_key: str) -> EmotionScore:
        try:
            txt = _score_clip_boson(y, sr, retry=2)
        except Exception as e:
            logger.error(f"[HIGGS-U] scoring failed: {e}")
            return EmotionScore(0.0, 0.0, "neutral", 0.0)

        # Parse tolerant JSON
        m = re.search(r"\{.*\}", txt or "", re.S)
        if not m:
            logger.warning(f"[HIGGS-U] Could not parse JSON from: {txt[:200] if txt else 'EMPTY'}")
            return EmotionScore(0.0, 0.0, "neutral", 0.0)
        try:
            j = json.loads(m.group(0))
            v   = max(-1.0, min(1.0, float(j.get("valence", 0.0))))
            a   = max(-1.0, min(1.0, float(j.get("arousal", 0.0))))
            lab = str(j.get("label", "neutral")).lower()
            cf  = max(0.0, min(1.0, float(j.get("confidence", 0.0))))
            return EmotionScore(v, a, lab, cf)
        except Exception as e:
            logger.warning(f"[HIGGS-U] JSON parse error: {e} from {txt[:200]}...")
            return EmotionScore(0.0, 0.0, "neutral", 0.0)

    if force:
        _runner.cache_clear()
    return _runner(key)
