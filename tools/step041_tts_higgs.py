# -*- coding: utf-8 -*-
"""
step041_tts_higgs.py
HIGGS/Boson TTS — simple, stable, accent-aware (per-line synthesis).

Env (.env):
  BOSON_API_KEY=...
  BOSON_BASE_URL=https://hackathon.boson.ai/v1
  HIGGS_TTS_MODEL=higgs-audio-generation-Hackathon
Optional:
  HIGGS_TTS_SPEED=1.0     # speaking rate hint (server may clamp/ignore)
  HIGGS_TTS_PAD_MS=8      # tiny pad at start/end (ms)
  HIGGS_TTS_ALLOW_MISMATCH=0  # if 0 and text looks ASCII-English but lang != en, force 'en'

Public API (dispatcher-compatible):
  init_TTS()
  load_model()
  tts(text, output_path, speaker_wav=None, *, voice_type=None, target_language=None)

Notes:
  - Speak EXACTLY the provided `text` (pipeline passes line['translation']).
  - Unified language codes expected: zh-cn, zh-tw, en, ko, ja, es, fr.
"""

from __future__ import annotations
import os, base64, wave, time, random
from typing import Optional, Dict

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# ------------------------------- Config ---------------------------------------

SR = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM
NCHANNELS = 1

_client: Optional[OpenAI] = None
_model_name: Optional[str] = None

# env knobs
_HIGGS_SPEED = float(os.getenv("HIGGS_TTS_SPEED") or 1.0)
_PAD_MS      = int(os.getenv("HIGGS_TTS_PAD_MS") or 8)
_ALLOW_MISMATCH = bool(int(os.getenv("HIGGS_TTS_ALLOW_MISMATCH") or "0"))

# ------------------------ Unified language normalization -----------------------

# Accept labels OR codes -> return canonical code
_LANG_ALIASES: Dict[str, str] = {
    # Simplified Chinese
    "zh-cn": "zh-cn", "zh_cn": "zh-cn", "cn": "zh-cn",
    "chinese (中文)": "zh-cn", "chinese": "zh-cn", "中文": "zh-cn",
    "simplified chinese (简体中文)": "zh-cn", "simplified chinese": "zh-cn", "简体中文": "zh-cn",

    # Traditional Chinese
    "zh-tw": "zh-tw", "zh_tw": "zh-tw", "tw": "zh-tw",
    "traditional chinese (繁体中文)": "zh-tw", "traditional chinese": "zh-tw", "繁体中文": "zh-tw",

    # English
    "en": "en", "english": "en",

    # Korean
    "ko": "ko", "korean": "ko", "한국어": "ko",

    # Japanese
    "ja": "ja", "japanese": "ja", "日本語": "ja",

    # Spanish
    "es": "es", "spanish": "es", "español": "es",

    # French
    "fr": "fr", "french": "fr", "français": "fr",
}

_ALLOWED_LANGS = {"zh-cn", "zh-tw", "en", "ko", "ja", "es", "fr"}

# Accent defaults by language code
DEFAULT_REGION: Dict[str, str] = {
    "en": "US",
    "zh-cn": "China",
    "zh-tw": "Taiwan",
    "ja": "Japan",
    "ko": "Korea",
    "fr": "France",
    "es": "Spain",
}

# ---------------------------- Initialization ----------------------------------

def init_TTS():
    load_model()

def load_model():
    global _client, _model_name
    if _client is not None:
        return
    load_dotenv()
    api_key  = os.getenv("BOSON_API_KEY", "").strip()
    base_url = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1").strip()
    _model_name = os.getenv("HIGGS_TTS_MODEL", "higgs-audio-generation-Hackathon").strip()
    if not api_key:
        raise RuntimeError("BOSON_API_KEY is not set.")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info(f"[HIGGS TTS] Client ready | base={base_url} | model={_model_name}")

# ------------------------------ Helpers ---------------------------------------

def _canon(s: Optional[str]) -> str:
    return "" if not s else str(s).strip().lower()

def _norm_lang(s: Optional[str]) -> str:
    key = _canon(s)
    code = _LANG_ALIASES.get(key, key)
    if code not in _ALLOWED_LANGS:
        # If empty, fallback to en; otherwise raise loudly to catch misconfig upstream
        if code == "":
            return "en"
        raise ValueError(f"[HIGGS TTS] Unsupported language: {s} -> {code}")
    return code

def _looks_ascii_english(text: str) -> bool:
    if not text:
        return False
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    # ASCII but not just punctuation/whitespace
    return any(c.isalpha() for c in text)

def _accent_from_voice_or_default(voice_type: Optional[str], lang_code: str) -> str:
    # Keep simple + deterministic; if you later encode region into voice_type, adapt here.
    return DEFAULT_REGION.get(lang_code, "US")

def _system_prompt(lang_code: str, region: str) -> str:
    # Keep the model on-task: speak-only, no paraphrase/translation/additions.
    # Use language code in prompt (server interprets code).
    return (
        f"Speak ONLY in {lang_code} with a native accent from {region}. "
        "Read the user's text verbatim; do NOT translate, paraphrase, or add words. "
        "Timing rules: treat commas as ~120ms pauses and sentence endings as ~220ms pauses. "
        "Do NOT read tags or metadata aloud. "
        "Keep natural prosody and native pronunciation. "
        "Maintain a consistent timbre, pitch, and speaking style across the entire utterance."
    )

def _b64_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _jittered_sleep(base: float, attempt: int):
    jitter = 0.2 + random.random() * 0.4
    time.sleep(base * (attempt + 1) * jitter)

# --------------------------- Streaming synthesis --------------------------------

def _stream_pcm16_to_wav(
    text: str,
    out_path: str,
    lang_code: str,
    region: str,
    ref_b64: Optional[str],
    max_retries: int = 3,
    backoff: float = 0.6,
):
    assert _client is not None and _model_name is not None

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sys_prompt = _system_prompt(lang_code, region)

    messages = [{"role": "system", "content": sys_prompt}]
    if ref_b64:
        messages.append({
            "role": "assistant",
            "content": [{"type": "input_audio", "input_audio": {"data": ref_b64, "format": "wav"}}],
        })
    messages.append({"role": "user", "content": text})

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(NCHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SR)

        # Leading pad
        if _PAD_MS > 0:
            wf.writeframes(b"\x00\x00" * int(SR * _PAD_MS / 1000.0))

        for attempt in range(max_retries + 1):
            try:
                stream = _client.chat.completions.create(
                    model=_model_name,
                    messages=messages,
                    modalities=["text", "audio"],
                    audio={"format": "pcm16"},
                    stream=True,
                    extra_body={"language": lang_code, "speed": float(_HIGGS_SPEED)},
                )
                got_audio = False
                for chunk in stream:
                    delta = getattr(chunk.choices[0], "delta", None)
                    audio = getattr(delta, "audio", None)
                    if not audio:
                        continue
                    wf.writeframes(base64.b64decode(audio["data"]))
                    got_audio = True

                # trailing pad
                if _PAD_MS > 0:
                    wf.writeframes(b"\x00\x00" * int(SR * _PAD_MS / 1000.0))

                if not got_audio:
                    wf.writeframes(b"\x00\x00" * int(0.1 * SR))  # brief silence fallback
                    logger.warning("[HIGGS TTS] No audio chunks received; wrote brief silence.")
                break
            except Exception as e:
                msg = str(e)
                logger.warning(f"[HIGGS TTS] stream attempt {attempt + 1} failed: {msg}")
                if attempt >= max_retries:
                    raise
                is_rate = ("429" in msg) or ("rate limit" in msg.lower())
                _jittered_sleep(backoff * (2.0 if is_rate else 1.0), attempt)

# ------------------------------- Public API ------------------------------------

def tts(
    text: str,
    output_path: str,
    speaker_wav: Optional[str] = None,
    *,
    voice_type: Optional[str] = None,
    target_language: Optional[str] = None,
) -> None:
    """
    Perform per-line synthesis and write a mono 16-bit PCM WAV at SR=24k.
    `target_language` can be a UI label or a canonical code; it will be normalized to a code.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        logger.info(f"[HIGGS TTS] Exists, skipping {output_path}")
        return

    load_model()

    # Normalize language to unified code
    lang_code = _norm_lang(target_language) if target_language else "en"

    # Safety: if text looks ASCII-English but non-English target, and mismatch not allowed -> force 'en'
    if not _ALLOW_MISMATCH and lang_code != "en" and _looks_ascii_english(text):
        logger.warning(f"[HIGGS TTS] ASCII-looking text with lang={lang_code}; forcing 'en'. "
                       f"Set HIGGS_TTS_ALLOW_MISMATCH=1 to disable.")
        lang_code = "en"

    region = _accent_from_voice_or_default(voice_type, lang_code)

    # Optional timbre reference
    ref_b64 = _b64_file(speaker_wav) if speaker_wav else None
    if ref_b64:
        logger.info(f"[HIGGS TTS] Using reference timbre: {speaker_wav}")

    # Empty text guard -> write a breath of silence
    text = (text or "").strip()
    if not text:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(NCHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SR)
            wf.writeframes(b"\x00\x00" * int(0.08 * SR))
        logger.warning("[HIGGS TTS] Empty input text; wrote brief silence.")
        return

    _stream_pcm16_to_wav(
        text=text,
        out_path=output_path,
        lang_code=lang_code,
        region=region,
        ref_b64=ref_b64,
        max_retries=3,
        backoff=0.6,
    )
    logger.info(f"[HIGGS TTS] Saved {output_path} | lang={lang_code}-{region} | speed={_HIGGS_SPEED} | pad_ms={_PAD_MS}")
