# -*- coding: utf-8 -*-
"""
step041_tts_higgs.py
Higgs/Boson TTS — OpenAI-compatible, robust + fast (Boson chat schema)

Env (.env):
  BOSON_API_KEY=...
  BOSON_BASE_URL=https://hackathon.boson.ai/v1
  HIGGS_TTS_MODEL=higgs-audio-generation-Hackathon
  Optional:
    HIGGS_TTS_SPEED=1.0              # hint only; server may ignore
    HIGGS_TTS_MAX_WORKERS=4
    HIGGS_TTS_CHUNK_CHARS=280
    HIGGS_TTS_RETRIES=3
    HIGGS_TTS_BACKOFF=0.6
    HIGGS_TTS_XFADE_MS=28
    HIGGS_TTS_STREAM=0              # set to 1 to enable streaming PCM16

Public API (matches dispatcher):
    init_TTS()
    load_model()
    tts(text, output_path, speaker_wav, **kwargs)
"""

from __future__ import annotations
import os, io, re, time, base64, random
from functools import lru_cache
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

try:
    from .utils import save_wav as project_save_wav
except Exception:
    project_save_wav = None

try:
    import soundfile as sf
except Exception:
    sf = None

SR = 24000
EPS = 1e-8

# ------------------------------------------------------------------------------
# WAV helpers
def _save_wav_fallback(wav: np.ndarray, path: str, sr: int = SR) -> None:
    if sf is None:
        raise RuntimeError("soundfile is required: pip install soundfile")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, wav.astype(np.float32), sr)

def save_wav(wav: np.ndarray, path: str, sr: int = SR) -> None:
    if project_save_wav:
        project_save_wav(wav, path, sr)
    else:
        _save_wav_fallback(wav, path, sr)

# ------------------------------------------------------------------------------
# Environment / config
load_dotenv()
_BOSON_BASE_URL = (os.getenv("BOSON_BASE_URL") or "https://hackathon.boson.ai/v1").strip()
_BOSON_API_KEY  = (os.getenv("BOSON_API_KEY")  or "").strip()
_DEFAULT_MODEL  = (os.getenv("HIGGS_TTS_MODEL") or "higgs-audio-generation-Hackathon").strip()
_DEF_SPEED      = float(os.getenv("HIGGS_TTS_SPEED") or 1.0)   # hint only
_DEF_WORKERS    = max(1, int(os.getenv("HIGGS_TTS_MAX_WORKERS") or 4))
_DEF_CHARS      = max(120, int(os.getenv("HIGGS_TTS_CHUNK_CHARS") or 280))
_DEF_RETRIES    = max(1, int(os.getenv("HIGGS_TTS_RETRIES") or 3))
_DEF_BACKOFF    = float(os.getenv("HIGGS_TTS_BACKOFF") or 0.6)
_DEF_XF_MS      = int(os.getenv("HIGGS_TTS_XFADE_MS") or 28)
_USE_STREAM     = bool(int(os.getenv("HIGGS_TTS_STREAM", "0")))  # opt-in

_client: Optional[OpenAI] = None
_initialized = False

def init_TTS():
    load_model()

def load_model(model_path: Optional[str] = None, device: str = 'auto'):
    """Initialize OpenAI-compatible Higgs client (lazy singleton)."""
    global _client, _initialized
    if _initialized:
        return
    if not _BOSON_BASE_URL or not _BOSON_API_KEY:
        raise RuntimeError("Missing BOSON_BASE_URL or BOSON_API_KEY in .env")
    _client = OpenAI(base_url=_BOSON_BASE_URL, api_key=_BOSON_API_KEY)
    _initialized = True
    logger.info(f"[Higgs TTS] Connected to Boson/Higgs API @ {_BOSON_BASE_URL}")

# ------------------------------------------------------------------------------
# Language mapping
_language_map = {
    '中文': 'zh-cn', 'Chinese': 'zh-cn',
    'Korean': 'ko', '한국어': 'ko',
    'English': 'en',
    'Japanese': 'ja', '日本語': 'ja',
    'Tamil': 'ta', 'தமிழ்': 'ta',
    'Spanish': 'es', 'Español': 'es',
}
_LANG_NATIVE = {
    'zh-cn': 'Mandarin Chinese',
    'ko':    'Korean',
    'en':    'English',
    'ja':    'Japanese',
    'ta':    'Tamil',
    'es':    'Spanish',
}
_SUPPORTED = set(_language_map.values())

# Precompiled unicode ranges for fast detection
_RE_CJK   = re.compile(r'[\u4e00-\u9fff]')
_RE_HANG  = re.compile(r'[\uac00-\ud7af]')
_RE_KANA  = re.compile(r'[\u3040-\u30ff]')
_RE_TAMIL = re.compile(r'[\u0b80-\u0bff]')

@lru_cache(maxsize=1024)
def _map_lang(label: str) -> str:
    return _language_map.get(label, label).lower().strip()

def _guess_lang(text: str) -> str:
    """Fast language detection (CJK / Hangul / Kana / Tamil / else English)."""
    if _RE_CJK.search(text):   return 'zh-cn'
    if _RE_HANG.search(text):  return 'ko'
    if _RE_KANA.search(text):  return 'ja'
    if _RE_TAMIL.search(text): return 'ta'
    return 'en'

# ------------------------------------------------------------------------------
# System prompt for language lock
def _system_prompt(lang_code: str) -> str:
    native = _LANG_NATIVE.get(lang_code, "target language")
    return (
        f"Speak ONLY in {native} (code: {lang_code}). "
        "Read the user text verbatim with native pronunciation and prosody. "
        "Do NOT translate or paraphrase. Do NOT switch languages. "
        "If the input contains other-language words, read them using the target language phonotactics; "
        "do not output in another language. Never read bracketed hints aloud. "
        "Timing rules: treat commas as ~120ms pauses and sentence endings as ~220ms pauses. "
        "Avoid adding filler sounds or breaths. Keep a steady pace; do not elongate vowels unless written. "
        "Read numerals as written; do not expand or abbreviate."
    )

# ------------------------------------------------------------------------------
# Text split helpers (CJK-aware budgets)
_SENT_SPLIT = re.compile(r'(?<=[。！？!?…．\.，,；;：:])\s+|[\n\r]+')

def _split_text(text: str, chunk_chars: int, lang_code: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    budget = int(chunk_chars * (0.75 if lang_code in ("zh-cn", "ja", "ko") else 1.0))

    parts = re.split(_SENT_SPLIT, text)
    out, cur = [], ""
    for raw in parts:
        s = raw.strip()
        if not s:
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= budget:
            cur = f"{cur} {s}"
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    if len(out) >= 2 and len(out[-1]) < max(8, budget // 4):
        out[-2] = f"{out[-2]} {out[-1]}"; out.pop()
    return out

# ------------------------------------------------------------------------------
# Audio utilities
def _loudness_normalize(wav: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    if wav.size == 0:
        return wav
    rms = float(np.sqrt(np.mean(np.square(wav))) + EPS)
    gain = target_rms / rms
    return np.clip(wav * gain, -1.0, 1.0)

def _concat_xfade(chunks: List[np.ndarray], sr: int = SR, xf_ms: int = _DEF_XF_MS) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32, copy=False)
    xf = max(1, int(sr * xf_ms / 1000.0))
    out = chunks[0].astype(np.float32, copy=True)
    for seg in chunks[1:]:
        if seg is None or seg.size == 0:
            continue
        seg = seg.astype(np.float32, copy=False)
        n = min(xf, out.size, seg.size)
        if n > 0:
            fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
            mix = out[-n:] * (1.0 - fade) + seg[:n] * fade
            out = np.concatenate([out[:-n], mix, seg[n:]], dtype=np.float32)
        else:
            out = np.concatenate([out, seg], dtype=np.float32)
    return out

# ------------------------------------------------------------------------------
# Chat extractor (Boson schema)
def _extract_b64_from_chat(resp) -> Optional[str]:
    # Preferred: choices[0].message.audio.data
    try:
        data = resp.choices[0].message.audio.data
        if data:
            return data
    except Exception:
        pass
    # Fallback: content list blocks
    try:
        content = resp.choices[0].message.content
        if isinstance(content, list):
            for it in content:
                if isinstance(it, dict):
                    # output_audio style
                    if it.get("type") in ("output_audio", "audio"):
                        aud = it.get("audio") or it
                        data = aud.get("data")
                        if data:
                            return data
    except Exception:
        pass
    return None

# ------------------------------------------------------------------------------
# Core synthesis (non-streaming, WAV in base64 inside chat response)
def _synthesize_once(
    text: str,
    language: str,
    ref_b64: Optional[str],
    model_name: str,
    speed: float,
) -> np.ndarray:
    assert _client is not None, "Call init_TTS() first."

    sys_prompt = _system_prompt(language)

    # Boson expects: system -> (assistant with input_audio?) -> user (final text)
    messages: List[Dict] = [{"role": "system", "content": sys_prompt}]
    if ref_b64:
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": ref_b64,
                    "format": "wav"
                }
            }],
        })
    messages.append({"role": "user", "content": text})

    # NOTE: no explicit client timeout (let long generations finish)
    chat_resp = _client.chat.completions.create(
        model=model_name,
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={
            "language": language,
            "speed": float(max(0.5, min(2.0, speed))),
            "top_k": 50,
        },
    )

    b64 = _extract_b64_from_chat(chat_resp)
    if not b64:
        raise RuntimeError("No audio data in chat response.")
    wav_bytes = base64.b64decode(b64)

    # Decode WAV
    if not (len(wav_bytes) > 12 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"):
        raise RuntimeError("Server did not return WAV. Ensure WAV output is enabled.")

    data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False, dtype="float32")
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=1)
    wav = data.astype(np.float32, copy=False)

    # Resample to 24k if needed
    if sr != SR:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        except Exception:
            logger.warning(f"[Higgs TTS] Resample unavailable; keeping native SR={sr}")
            return _loudness_normalize(wav)
    return _loudness_normalize(wav)

# ------------------------------------------------------------------------------
# Core synthesis (STREAMING PCM16 @ 24kHz mono)
def _synthesize_once_stream_pcm16(
    text: str,
    language: str,
    ref_b64: Optional[str],
    model_name: str,
    speed: float,
) -> np.ndarray:
    """
    Streaming PCM16 path (24 kHz mono). Returns float32 [-1,1].
    """
    assert _client is not None, "Call init_TTS() first."

    sys_prompt = _system_prompt(language)

    messages: List[Dict] = [{"role": "system", "content": sys_prompt}]
    if ref_b64:
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": ref_b64,
                    "format": "wav"
                }
            }],
        })
    messages.append({"role": "user", "content": text})

    stream = _client.chat.completions.create(
        model=model_name,
        messages=messages,
        modalities=["text", "audio"],
        audio={"format": "pcm16"},  # request raw PCM16 chunks
        stream=True,
        max_completion_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={
            "language": language,
            "speed": float(max(0.5, min(2.0, speed))),
            "top_k": 50,
        },
    )

    # accumulate PCM16 frames
    chunks_i16: List[np.ndarray] = []
    for chunk in stream:
        delta = getattr(chunk.choices[0], "delta", None)
        if not delta:
            continue
        audio = getattr(delta, "audio", None)
        if not audio:
            continue
        b = base64.b64decode(audio["data"])
        if not b:
            continue
        # little-endian PCM16 mono @ 24kHz
        arr = np.frombuffer(b, dtype="<i2")
        if arr.size:
            chunks_i16.append(arr)

    if not chunks_i16:
        # fallback to brief silence if nothing streamed
        return np.zeros(int(0.1 * SR), dtype=np.float32)

    i16 = np.concatenate(chunks_i16)
    wav = (i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return _loudness_normalize(wav)

# ------------------------------------------------------------------------------
# Public entry (compatible with dispatcher)
def tts(text: str, output_path: str, speaker_wav: Optional[str], **kwargs) -> None:
    """
    Dispatcher-compatible entry:
        higgs_tts(text, out_path, speaker_wav, target_language='中文', ...)

    kwargs:
        model_name, speed, max_workers, chunk_chars, retries, retry_backoff, xfade_ms
        target_language   -> string label (mapped to 'zh-cn', 'en', ...)
        use_stream        -> bool (override env HIGGS_TTS_STREAM)
    """
    if os.path.exists(output_path):
        logger.info(f"[Higgs TTS] Exists, skipping {output_path}")
        return
    if not _initialized:
        load_model()

    model_name  = (kwargs.get("model_name") or _DEFAULT_MODEL).strip()
    speed       = float(kwargs.get("speed") or _DEF_SPEED)
    max_workers = max(1, int(kwargs.get("max_workers") or _DEF_WORKERS))
    chunk_chars = max(120, int(kwargs.get("chunk_chars") or _DEF_CHARS))
    retries     = max(1, int(kwargs.get("retries") or _DEF_RETRIES))
    retry_back  = float(kwargs.get("retry_backoff") or _DEF_BACKOFF)
    xfade_ms    = int(kwargs.get("xfade_ms") or _DEF_XF_MS)
    use_stream  = bool(kwargs.get("use_stream", _USE_STREAM))

    raw_lang = kwargs.get("target_language")
    if raw_lang:
        language = _map_lang(raw_lang)
    else:
        language = _guess_lang(text)
    if language not in _SUPPORTED:
        logger.warning(f"[Higgs TTS] Unsupported language '{language}', defaulting to English")
        language = 'en'

    # Read reference timbre once and pre-encode base64 (avoid per-chunk re-encoding)
    ref_b64 = None
    if speaker_wav and os.path.isfile(speaker_wav):
        try:
            with open(speaker_wav, "rb") as f:
                ref_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.warning(f"[Higgs TTS] Failed to read speaker wav: {e}")

    text = (text or "").strip()
    if not text:
        save_wav(np.zeros(int(0.1 * SR), dtype=np.float32), output_path)
        return

    segments = _split_text(text, chunk_chars, language)
    logger.debug(f"[Higgs TTS] {language} | chunks={len(segments)} | model={model_name} | stream={use_stream}")

    # Conservative parallelism for CJK to improve prosody/consistency
    effective_workers = min(max_workers, len(segments))
    if language in ("zh-cn", "ja", "ko"):
        effective_workers = 1

    synth_fn = _synthesize_once_stream_pcm16 if use_stream else _synthesize_once

    def _task(i_text):
        i, seg = i_text
        delay = 0.0
        for r in range(retries):
            try:
                if delay:
                    time.sleep(delay + random.random() * 0.15)  # jitter
                audio = synth_fn(seg, language, ref_b64, model_name, speed)
                return i, audio
            except Exception as e:
                logger.warning(f"[Higgs TTS] chunk {i+1}/{len(segments)} failed: {e}")
                delay = retry_back * (2 ** r)
        # keep timeline sane on total failure
        return i, np.zeros(int(0.2 * SR), dtype=np.float32)

    results: List[Tuple[int, np.ndarray]] = []
    if len(segments) == 1:
        # Fast path (no executor overhead)
        results = [(0, synth_fn(segments[0], language, ref_b64, model_name, speed))]
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as ex:
            futs = [ex.submit(_task, (i, s)) for i, s in enumerate(segments)]
            for fut in as_completed(futs):
                results.append(fut.result())

    ordered = [w for (_i, w) in sorted(results, key=lambda t: t[0])]
    joined = _concat_xfade(ordered, sr=SR, xf_ms=xfade_ms)
    if joined.size < int(0.1 * SR):
        joined = np.zeros(int(0.1 * SR), dtype=np.float32)

    save_wav(joined, output_path)
    logger.info(f"[Higgs TTS] Saved {output_path} | dur≈{joined.size/SR:.2f}s")
