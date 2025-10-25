# -*- coding: utf-8 -*-
"""
Higgs/Boson ASR (OpenAI-compatible) with VAD-style chunking and verbatim transcription.

- Reads config from .env:
    BOSON_API_KEY=...
    BOSON_BASE_URL=https://hackathon.boson.ai/v1
    HIGGS_ASR_MODEL=higgs-audio-understanding-Hackathon

- Public API:
    higgs_transcribe_audio(wav_path, device='auto', batch_size=32, diarization=False, ...)

  Returns:
    List[{"start": float, "end": float, "text": str, "speaker": "SPEAKER_00"}]
  (Compatible with the structure used by WhisperX in your pipeline.)
"""
from __future__ import annotations

import io
import os
import time
import math
import base64
from typing import List, Tuple

import numpy as np
import librosa
import soundfile as sf  # librosa dependency; used to write wav buffers
from dotenv import load_dotenv
from loguru import logger

# Load env once
load_dotenv()

# ---- Boson / Higgs config ----
BOSON_API_KEY = os.getenv("BOSON_API_KEY")
BOSON_BASE_URL = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
HIGGS_ASR_MODEL = os.getenv("HIGGS_ASR_MODEL", "higgs-audio-understanding-Hackathon")

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
            raise RuntimeError(
                "openai package is required for Boson API. pip install openai"
            ) from e
        _client = OpenAI(api_key=BOSON_API_KEY, base_url=BOSON_BASE_URL)
        logger.info(f"[HIGGS] Using Boson endpoint: {BOSON_BASE_URL}")
    return _client


# -----------------------------
#           VAD
# -----------------------------
def _simple_energy_vad(y: np.ndarray, sr: int, top_db: float = 35.0) -> List[Tuple[int, int]]:
    """
    A light-weight voice-activity segmentation based on librosa.effects.split.
    Returns a list of (start_sample, end_sample) segments for voiced audio.
    """
    if y.ndim > 1:
        y = librosa.to_mono(y)
    # librosa.effects.split returns sample indices for non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    return [(int(s), int(e)) for s, e in intervals]


def _chunk_interval(max_chunk_s: float, sr: int, start: int, end: int) -> List[Tuple[int, int]]:
    """
    Split an interval [start, end) into ≤ max_chunk_s windows (in samples).
    """
    max_len = int(max_chunk_s * sr)
    length = end - start
    if length <= max_len:
        return [(start, end)]
    chunks = []
    n = int(math.ceil(length / max_len))
    for i in range(n):
        s = start + i * max_len
        e = min(start + (i + 1) * max_len, end)
        chunks.append((s, e))
    return chunks


def _wav_bytes_from_array(y: np.ndarray, sr: int) -> Tuple[bytes, str]:
    """
    Write mono float32 audio array to WAV bytes. Returns (wav_bytes, format='wav').
    """
    if y.ndim > 1:
        y = librosa.to_mono(y)
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return buf.read(), "wav"


def _b64_from_audio_array(y: np.ndarray, sr: int) -> Tuple[str, str]:
    data, fmt = _wav_bytes_from_array(y, sr)
    return base64.b64encode(data).decode("utf-8"), fmt


# -----------------------------
#      Boson ASR (verbatim)
# -----------------------------
_VERBATIM_SYSTEM = (
    "You are an automatic speech recognition engine.\n"
    "Transcribe the audio **verbatim**.\n"
    "Do not summarize, do not translate, do not add or omit words.\n"
    "Keep disfluencies and filler words. Only output the raw transcript text."
)

def _transcribe_clip_boson(y: np.ndarray, sr: int, retry: int = 2) -> str:
    """
    Transcribe one short audio clip (<= ~30 s) via Boson ASR with deterministic settings.
    """
    client = _get_boson_client()
    b64, fmt = _b64_from_audio_array(y, sr)

    last_err = None
    for _ in range(max(1, retry + 1)):
        try:
            resp = client.chat.completions.create(
                model=HIGGS_ASR_MODEL,  # critical: use configured ASR model
                messages=[
                    {"role": "system", "content": _VERBATIM_SYSTEM},
                    {"role": "user", "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": fmt},
                    }]},
                ],
                modalities=["text", "audio"],
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_completion_tokens=4096,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text
        except Exception as e:
            last_err = e
            logger.warning(f"[HIGGS] clip transcription retry due to: {e}")
            time.sleep(0.5)
    raise RuntimeError(f"Boson ASR failed after retries: {last_err}")


# -----------------------------
#     Public entry function
# -----------------------------
def higgs_transcribe_audio(
    wav_path: str,
    device: str = "auto",
    batch_size: int = 32,
    diarization: bool = False,
    min_speakers=None,
    max_speakers=None,
    target_sr: int = 16000,
    max_chunk_seconds: float = 25.0,
    vad_top_db: float = 35.0,
) -> List[dict]:
    """
    Verbatim ASR for a single audio file using Boson (Higgs) API.
    Output mirrors WhisperX transcript list shape used by your pipeline.

    Args ignored but kept for signature compatibility:
        device, batch_size, diarization, min_speakers, max_speakers

    Returns:
        [
          {"start": float, "end": float, "text": str, "speaker": "SPEAKER_00"},
          ...
        ]
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)

    if not BOSON_API_KEY:
        raise RuntimeError("BOSON_API_KEY is not set. Put it in your .env.")

    # Load & resample → 16 kHz mono
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    if y.size == 0:
        logger.warning(f"[HIGGS] Empty audio: {wav_path}")
        return []

    # 1) VAD split (non-silent intervals)
    voiced_intervals = _simple_energy_vad(y, target_sr, top_db=vad_top_db)
    if not voiced_intervals:
        # fallback: treat entire file as one chunk
        voiced_intervals = [(0, len(y))]

    # 2) Within each VAD region, split into <= max_chunk_seconds
    segments = []
    for s, e in voiced_intervals:
        for cs, ce in _chunk_interval(max_chunk_seconds, target_sr, s, e):
            segments.append((cs, ce))

    # 3) Transcribe each chunk deterministically with a verbatim prompt
    results: List[dict] = []
    for cs, ce in segments:
        clip = y[cs:ce]
        start_t = cs / float(target_sr)
        end_t = ce / float(target_sr)
        try:
            text = _transcribe_clip_boson(clip, target_sr)
        except Exception as e:
            logger.error(f"[HIGGS] Failed to transcribe chunk {start_t:.2f}-{end_t:.2f}s: {e}")
            text = ""

        # WhisperX-like item
        results.append({
            "start": round(start_t, 3),
            "end": round(end_t, 3),
            "text": text.strip(),
            "speaker": "SPEAKER_00",  # diarization is not provided by Boson; keep stable default
        })

    return results


# -----------------------------
#             CLI
# -----------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path:
        print("Usage: python step023_asr_higgs.py /path/to/audio.wav")
        raise SystemExit(2)

    logger.info(f"[HIGGS] Transcribing: {path}")
    out = higgs_transcribe_audio(path)
    for seg in out[:5]:
        logger.info(seg)
    logger.info(f"Segments: {len(out)}")
