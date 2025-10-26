# -*- coding: utf-8 -*-
"""
step045_emotion.py
Post-TTS emotion control via lightweight DSP shaping + optional auto-tuning.

Public API:
    apply_emotion(wav, sr, preset='natural', strength=0.6, lang='en', sentence_times=None) -> np.ndarray
    auto_tune_emotion(wav, sr, target_preset='happy', strength=0.6, ...) -> (np.ndarray, dict)
"""

from __future__ import annotations
import time
from typing import List, Optional, Tuple

import numpy as np
import librosa
from loguru import logger
from scipy.signal import lfilter

from .step046_higgs_understanding import score_emotion, EmotionScore

# ------------------ PRESET DEFINITIONS ------------------ #
_PRESETS = {
    "natural":  dict(pitch_st=0.0,  rate=0.0,  shelf_db=0.0,  comp_ratio=1.0, pause_scale=1.0),
    "happy":    dict(pitch_st=+0.6, rate=+0.06, shelf_db=+1.5, comp_ratio=1.3, pause_scale=0.9),
    "sad":      dict(pitch_st=-0.6, rate=-0.07, shelf_db=-1.5, comp_ratio=1.0, pause_scale=1.1),
    "angry":    dict(pitch_st=+0.4, rate=+0.08, shelf_db=+0.5, comp_ratio=1.6, pause_scale=0.9),
}

# Safe limits
_MAX_PITCH_ST = 0.8     # semitones (global)
_MAX_RATE_FRAC = 0.10   # ±10%

# ------------------ HELPERS ------------------ #

def _soft_compress(y: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """Simple soft-knee compressor; ratio>1 tightens dynamics."""
    if ratio <= 1.0:
        return y.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(y**2) + 1e-8))
    gain = 1.0 / np.maximum(1.0, (np.abs(y) / (rms + 1e-8)) ** (ratio - 1.0))
    out = y * gain
    return np.asarray(out, dtype=np.float32)

def _high_shelf(y: np.ndarray, sr: int, gain_db: float = 0.0, cutoff: float = 3000.0) -> np.ndarray:
    """First-order high-shelf filter for brightness."""
    if abs(gain_db) < 1e-3:
        return y.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * cutoff / float(sr)
    alpha = np.sin(w0) / 2.0
    cosw0 = np.cos(w0)
    b0 =    A*((A+1)+(A-1)*cosw0+2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1)+(A+1)*cosw0)
    b2 =    A*((A+1)+(A-1)*cosw0-2*np.sqrt(A)*alpha)
    a0 =        (A+1)-(A-1)*cosw0+2*np.sqrt(A)*alpha
    a1 =  2*((A-1)-(A+1)*cosw0)
    a2 =        (A+1)-(A-1)*cosw0-2*np.sqrt(A)*alpha
    if abs(a0) < 1e-12:
        return y
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return lfilter(b, a, y).astype(np.float32)

def _stretch_pauses(
    y: np.ndarray,
    sr: int,
    sentence_times: Optional[List[Tuple[float, float]]],
    scale: float
) -> np.ndarray:
    """
    Scale inter-sentence pauses by 'scale' without touching voiced regions.
    Expects sentence_times as sorted [(start,end), ...] in seconds.
    """
    if sentence_times is None or len(sentence_times) == 0 or abs(scale - 1.0) < 1e-3:
        return y.astype(np.float32, copy=False)

    # Ensure sorted and clamped
    y = y.astype(np.float32, copy=False)
    n = len(y)
    sent = sorted(sentence_times, key=lambda x: x[0])
    sent = [(max(0.0, s), max(0.0, e)) for (s, e) in sent]

    out_chunks: List[np.ndarray] = []

    # Leading region before first sentence
    lead_start = 0
    lead_end = int(sent[0][0] * sr)
    lead_end = max(0, min(n, lead_end))
    if lead_end > lead_start:
        out_chunks.append(y[lead_start:lead_end])

    # For each sentence, copy voiced region, then scale following pause up to next start (or end of audio)
    for i, (s, e) in enumerate(sent):
        s_i = max(0, min(n, int(s * sr)))
        e_i = max(0, min(n, int(e * sr)))
        if e_i > s_i:
            out_chunks.append(y[s_i:e_i])

        # Determine pause region: end of this sentence -> start of next sentence or end of audio
        next_start = sent[i + 1][0] if i + 1 < len(sent) else (n / sr)
        p1 = e_i
        p2 = max(0, min(n, int(next_start * sr)))
        if p2 > p1:
            pause_seg = y[p1:p2]
            if len(pause_seg) > 16:
                try:
                    # scale>1.0 => longer pauses; implement by stretching (i.e., slower)
                    # librosa uses 'rate' (playback speed), so duration scales by 1/rate.
                    new_rate = 1.0 / float(scale)
                    pause_seg = librosa.effects.time_stretch(pause_seg, rate=new_rate)
                except Exception as ex:
                    logger.warning(f"[Emotion] pause time_stretch failed: {ex}")
            out_chunks.append(pause_seg)

    return np.concatenate(out_chunks) if out_chunks else y

# ------------------ MAIN ENTRY ------------------ #

def apply_emotion(
    wav: np.ndarray,
    sr: int,
    preset: str = "natural",
    strength: float = 0.6,
    lang: str = "en",
    sentence_times: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Apply emotional prosody shaping to a waveform."""
    preset = (preset or "natural").lower()
    if preset not in _PRESETS:
        logger.warning(f"[Emotion] Unknown preset '{preset}', defaulting to natural.")
        preset = "natural"
    if preset == "natural" or strength <= 0:
        return wav.astype(np.float32, copy=False)

    # Copy & scale parameters
    p = {k: v * strength for k, v in _PRESETS[preset].items()}

    # Tone-language guard (keep pitch small)
    if lang.lower().startswith("zh"):
        p["pitch_st"] = float(np.clip(p["pitch_st"], -0.3, 0.3))

    # Global safety clamps
    p["pitch_st"] = float(np.clip(p["pitch_st"], -_MAX_PITCH_ST, _MAX_PITCH_ST))
    p["rate"]     = float(np.clip(p["rate"],    -_MAX_RATE_FRAC, _MAX_RATE_FRAC))

    logger.info(
        f"[Emotion] Applying {preset} | pitch={p['pitch_st']:+.2f}st rate={p['rate']:+.2f} "
        f"shelf={p['shelf_db']:+.1f}dB comp={p['comp_ratio']:.2f} pause={p['pause_scale']:.2f}"
    )

    y = np.asarray(wav, dtype=np.float32)

    # 1) Small global pitch shift (final clamp before call)
    if abs(p["pitch_st"]) > 1e-3:
        try:
            n_steps = float(np.clip(p["pitch_st"], -_MAX_PITCH_ST, _MAX_PITCH_ST))
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)
        except Exception as e:
            logger.warning(f"[Emotion] pitch_shift failed: {e}")

    # 2) Slight rate change (tempo)
    if abs(p["rate"]) > 1e-3:
        try:
            rate = float(np.clip(1.0 + p["rate"], 0.85, 1.15))
            y = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
        except Exception as e:
            logger.warning(f"[Emotion] time_stretch failed: {e}")

    # 3) Pause shaping (only for modest scaling and when we have boundaries)
    if sentence_times and 0.8 <= float(p["pause_scale"]) <= 1.25:
        y = _stretch_pauses(y, sr, sentence_times, float(p["pause_scale"]))

    # 4) Dynamics
    y = _soft_compress(y, ratio=float(p["comp_ratio"]))

    # 5) Spectral tilt
    y = _high_shelf(y, sr, gain_db=float(p["shelf_db"]))

    # 6) Normalize safely once
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

# ---- VA targets for 4 presets (scaled by strength) ----
_VA_TARGETS = {
    "natural": (0.00, 0.00),
    "happy":   (+0.60, +0.50),
    "sad":     (-0.60, -0.50),
    "angry":   (+0.20, +0.60),
}

def _blend_va(base: Tuple[float, float], strength: float) -> Tuple[float, float]:
    return (base[0] * strength, base[1] * strength)

def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def auto_tune_emotion(
    wav: np.ndarray,
    sr: int,
    target_preset: str = "happy",
    strength: float = 0.6,
    lang: str = "en",
    sentence_times: Optional[List[Tuple[float, float]]] = None,
    latency_budget_s: float = 0.7,
    min_confidence: float = 0.50,
    max_iters: int = 2,
):
    """
    Closed-loop prosody control steered by Higgs-understanding.
    Returns (y_out, telemetry_dict)
    """
    t0 = time.time()
    preset = (target_preset or "natural").lower()
    if preset not in _VA_TARGETS:
        logger.warning(f"[EmotionAuto] Unknown preset '{preset}', defaulting to natural.")
        preset = "natural"

    # Score baseline
    base = score_emotion(wav, sr)
    base_va = (base.valence, base.arousal)
    target = _blend_va(_VA_TARGETS[preset], strength)
    best = {"y": wav, "score": base, "va": base_va, "preset": preset, "strength": strength}

    # If low confidence, do a single pass and bail.
    if base.confidence < min_confidence and preset != "natural":
        y1 = apply_emotion(wav, sr, preset=preset, strength=strength, lang=lang, sentence_times=sentence_times)
        s1 = score_emotion(y1, sr)
        return y1, {"base": base.__dict__, "final": s1.__dict__, "preset": preset, "strength": strength, "iters": 1}

    # Tiny coordinate search over strength
    iters = 0
    try_strengths = [strength, min(1.0, strength * 1.25), max(0.2, strength * 0.75)]
    for st in try_strengths:
        iters += 1
        y = apply_emotion(wav, sr, preset=preset, strength=st, lang=lang, sentence_times=sentence_times)
        sc = score_emotion(y, sr)
        va = (sc.valence, sc.arousal)
        if _dist2(va, target) < _dist2(best["va"], target):
            best = {"y": y, "score": sc, "va": va, "preset": preset, "strength": st}
        if iters >= max_iters or (time.time() - t0) > latency_budget_s:
            break

    return best["y"], {
        "base": base.__dict__,
        "final": best["score"].__dict__,
        "preset": preset,
        "requested_strength": strength,
        "applied_strength": best["strength"],
        "iters": iters,
        "target_va": {"valence": target[0], "arousal": target[1]},
        "elapsed_s": round(time.time() - t0, 3),
    }
