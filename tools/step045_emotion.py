# -*- coding: utf-8 -*-
"""
step045_emotion.py
Rate-SAFE, obvious DSP emotion shaping (no TTS prompt changes).

Additions:
- Loudness normalizer (LUFS-like, K-weighted) with gate & max gain
- Soft limiter after normalization to avoid clipping
"""

from __future__ import annotations
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import librosa
from loguru import logger
from scipy.signal import lfilter, butter, sosfiltfilt

from .step046_higgs_understanding import score_emotion, EmotionScore

# ---------------------------------------------------------
# Strong targets with MINIMAL rate changes
# ---------------------------------------------------------
_BASE_PRESETS = {
    "neutral": dict(pitch_st=0.0,  rate= 0.00, shelf_db= 0.0,  mid_db= 0.0,  comp_ratio=1.25, pause_scale=1.00, drive=0.00),
    "happy":   dict(pitch_st=+2.0, rate=+0.06, shelf_db=+10.0, mid_db=+4.0,  comp_ratio=2.6,  pause_scale=0.90, drive=0.22),
    "sad":     dict(pitch_st=-2.0, rate=-0.06, shelf_db=-9.0,  mid_db=-2.4, comp_ratio=1.45, pause_scale=1.50, drive=0.00),
    "angry":   dict(pitch_st=+2.6, rate=+0.06, shelf_db=+12.0, mid_db=+10.0, comp_ratio=9.0,  pause_scale=0.80, drive=0.60),
}

# Hard clamps
_LIMITS = {
    "neutral": dict(pitch_st=3.0, rate=0.08, shelf_db=12.0, mid_db=10.0, comp_ratio=9.0,  drive=0.55),
    "happy":   dict(pitch_st=3.0, rate=0.08, shelf_db=12.0, mid_db=10.0, comp_ratio=8.0,  drive=0.30),
    "sad":     dict(pitch_st=3.0, rate=0.08, shelf_db=12.0, mid_db=10.0, comp_ratio=1.8,  drive=0.20),
    "angry":   dict(pitch_st=3.0, rate=0.08, shelf_db=13.5, mid_db=12.0, comp_ratio=12.0, drive=0.70),
}
_MAX_PITCH_ST_GLOBAL  = 3.0
_MAX_RATE_FRAC_GLOBAL = 0.08

# Target LUFS (configurable via env); per-clip normalizer
_TARGET_LUFS = float(os.getenv("EMO_TARGET_LUFS", "-16.0"))
_MAX_MAKEUP_DB = float(os.getenv("EMO_MAX_MAKEUP_DB", "9.0"))
_GATE_LUFS = float(os.getenv("EMO_GATE_LUFS", "-45.0"))

# ---------- helpers ----------

def _db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))

def _lin_to_db(x: float) -> float:
    x = max(x, 1e-12)
    return 20.0 * np.log10(x)

def _soft_compress(y: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if ratio <= 1.0: return y
    rms = float(np.sqrt(np.mean(y**2) + 1e-8))
    gain = 1.0 / np.maximum(1.0, (np.abs(y) / (rms + 1e-8)) ** (ratio - 1.0))
    return (y * gain).astype(np.float32)

def _parallel_compress(y: np.ndarray, ratio: float = 2.0, mix: float = 0.35) -> np.ndarray:
    if ratio <= 1.0 or mix <= 1e-4: return y
    c = _soft_compress(y, ratio=ratio)
    m = float(np.clip(mix, 0.0, 0.9))
    out = (1.0 - m) * y + m * c
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def _limiter_soft(y: np.ndarray, thr_db: float = -1.5) -> np.ndarray:
    thr = _db_to_lin(thr_db)
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak <= thr: return y
    return (y / peak * thr).astype(np.float32)

def _saturate(y: np.ndarray, drive: float = 0.15) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if drive <= 1e-4: return y
    t = np.tanh(y * (1.0 + float(drive)))
    c = y - (y**3)/3.0
    out = 0.6*t + 0.4*c
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def _biquad_peak(sr: int, f0: float, Q: float, gain_db: float):
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / float(sr)
    alpha = np.sin(w0) / (2.0 * Q)
    cosw0 = np.cos(w0)
    b0 = 1 + alpha*A
    b1 = -2*cosw0
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*cosw0
    a2 = 1 - alpha/A
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a

def _peaking_eq(y: np.ndarray, sr: int, gain_db: float, f0: float, Q: float) -> np.ndarray:
    if abs(gain_db) < 1e-3: return y
    b, a = _biquad_peak(sr, f0=f0, Q=Q, gain_db=gain_db)
    return lfilter(b, a, y).astype(np.float32)

# ---- MISSING IN ORIGINAL: de-esser & pause stretcher ----

def _de_ess(y: np.ndarray, sr: int, center: float = 7200.0, Q: float = 3.0, depth_db: float = -7.0) -> np.ndarray:
    """Simple static de-esser via narrow peaking cut."""
    return _peaking_eq(y, sr, gain_db=depth_db, f0=center, Q=Q)

def _stretch_pauses(
    y: np.ndarray,
    sr: int,
    sentence_times: Optional[List[Tuple[float, float]]],
    scale: float
) -> np.ndarray:
    """
    Stretch the *pauses* between the provided sentence (start, end) spans.
    Keeps sentence audio intact; applies time-stretch (librosa) only to gaps.
    """
    y = np.asarray(y, dtype=np.float32)
    if not sentence_times or abs(scale-1.0) < 1e-3:
        return y
    n = len(y)
    sent = sorted([(max(0.0,s), max(0.0,e)) for (s,e) in sentence_times], key=lambda x:x[0])
    out: List[np.ndarray] = []
    lead_end = max(0, min(n, int(sent[0][0]*sr)))
    if lead_end > 0:
        out.append(y[:lead_end])
    for i,(s,e) in enumerate(sent):
        s_i = max(0, min(n, int(s*sr))); e_i = max(0, min(n, int(e*sr)))
        if e_i > s_i:
            out.append(y[s_i:e_i])
        nxt = sent[i+1][0] if i+1 < len(sent) else (n/ sr)
        p1, p2 = e_i, max(0, min(n, int(nxt*sr)))
        if p2 > p1:
            pause_seg = y[p1:p2]
            if len(pause_seg) > 16 and abs(scale-1.0) > 1e-3:
                try:
                    new_rate = float(np.clip(1.0/float(scale), 0.70, 1.30))
                    pause_seg = librosa.effects.time_stretch(pause_seg, rate=new_rate)
                except Exception as ex:
                    logger.warning(f"[Emotion] pause time_stretch failed: {ex}")
            out.append(pause_seg)
    try:
        return np.concatenate(out).astype(np.float32, copy=False)
    except Exception:
        return y

# zero-phase helpers
def _butter_sos(sr, cutoff, btype, order=4):
    nyq = 0.5 * float(sr)
    wc = float(cutoff) / max(1e-9, nyq)
    wc = np.clip(wc, 1e-6, 0.999999)
    return butter(order, wc, btype=btype, output='sos')

def _lp_zp(y: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    sos = _butter_sos(sr, cutoff, 'low', order=order)
    return sosfiltfilt(sos, y).astype(np.float32)

def _hp_zp(y: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    sos = _butter_sos(sr, cutoff, 'high', order=order)
    return sosfiltfilt(sos, y).astype(np.float32)

def _high_shelf_zp(y: np.ndarray, sr: int, gain_db: float, cutoff: float) -> np.ndarray:
    if abs(gain_db) < 1e-3: return y
    hp = _hp_zp(y, sr, cutoff, order=2)
    m = 1.0 + (gain_db / 18.0)
    m = float(np.clip(m, 0.0, 2.0))
    return np.clip((1.0 - 0.5*m)*y + (0.5*m)*hp, -1.0, 1.0).astype(np.float32)

def _low_shelf_zp(y: np.ndarray, sr: int, gain_db: float, cutoff: float) -> np.ndarray:
    if abs(gain_db) < 1e-3: return y
    lp = _lp_zp(y, sr, cutoff, order=2)
    m = 1.0 + (gain_db / 18.0)
    m = float(np.clip(m, 0.0, 2.0))
    return np.clip((1.0 - 0.5*m)*y + (0.5*m)*lp, -1.0, 1.0).astype(np.float32)

def _fade_edges(y: np.ndarray, sr: int, ms: float = 6.0) -> np.ndarray:
    if len(y) == 0: return y
    n = int(sr * (ms/1000.0))
    n = max(8, min(n, len(y)//4))
    if n <= 0: return y
    env = np.ones_like(y)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    env[:n] *= ramp
    env[-n:] *= ramp[::-1]
    return (y * env).astype(np.float32)

def _anti_alias(y: np.ndarray, sr: int) -> np.ndarray:
    cutoff = min(0.45*sr, 11000.0)
    return _lp_zp(y, sr, cutoff, order=4)

# --------- LUFS-like measurement & normalization ----------
def _k_weighting_sos(sr: int):
    # Very light approximation of ITU-R BS.1770 K-weighting
    return _butter_sos(sr, 40.0, 'high', order=2)

def _lufs_like(y: np.ndarray, sr: int) -> float:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0: return -np.inf
    sos = _k_weighting_sos(sr)
    yk = sosfiltfilt(sos, y).astype(np.float32)
    ms = float(np.mean(yk**2) + 1e-12)
    # Empirical offset to roughly map RMS to LUFS after K-weighting
    return float(-0.691 + 10.0*np.log10(ms))

def _loudness_normalize(y: np.ndarray, sr: int,
                        target_lufs: float = _TARGET_LUFS,
                        max_gain_db: float = _MAX_MAKEUP_DB,
                        gate_lufs: float = _GATE_LUFS) -> np.ndarray:
    cur = _lufs_like(y, sr)
    if not np.isfinite(cur) or cur < gate_lufs:
        # too quiet / essentially silence → skip to avoid boosting noise floor
        return y
    delta = target_lufs - cur
    delta = float(np.clip(delta, -0.1, max_gain_db))  # only upward makeup up to cap
    gain = _db_to_lin(delta)
    y2 = (y * gain).astype(np.float32)
    # safety limiter
    y2 = _limiter_soft(y2, thr_db=-1.5)
    return y2

# ---------- Parameter calibration ----------
def _calibrate_params(preset: str, params: dict, lang: str) -> dict:
    lim = _LIMITS.get(preset, _LIMITS["neutral"])
    out = params.copy()

    if lang.lower().startswith("zh"):
        out["pitch_st"] = float(np.clip(out["pitch_st"], -1.0, 1.0))

    def cap(v, lo, hi): return float(np.clip(v, lo, hi))
    req = dict(**out)
    out["pitch_st"]   = cap(out["pitch_st"], -lim["pitch_st"],  lim["pitch_st"])
    out["rate"]       = cap(out["rate"],     -lim["rate"],      lim["rate"])
    out["mid_db"]     = cap(out["mid_db"],   -lim["mid_db"],    lim["mid_db"])
    out["shelf_db"]   = cap(out["shelf_db"], -lim["shelf_db"],  lim["shelf_db"])
    out["comp_ratio"] = max(1.0, min(lim["comp_ratio"], float(out["comp_ratio"])))
    out["drive"]      = max(0.0, min(lim["drive"], float(out.get("drive", 0.0))))

    out["pitch_st"] = cap(out["pitch_st"], -_MAX_PITCH_ST_GLOBAL, _MAX_PITCH_ST_GLOBAL)
    out["rate"]     = cap(out["rate"],     -_MAX_RATE_FRAC_GLOBAL, _MAX_RATE_FRAC_GLOBAL)

    def log_clamp(name):
        if abs(req[name] - out[name]) > 1e-6:
            logger.debug(f"[Emotion] clamp {name}: {req[name]:+.2f} -> {out[name]:+.2f}")
    for k in ("pitch_st","rate","mid_db","shelf_db","comp_ratio","drive"):
        log_clamp(k)
    return out

# ---------- Main effect ----------
def apply_emotion(
    wav: np.ndarray,
    sr: int,
    preset: str = "neutral",
    strength: float = 0.85,
    lang: str = "en",
    sentence_times: Optional[List[Tuple[float, float]]] = None,
    exaggerate: bool = True,
) -> np.ndarray:
    p = (preset or "neutral").lower()
    if p not in _BASE_PRESETS:
        logger.warning(f"[Emotion] Unknown preset '{preset}', defaulting to neutral.")
        p = "neutral"
    if strength <= 0:
        return np.asarray(wav, dtype=np.float32)

    ex = 1.0
    if exaggerate:
        ex = 1.55 if p == "angry" else 1.35 if p == "happy" else 1.35 if p == "sad" else 1.05
    base = {k: (v * strength * ex if isinstance(v,(int,float)) else v) for k,v in _BASE_PRESETS[p].items()}
    params = _calibrate_params(p, base, lang)

    logger.info(
        f"[Emotion] {p}{' (EXAG)' if exaggerate else ''} | "
        f"pitch={params['pitch_st']:+.2f}st rate={params['rate']:+.2f} "
        f"shelf={params['shelf_db']:+.1f}dB mid={params['mid_db']:+.1f}dB "
        f"comp={params['comp_ratio']:.2f} pause={params['pause_scale']:.2f} drive={params['drive']:.2f}"
    )

    y = np.asarray(wav, dtype=np.float32)

    # Prosody (small)
    if abs(params["pitch_st"]) > 1e-3:
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(params["pitch_st"])).astype(np.float32)
        except Exception as e:
            logger.warning(f"[Emotion] pitch_shift failed: {e}")

    if abs(params["rate"]) > 1e-3:
        try:
            y = librosa.effects.time_stretch(y, rate=float(1.0 + params["rate"])).astype(np.float32)
        except Exception as e:
            logger.warning(f"[Emotion] time_stretch failed: {e}")

    if sentence_times:
        y = _stretch_pauses(y, sr, sentence_times, float(params["pause_scale"]))

    # Anti-alias after time/pitch
    y = _anti_alias(y, sr)

    # Timbre/dynamics
    if p == "angry":
        y = _hp_zp(y, sr, cutoff=240.0, order=2)
        y = _low_shelf_zp(y, sr, gain_db=-3.0, cutoff=380.0)
        y = _peaking_eq(y, sr, gain_db=float(params["mid_db"]),      f0=2850.0, Q=0.9)
        y = _peaking_eq(y, sr, gain_db=float(params["mid_db"]*0.65), f0=4300.0, Q=1.0)
        y = _high_shelf_zp(y, sr, gain_db=float(params["shelf_db"]), cutoff=3800.0)
        y = _de_ess(y, sr, center=7300.0, Q=3.0, depth_db=-6.5)
        y = _soft_compress(y, ratio=float(max(params["comp_ratio"], 9.0)))
        y = _saturate(y, drive=float(min(max(params["drive"], 0.58), 0.70)))
        y = _parallel_compress(y, ratio=1.6, mix=0.18)
    elif p == "happy":
        y = _low_shelf_zp(y, sr, gain_db=+2.0, cutoff=170.0)
        y = _peaking_eq(y, sr, gain_db=float(max(params["mid_db"], 3.2)), f0=2400.0, Q=1.05)
        y = _high_shelf_zp(y, sr, gain_db=float(max(params["shelf_db"], 9.0)), cutoff=4200.0)
        y = _parallel_compress(y, ratio=float(max(params["comp_ratio"], 2.6)), mix=0.40)
        y = _saturate(y, drive=float(min(max(params["drive"], 0.20), 0.26)))
    elif p == "sad":
        y = _lp_zp(y, sr, cutoff=6000.0, order=4)
        y = _high_shelf_zp(y, sr, gain_db=float(min(params["shelf_db"], -9.0)), cutoff=3600.0)
        y = _peaking_eq(y, sr, gain_db=float(min(params["mid_db"], -2.2)), f0=1700.0, Q=1.15)
        y = _soft_compress(y, ratio=float(min(params["comp_ratio"], 1.45)))

    # Edge de-click + AA
    y = _fade_edges(y, sr, ms=6.0)
    y = _anti_alias(y, sr)

    # Loudness makeup (pre-limiter)
    pre_lufs = _lufs_like(y, sr)
    y = _loudness_normalize(y, sr, target_lufs=_TARGET_LUFS,
                            max_gain_db=_MAX_MAKEUP_DB, gate_lufs=_GATE_LUFS)
    post_lufs = _lufs_like(y, sr)
    logger.debug(f"[Emotion] Loudness {pre_lufs:.1f} LUFS → {post_lufs:.1f} LUFS (target {_TARGET_LUFS:.1f})")

    # Final safety
    y = _limiter_soft(y, thr_db=-1.5)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

# ---------- Auto-tune w/ VA feedback ----------
def _angry_ok(v: float, a: float) -> bool: return (a >= 0.92) and (v <= -0.40)
def _happy_ok(v: float, a: float) -> bool: return (a >= 0.65) and (v >= +0.40)
def _sad_ok(v: float, a: float) -> bool:   return (v <= -0.60) and (a <= 0.25)

def auto_tune_emotion(
    wav: np.ndarray, sr: int, target_preset: str = "happy", strength: float = 0.85,
    lang: str = "en", sentence_times: Optional[List[Tuple[float, float]]] = None,
    latency_budget_s: float = 1.0, min_confidence: float = 0.35, max_iters: int = 6,
    exaggerate: bool = True
):
    t0 = time.time()
    p = (target_preset or "neutral").lower()
    if p not in _BASE_PRESETS:
        logger.warning(f"[EmotionAuto] Unknown preset '{target_preset}', defaulting to neutral.")
        p = "neutral"

    def _ok(v, a):
        return _angry_ok(v,a) if p=="angry" else _happy_ok(v,a) if p=="happy" else _sad_ok(v,a) if p=="sad" else True

    best_y = wav
    best_sc = score_emotion(best_y, sr)

    cur_y = apply_emotion(best_y, sr, preset=p, strength=strength, lang=lang,
                          sentence_times=sentence_times, exaggerate=exaggerate)
    cur_sc = score_emotion(cur_y, sr)
    if cur_sc.confidence >= best_sc.confidence or _ok(cur_sc.valence, cur_sc.arousal):
        best_y, best_sc = cur_y, cur_sc

    it = 1
    bite_boost = 0.0
    shelf_boost = 0.0
    drive_boost = 0.0
    comp_boost  = 0.0
    shelf_cut_boost = 0.0

    while it < max_iters and (time.time() - t0) < latency_budget_s:
        it += 1
        v, a = best_sc.valence, best_sc.arousal
        if _ok(v,a) and best_sc.confidence >= min_confidence:
            break

        if p == "angry":
            if a < 0.92:
                shelf_boost += 1.6; comp_boost += 0.9
            if v > -0.40:
                bite_boost  += 2.0; drive_boost += 0.10
        elif p == "happy":
            if a < 0.65: shelf_boost += 1.3
            if v < 0.40: bite_boost  += 1.1; drive_boost += 0.05
        elif p == "sad":
            if a > 0.25: shelf_cut_boost += 1.6
            if v > -0.60: bite_boost -= 0.7

        local_strength = min(1.0, strength * (1.035 ** it))
        y_try = apply_emotion(best_y, sr, preset=p, strength=local_strength, lang=lang,
                              sentence_times=sentence_times, exaggerate=True)

        if p == "angry":
            if bite_boost > 0:
                y_try = _peaking_eq(y_try, sr, gain_db=+min(4.5, bite_boost), f0=2950.0, Q=0.95)
                y_try = _peaking_eq(y_try, sr, gain_db=+min(3.2, bite_boost*0.7), f0=4300.0, Q=1.0)
            if shelf_boost > 0:
                y_try = _high_shelf_zp(y_try, sr, gain_db=+min(4.0, shelf_boost), cutoff=4000.0)
            if drive_boost > 0:
                y_try = _saturate(y_try, drive=min(0.25, drive_boost))
            if comp_boost > 0:
                y_try = _soft_compress(y_try, ratio=1.0 + min(3.2, comp_boost))
            y_try = _limiter_soft(y_try, thr_db=-1.5)

        elif p == "happy":
            if bite_boost > 0:
                y_try = _peaking_eq(y_try, sr, gain_db=+min(3.2, bite_boost), f0=2400.0, Q=1.0)
            if shelf_boost > 0:
                y_try = _high_shelf_zp(y_try, sr, gain_db=+min(3.2, shelf_boost), cutoff=4200.0)
            if drive_boost > 0:
                y_try = _saturate(y_try, drive=min(0.14, drive_boost))
            y_try = _limiter_soft(y_try, thr_db=-1.5)

        elif p == "sad":
            if shelf_cut_boost > 0:
                y_try = _high_shelf_zp(y_try, sr, gain_db=-min(4.5, shelf_cut_boost), cutoff=3600.0)
                y_try = _lp_zp(y_try, sr, cutoff=6800.0, order=4)
            if bite_boost < 0:
                y_try = _peaking_eq(y_try, sr, gain_db=max(-2.2, bite_boost), f0=2000.0, Q=1.1)
            y_try = _limiter_soft(y_try, thr_db=-1.5)

        sc_try = score_emotion(y_try, sr)
        better = (sc_try.confidence > best_sc.confidence) or (_ok(sc_try.valence, sc_try.arousal) and not _ok(best_sc.valence, best_sc.arousal))
        if better:
            best_y, best_sc = y_try, sc_try

    meta = {
        "final": dict(valence=best_sc.valence, arousal=best_sc.arousal,
                      label=best_sc.label, confidence=best_sc.confidence),
        "preset": p, "strength": strength, "iters": it,
        "exaggerate": exaggerate,
    }
    return best_y, meta
