# -*- coding: utf-8 -*-
"""
step045_emotion.py
Rate-SAFE, ultra-obvious DSP emotion shaping (no TTS prompt changes).
- Angry: lean low end, huge bite, bright, hard compression, gritty saturation, consonant snap, tense micro-jitter
- Happy: bright, sparkly, buoyant with parallel (upward) compression, mild grit
- Sad: darker/warmer, slower *feel* via pauses & HF roll-off, relaxed dynamics
- Speaking RATE is hard-limited to small changes (<= ±0.08), per request.

Public API (unchanged):
    apply_emotion(wav, sr, preset="angry", strength=0.85, lang="en",
                  sentence_times=None, exaggerate=True) -> np.ndarray
    auto_tune_emotion(wav, sr, target_preset="angry", strength=0.85, lang="en",
                      sentence_times=None, latency_budget_s=1.0, min_confidence=0.35,
                      max_iters=6, exaggerate=True)
"""

from __future__ import annotations
import time
from typing import List, Optional, Tuple

import numpy as np
import librosa
from loguru import logger
from scipy.signal import lfilter, butter

from .step046_higgs_understanding import score_emotion, EmotionScore

# ---------------------------------------------------------
# Strong targets with MINIMAL rate changes
# ---------------------------------------------------------
_BASE_PRESETS = {
    "neutral": dict(pitch_st=0.0,  rate= 0.00, shelf_db= 0.0, mid_db= 0.0, comp_ratio=1.2, pause_scale=1.00, drive=0.00),
    "happy":   dict(pitch_st=+1.8, rate=+0.06, shelf_db=+8.0, mid_db=+3.0, comp_ratio=2.2, pause_scale=0.92, drive=0.18),
    "sad":     dict(pitch_st=-1.8, rate=-0.05, shelf_db=-6.0, mid_db=-2.0, comp_ratio=1.3, pause_scale=1.40, drive=0.00),
    "angry":   dict(pitch_st=+2.4, rate=+0.05, shelf_db=+11.0, mid_db=+9.0, comp_ratio=8.0, pause_scale=0.82, drive=0.55),
}

# Hard clamps (keep rate small)
_LIMITS = {
    "neutral": dict(pitch_st=2.5, rate=0.08, shelf_db=12.0, mid_db=10.0, comp_ratio=8.0, drive=0.50),
    "happy":   dict(pitch_st=3.0, rate=0.08, shelf_db=12.0, mid_db=10.0, comp_ratio=8.0, drive=0.45),
    "sad":     dict(pitch_st=3.0, rate=0.08, shelf_db=12.0, mid_db=10.0, comp_ratio=6.0, drive=0.35),
    "angry":   dict(pitch_st=3.0, rate=0.08, shelf_db=13.5, mid_db=12.0, comp_ratio=12.0, drive=0.85),
}
_MAX_PITCH_ST_GLOBAL  = 3.0
_MAX_RATE_FRAC_GLOBAL = 0.08   # <= 8% speed change total

# Guidance targets (Higgs VA)
_VA_TARGETS = {
    "neutral": ( 0.00, 0.00),
    "happy":   (+0.60, +0.60),
    "sad":     (-0.60, -0.50),
    "angry":   (-0.40, +0.88),
}

# ---------- DSP helpers ----------
def _db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))

def _soft_compress(y: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if ratio <= 1.0: return y
    rms = float(np.sqrt(np.mean(y**2) + 1e-8))
    gain = 1.0 / np.maximum(1.0, (np.abs(y) / (rms + 1e-8)) ** (ratio - 1.0))
    return (y * gain).astype(np.float32)

def _parallel_compress(y: np.ndarray, ratio: float = 2.0, mix: float = 0.35) -> np.ndarray:
    """Upward(ish) compression via parallel mix of a compressed copy."""
    if ratio <= 1.0 or mix <= 1e-4: return y
    c = _soft_compress(y, ratio=ratio)
    m = float(np.clip(mix, 0.0, 0.9))
    out = (1.0 - m) * y + m * c
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def _limiter(y: np.ndarray, thr_db: float = -1.0) -> np.ndarray:
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

def _shelf(y: np.ndarray, sr: int, gain_db: float, cutoff: float, high: bool) -> np.ndarray:
    if abs(gain_db) < 1e-3: return y
    A = 10 ** (gain_db / 40.0)
    w0 = 2*np.pi*cutoff/float(sr)
    alpha = np.sin(w0)/2.0
    cosw0 = np.cos(w0)
    if high:
        b0 =    A*((A+1)+(A-1)*cosw0+2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1)+(A+1)*cosw0)
        b2 =    A*((A+1)+(A-1)*cosw0-2*np.sqrt(A)*alpha)
        a0 =        (A+1)-(A-1)*cosw0+2*np.sqrt(A)*alpha
        a1 =  2*((A-1)-(A+1)*cosw0)
        a2 =        (A+1)-(A-1)*cosw0-2*np.sqrt(A)*alpha
    else:
        b0 =    A*((A+1)-(A-1)*cosw0+2*np.sqrt(A)*alpha)
        b1 =  2*A*((A-1)-(A+1)*cosw0)
        b2 =    A*((A+1)-(A-1)*cosw0-2*np.sqrt(A)*alpha)
        a0 =        (A+1)+(A-1)*cosw0+2*np.sqrt(A)*alpha
        a1 = -2*((A-1)+(A+1)*cosw0)
        a2 =        (A+1)+(A-1)*cosw0-2*np.sqrt(A)*alpha
    if abs(a0) < 1e-12: return y
    b = np.array([b0,b1,b2],dtype=np.float64)/a0
    a = np.array([1.0,a1/a0,a2/a0],dtype=np.float64)
    return lfilter(b,a,y).astype(np.float32)

def _high_shelf(y, sr, gain_db, cutoff): return _shelf(y,sr,gain_db,cutoff,True)
def _low_shelf(y,  sr, gain_db, cutoff): return _shelf(y,sr,gain_db,cutoff,False)

def _hp(y: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    if cutoff <= 0.0: return y
    b, a = butter(order, cutoff / (0.5 * sr), btype='high', output='ba')
    return lfilter(b, a, y).astype(np.float32)

def _lp(y: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    if cutoff <= 0.0: return y
    b, a = butter(order, cutoff / (0.5 * sr), btype='low', output='ba')
    return lfilter(b, a, y).astype(np.float32)

def _de_ess(y: np.ndarray, sr: int, center: float = 7200.0, Q: float = 3.0, depth_db: float = -7.0) -> np.ndarray:
    return _peaking_eq(y, sr, gain_db=depth_db, f0=center, Q=Q)

def _transient_snap(y: np.ndarray, amount: float = 0.32) -> np.ndarray:
    if amount <= 1e-4: return y
    yy = np.abs(y) - librosa.effects.preemphasis(np.abs(y), coef=0.85)
    yy = np.clip(yy, 0.0, 1.0).astype(np.float32)
    mix = float(np.clip(amount, 0.0, 0.6))
    return np.clip((1.0 - mix) * y + mix * yy * np.sign(y), -1.0, 1.0).astype(np.float32)

def _micro_jitter(y: np.ndarray, sr: int, pitch_cents: float = 12.0, rate_ppm: float = 900.0) -> np.ndarray:
    if len(y) < sr//3: return y
    t = np.linspace(0, len(y)/sr, num=len(y), dtype=np.float32, endpoint=False)
    p_lfo = 2*np.pi*0.9*t
    r_lfo = 2*np.pi*0.7*t
    n_steps = (pitch_cents / 100.0) * np.sin(p_lfo)
    try:
        yp = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps.astype(np.float32))
    except Exception:
        yp = y
    rate = 1.0 + (rate_ppm / 1_000_000.0) * np.sin(r_lfo)
    try:
        idx = np.cumsum(rate).astype(np.float32)
        idx = (idx / idx[-1]) * (len(yp)-1)
        yj = np.interp(idx, np.arange(len(yp), dtype=np.float32), yp).astype(np.float32)
    except Exception:
        yj = yp
    return yj

def _stretch_pauses(y: np.ndarray, sr: int, sentence_times: Optional[List[Tuple[float, float]]], scale: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if not sentence_times or abs(scale-1.0) < 1e-3: return y
    n = len(y)
    sent = sorted([(max(0.0,s), max(0.0,e)) for (s,e) in sentence_times], key=lambda x:x[0])
    out: List[np.ndarray] = []
    lead_end = max(0, min(n, int(sent[0][0]*sr)))
    if lead_end > 0: out.append(y[:lead_end])
    for i,(s,e) in enumerate(sent):
        s_i = max(0, min(n, int(s*sr))); e_i = max(0, min(n, int(e*sr)))
        if e_i > s_i: out.append(y[s_i:e_i])
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

# ---------- Parameter calibration ----------
def _calibrate_params(preset: str, params: dict, lang: str) -> dict:
    lim = _LIMITS.get(preset, _LIMITS["neutral"])
    out = params.copy()

    # Mandarin pitch safety
    if lang.lower().startswith("zh"):
        out["pitch_st"] = float(np.clip(out["pitch_st"], -0.9, 0.9))

    def cap(v, lo, hi): return float(np.clip(v, lo, hi))
    req = dict(**out)
    out["pitch_st"]   = cap(out["pitch_st"], -lim["pitch_st"],  lim["pitch_st"])
    out["rate"]       = cap(out["rate"],     -lim["rate"],      lim["rate"])
    out["mid_db"]     = cap(out["mid_db"],   -lim["mid_db"],    lim["mid_db"])
    out["shelf_db"]   = cap(out["shelf_db"], -lim["shelf_db"],  lim["shelf_db"])
    out["comp_ratio"] = max(1.0, min(lim["comp_ratio"], float(out["comp_ratio"])))
    out["drive"]      = max(0.0, min(lim["drive"], float(out.get("drive", 0.0))))

    # global caps
    out["pitch_st"] = cap(out["pitch_st"], -_MAX_PITCH_ST_GLOBAL, _MAX_PITCH_ST_GLOBAL)
    out["rate"]     = cap(out["rate"],     -_MAX_RATE_FRAC_GLOBAL, _MAX_RATE_FRAC_GLOBAL)

    # clamp logs
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
    """
    Ultra-obvious pure-DSP shaping with SMALL rate adjustments.
    """
    p = (preset or "neutral").lower()
    if p not in _BASE_PRESETS:
        logger.warning(f"[Emotion] Unknown preset '{preset}', defaulting to neutral.")
        p = "neutral"
    if strength <= 0:
        return np.asarray(wav, dtype=np.float32)

    ex = 1.0
    if exaggerate:
        ex = 1.45 if p == "angry" else 1.25 if p == "happy" else 1.25 if p == "sad" else 1.05
    base = {k: (v * strength * ex if isinstance(v,(int,float)) else v) for k,v in _BASE_PRESETS[p].items()}
    params = _calibrate_params(p, base, lang)

    logger.info(
        f"[Emotion] {p}{' (EXAG)' if exaggerate else ''} | "
        f"pitch={params['pitch_st']:+.2f}st rate={params['rate']:+.2f} "
        f"shelf={params['shelf_db']:+.1f}dB mid={params['mid_db']:+.1f}dB "
        f"comp={params['comp_ratio']:.2f} pause={params['pause_scale']:.2f} drive={params['drive']:.2f}"
    )

    y = np.asarray(wav, dtype=np.float32)

    # Prosody (keep rate subtle)
    if abs(params["pitch_st"]) > 1e-3:
        try: y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(params["pitch_st"])).astype(np.float32)
        except Exception as e: logger.warning(f"[Emotion] pitch_shift failed: {e}")

    if abs(params["rate"]) > 1e-3:
        try: y = librosa.effects.time_stretch(y, rate=float(1.0 + params["rate"])).astype(np.float32)
        except Exception as e: logger.warning(f"[Emotion] time_stretch failed: {e}")

    if sentence_times:
        y = _stretch_pauses(y, sr, sentence_times, float(params["pause_scale"]))

    # Timbre/dynamics chains
    if p == "angry":
        # Thin warmth, add dual bite + bright tilt, control hiss, crush, grit, snap, tension
        y = _hp(y, sr, cutoff=200.0, order=2)
        y = _low_shelf(y, sr, gain_db=-3.0, cutoff=360.0)
        y = _peaking_eq(y, sr, gain_db=float(params["mid_db"]),      f0=2850.0, Q=0.9)
        y = _peaking_eq(y, sr, gain_db=float(params["mid_db"]*0.65), f0=4300.0, Q=1.0)
        y = _high_shelf(y, sr, gain_db=float(params["shelf_db"]),    cutoff=3800.0)
        y = _de_ess(y, sr, center=7200.0, Q=3.0, depth_db=-6.5)
        # compression → saturation → transient snap → micro-jitter
        y = _soft_compress(y, ratio=float(max(params["comp_ratio"], 7.0)))
        y = _saturate(y, drive=float(max(params["drive"], 0.55)))
        y = _transient_snap(y, amount=0.34)
        y = _micro_jitter(y, sr, pitch_cents=12.0, rate_ppm=800.0)

    elif p == "happy":
        # Buoyant brightness + presence + upward compression + mild grit
        y = _low_shelf(y, sr, gain_db=+2.0, cutoff=180.0)
        y = _peaking_eq(y, sr, gain_db=float(max(params["mid_db"], 2.5)), f0=2400.0, Q=1.1)
        y = _high_shelf(y, sr, gain_db=float(max(params["shelf_db"], 7.0)), cutoff=4200.0)
        y = _parallel_compress(y, ratio=float(max(params["comp_ratio"], 2.2)), mix=0.38)
        y = _saturate(y, drive=float(max(params["drive"], 0.16)))

    elif p == "sad":
        # Warmth + HF roll-off + relaxed dynamics (longer pauses already applied)
        y = _lp(y, sr, cutoff=7000.0, order=2)
        y = _high_shelf(y, sr, gain_db=float(min(params["shelf_db"], -6.0)), cutoff=3600.0)
        y = _peaking_eq(y, sr, gain_db=float(min(params["mid_db"], -1.5)), f0=1800.0, Q=1.1)
        y = _soft_compress(y, ratio=float(min(params["comp_ratio"], 1.6)))

    # Final safety
    y = _limiter(y, thr_db=-1.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

# ---------- Auto-tune with VA feedback (no rate escalation) ----------
def _angry_ok(v: float, a: float) -> bool: return (a >= 0.88) and (v <= -0.35)
def _happy_ok(v: float, a: float) -> bool: return (a >= 0.62) and (v >= +0.35)
def _sad_ok(v: float, a: float) -> bool:   return (v <= -0.50) and (a <= 0.25)

def auto_tune_emotion(
    wav: np.ndarray, sr: int, target_preset: str = "happy", strength: float = 0.85,
    lang: str = "en", sentence_times: Optional[List[Tuple[float, float]]] = None,
    latency_budget_s: float = 1.0, min_confidence: float = 0.35, max_iters: int = 6,
    exaggerate: bool = True
):
    """
    Escalates *non-rate* parameters until VA thresholds are met (rate stays clamped).
    """
    t0 = time.time()
    p = (target_preset or "neutral").lower()
    if p not in _BASE_PRESETS:
        logger.warning(f"[EmotionAuto] Unknown preset '{target_preset}', defaulting to neutral.")
        p = "neutral"

    def _ok(v, a):
        return _angry_ok(v,a) if p=="angry" else _happy_ok(v,a) if p=="happy" else _sad_ok(v,a) if p=="sad" else True

    best_y = wav
    best_sc = score_emotion(best_y, sr)

    # strong first pass
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
    shelf_cut_boost = 0.0  # for sad high cut

    while it < max_iters and (time.time() - t0) < latency_budget_s:
        it += 1
        v, a = best_sc.valence, best_sc.arousal
        if _ok(v,a) and best_sc.confidence >= min_confidence:
            break

        # Escalate WITHOUT touching rate
        if p == "angry":
            if a < 0.88:   # more arousal → brighter + tighter
                shelf_boost += 1.5; comp_boost += 0.8
            if v > -0.35:  # more negative valence → harsher bite + drive + low warmth cut
                bite_boost  += 1.8; drive_boost += 0.10
        elif p == "happy":
            if a < 0.62: shelf_boost += 1.2
            if v < 0.35: bite_boost  += 1.0; drive_boost += 0.05
        elif p == "sad":
            if a > 0.25: shelf_cut_boost += 1.5  # darker feel
            if v > -0.50: bite_boost -= 0.6      # soften presence

        # Re-run apply_emotion with slightly higher strength (still rate-clamped)
        local_strength = min(1.0, strength * (1.03 ** it))
        y_try = apply_emotion(best_y, sr, preset=p, strength=local_strength, lang=lang,
                              sentence_times=sentence_times, exaggerate=True)

        # Macro post-tweaks (no rate)
        if p == "angry":
            if bite_boost > 0:
                y_try = _peaking_eq(y_try, sr, gain_db=+min(4.0, bite_boost), f0=2950.0, Q=0.95)
                y_try = _peaking_eq(y_try, sr, gain_db=+min(3.0, bite_boost*0.7), f0=4300.0, Q=1.0)
            if shelf_boost > 0:
                y_try = _high_shelf(y_try, sr, gain_db=+min(4.0, shelf_boost), cutoff=4000.0)
            if drive_boost > 0:
                y_try = _saturate(y_try, drive=min(0.25, drive_boost))
            if comp_boost > 0:
                y_try = _soft_compress(y_try, ratio=1.0 + min(3.0, comp_boost))
            y_try = _limiter(y_try, thr_db=-1.0)

        elif p == "happy":
            if bite_boost > 0:
                y_try = _peaking_eq(y_try, sr, gain_db=+min(3.0, bite_boost), f0=2400.0, Q=1.0)
            if shelf_boost > 0:
                y_try = _high_shelf(y_try, sr, gain_db=+min(3.0, shelf_boost), cutoff=4200.0)
            if drive_boost > 0:
                y_try = _saturate(y_try, drive=min(0.12, drive_boost))
            y_try = _limiter(y_try, thr_db=-1.0)

        elif p == "sad":
            if shelf_cut_boost > 0:
                y_try = _high_shelf(y_try, sr, gain_db=-min(4.0, shelf_cut_boost), cutoff=3600.0)
                y_try = _lp(y_try, sr, cutoff=6800.0, order=2)
            if bite_boost < 0:
                y_try = _peaking_eq(y_try, sr, gain_db=max(-2.0, bite_boost), f0=2000.0, Q=1.1)
            y_try = _limiter(y_try, thr_db=-1.0)

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
