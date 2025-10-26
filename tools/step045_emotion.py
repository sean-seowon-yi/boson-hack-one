# -*- coding: utf-8 -*-
"""
step045_emotion.py
MAX-OBVIOUS DSP emotion shaping (no TTS changes).
- NUCLEAR mode: unmistakable angry/happy/sad timbres & prosody
- Aggressive EQ signatures, multi-stage compression, saturation, transient snap
- Pause reshaping (angry: shorter, happy: slightly shorter, sad: longer)
- Higgs-understanding feedback with escalation until clear VA is reached
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
# Extreme base targets (you asked to push it hard)
# ---------------------------------------------------------
_BASE_PRESETS = {
    "neutral": dict(pitch_st=0.0,  rate=0.00, shelf_db=0.0,  mid_db=0.0,  comp_ratio=1.2, pause_scale=1.00, drive=0.00),
    "happy":   dict(pitch_st=+2.0, rate=+0.55, shelf_db=+8.0, mid_db=+4.0, comp_ratio=2.0, pause_scale=0.88, drive=0.15),
    "sad":     dict(pitch_st=-2.0, rate=-0.35, shelf_db=-8.0, mid_db=-3.0, comp_ratio=1.4, pause_scale=1.35, drive=0.00),
    "angry":   dict(pitch_st=+3.0, rate=+0.60, shelf_db=+12.0, mid_db=+10.0, comp_ratio=9.0, pause_scale=0.70, drive=0.60),
}

# Hard safety clamps
_LIMITS = {
    "neutral": dict(pitch_st=2.5, rate=0.60, shelf_db=12.0, mid_db=10.0, comp_ratio=10.0, drive=0.60),
    "happy":   dict(pitch_st=3.0, rate=0.60, shelf_db=12.0, mid_db=10.0, comp_ratio=10.0, drive=0.60),
    "sad":     dict(pitch_st=3.0, rate=0.60, shelf_db=12.0, mid_db=10.0, comp_ratio=10.0, drive=0.60),
    "angry":   dict(pitch_st=3.0, rate=0.60, shelf_db=14.0, mid_db=12.0, comp_ratio=12.0, drive=0.85),
}
_MAX_PITCH_ST_GLOBAL  = 3.0
_MAX_RATE_FRAC_GLOBAL = 0.60

# Guidance targets (Higgs VA)
_VA_TARGETS = {
    "neutral": ( 0.00, 0.00),
    "happy":   (+0.60, +0.60),
    "sad":     (-0.60, -0.60),
    "angry":   (-0.40, +0.90),
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

def _limiter(y: np.ndarray, thr_db: float = -1.0) -> np.ndarray:
    thr = _db_to_lin(thr_db)
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak <= thr: return y
    return (y / peak * thr).astype(np.float32)

def _saturate(y: np.ndarray, drive: float = 0.15) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if drive <= 1e-4: return y
    # odd-harmonic emphasis: mix tanh and cubic for aggressive grit
    t = np.tanh(y * (1.0 + drive))
    c = y - (y**3)/3.0
    out = 0.65*t + 0.35*c
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

def _transient_snap(y: np.ndarray, amount: float = 0.35) -> np.ndarray:
    if amount <= 1e-4: return y
    # crude enhancer using preemphasis of rectified signal
    yy = np.abs(y) - librosa.effects.preemphasis(np.abs(y), coef=0.85)
    yy = np.clip(yy, 0.0, 1.0).astype(np.float32)
    mix = float(np.clip(amount, 0.0, 0.6))
    return np.clip((1.0 - mix) * y + mix * yy * np.sign(y), -1.0, 1.0).astype(np.float32)

def _micro_jitter(y: np.ndarray, sr: int, pitch_cents: float = 14.0, rate_ppm: float = 1100.0) -> np.ndarray:
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
                    new_rate = float(np.clip(1.0/float(scale), 0.55, 1.45))
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

    # per-emotion caps
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
    strength: float = 0.9,
    lang: str = "en",
    sentence_times: Optional[List[Tuple[float, float]]] = None,
    exaggerate: bool = True,
    nuclear: bool = True,   # NEW: go all-in by default
) -> np.ndarray:
    """
    MAX-OBVIOUS pure-DSP shaping. 'nuclear' unlocks the most aggressive chain.
    """
    p = (preset or "neutral").lower()
    if p not in _BASE_PRESETS:
        logger.warning(f"[Emotion] Unknown preset '{preset}', defaulting to neutral.")
        p = "neutral"
    if strength <= 0:
        return np.asarray(wav, dtype=np.float32)

    # scale
    ex = 1.0
    if exaggerate:
        ex = 1.50 if p == "angry" else 1.30 if p == "happy" else 1.25 if p == "sad" else 1.05
    base = {k: (v * strength * ex if isinstance(v,(int,float)) else v) for k,v in _BASE_PRESETS[p].items()}
    params = _calibrate_params(p, base, lang)

    logger.info(
        f"[Emotion] {p}{' (EXAG)' if exaggerate else ''}{' [NUCLEAR]' if nuclear else ''} | "
        f"pitch={params['pitch_st']:+.2f}st rate={params['rate']:+.2f} "
        f"shelf={params['shelf_db']:+.1f}dB mid={params['mid_db']:+.1f}dB "
        f"comp={params['comp_ratio']:.2f} pause={params['pause_scale']:.2f} drive={params['drive']:.2f}"
    )

    y = np.asarray(wav, dtype=np.float32)

    # Prosody first
    if abs(params["pitch_st"]) > 1e-3:
        try: y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(params["pitch_st"])).astype(np.float32)
        except Exception as e: logger.warning(f"[Emotion] pitch_shift failed: {e}")

    if abs(params["rate"]) > 1e-3:
        try: y = librosa.effects.time_stretch(y, rate=float(1.0 + params["rate"])).astype(np.float32)
        except Exception as e: logger.warning(f"[Emotion] time_stretch failed: {e}")

    if sentence_times:
        scale = float(params["pause_scale"])
        y = _stretch_pauses(y, sr, sentence_times, scale)

    # Timbre chains (NUCLEAR)
    if p == "angry":
        # "angry signature": thin lows, huge bite, bright, de-ess, crush, grit, snap, wobble
        y = _hp(y, sr, cutoff=220.0 if nuclear else 170.0, order=2)
        y = _low_shelf(y, sr, gain_db=-4.0 if nuclear else -2.0, cutoff=350.0)
        y = _peaking_eq(y, sr, gain_db=float(params["mid_db"]),      f0=3000.0, Q=0.85)
        y = _peaking_eq(y, sr, gain_db=float(params["mid_db"]*0.70), f0=4600.0, Q=0.95)
        y = _high_shelf(y, sr, gain_db=float(params["shelf_db"]),    cutoff=3800.0)
        y = _de_ess(y, sr, center=7400.0, Q=3.2, depth_db=-8.0 if nuclear else -6.0)
        y = _soft_compress(y, ratio=float(max(params["comp_ratio"], 8.0)))
        y = _saturate(y, drive=float(max(params["drive"], 0.65 if nuclear else params["drive"])))
        y = _transient_snap(y, amount=0.40 if nuclear else 0.30)
        if nuclear:
            y = _micro_jitter(y, sr, pitch_cents=16.0, rate_ppm=1300.0)

    elif p == "happy":
        # bright, lively, sparkly
        y = _low_shelf(y, sr, gain_db=+2.0, cutoff=180.0)
        y = _peaking_eq(y, sr, gain_db=float(max(params["mid_db"], 3.0)), f0=2400.0, Q=1.0)
        y = _high_shelf(y, sr, gain_db=float(max(params["shelf_db"], 8.0 if nuclear else params["shelf_db"])), cutoff=4200.0)
        y = _soft_compress(y, ratio=float(max(params["comp_ratio"], 2.4)))
        y = _saturate(y, drive=float(max(params["drive"], 0.20 if nuclear else params["drive"])))
        if nuclear:
            y = _transient_snap(y, amount=0.25)

    elif p == "sad":
        # dark, slow, soft top
        y = _lp(y, sr, cutoff=7200.0 if nuclear else 8200.0, order=2)
        y = _high_shelf(y, sr, gain_db=float(min(params["shelf_db"], -8.0)), cutoff=3800.0)
        y = _peaking_eq(y, sr, gain_db=float(min(params["mid_db"], -2.0)), f0=1800.0, Q=1.1)
        y = _soft_compress(y, ratio=float(min(params["comp_ratio"], 1.6)))

    # Final loudness guard
    y = _limiter(y, thr_db=-1.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

# ---------- Escalation tuner (Higgs-understanding VA) ----------

def _angry_ok(v: float, a: float) -> bool: return (a >= 0.90) and (v <= -0.35)
def _happy_ok(v: float, a: float) -> bool: return (a >= 0.65) and (v >= +0.35)
def _sad_ok(v: float, a: float) -> bool:   return (a <= -0.50) and (a <= 0.20)

def auto_tune_emotion(
    wav: np.ndarray, sr: int, target_preset: str = "happy", strength: float = 0.9,
    lang: str = "en", sentence_times: Optional[List[Tuple[float, float]]] = None,
    latency_budget_s: float = 1.2, min_confidence: float = 0.35, max_iters: int = 8,
    exaggerate: bool = True, nuclear: bool = True
):
    """
    Repeat: apply_emotion -> score(Higgs) -> escalate until VA thresholds are met
    or we hit time/iteration budget. Returns (y, meta).
    """
    t0 = time.time()
    p = (target_preset or "neutral").lower()
    if p not in _BASE_PRESETS:
        logger.warning(f"[EmotionAuto] Unknown preset '{target_preset}', defaulting to neutral.")
        p = "neutral"

    def _ok(v, a):
        return _angry_ok(v,a) if p=="angry" else _happy_ok(v,a) if p=="happy" else _sad_ok(v,a) if p=="sad" else True

    # base score
    best_y = wav
    best_sc = score_emotion(best_y, sr)
    v, a = best_sc.valence, best_sc.arousal
    logger.debug(f"[EmotionAuto] base v={v:+.2f} a={a:+.2f} conf={best_sc.confidence:.2f} target={p}")

    # strong first pass
    cur_y = apply_emotion(best_y, sr, preset=p, strength=strength, lang=lang,
                          sentence_times=sentence_times, exaggerate=exaggerate, nuclear=nuclear)
    cur_sc = score_emotion(cur_y, sr)
    if cur_sc.confidence >= best_sc.confidence or _ok(cur_sc.valence, cur_sc.arousal):
        best_y, best_sc = cur_y, cur_sc

    it = 1
    while it < max_iters and (time.time() - t0) < latency_budget_s:
        it += 1
        v, a = best_sc.valence, best_sc.arousal
        if _ok(v,a) and best_sc.confidence >= min_confidence:
            break

        # escalate knobs by reapplying with higher strength (capped inside)
        local_strength = min(1.0, strength * (1.05 ** it))
        cur_y = apply_emotion(best_y, sr, preset=p, strength=local_strength, lang=lang,
                              sentence_times=sentence_times, exaggerate=True, nuclear=True)

        # macro nudges if still not clear
        if p == "angry":
            if a < 0.90:   # more arousal
                cur_y = _high_shelf(cur_y, sr, gain_db=+2.0, cutoff=4200.0)
                cur_y = _soft_compress(cur_y, ratio=1.4)
            if v > -0.35:  # push negative valence
                cur_y = _low_shelf(cur_y, sr, gain_db=-2.5, cutoff=400.0)
                cur_y = _saturate(cur_y, drive=0.12)
        elif p == "happy":
            if a < 0.65:
                cur_y = _high_shelf(cur_y, sr, gain_db=+2.5, cutoff=4300.0)
            if v < 0.35:
                cur_y = _peaking_eq(cur_y, sr, gain_db=+1.5, f0=2400.0, Q=1.0)
        elif p == "sad":
            if a > 0.20:
                # slow feel by blurring highs slightly more
                cur_y = _lp(cur_y, sr, cutoff=6800.0, order=2)
            if v > -0.50:
                cur_y = _high_shelf(cur_y, sr, gain_db=-2.0, cutoff=3600.0)

        cur_y = _limiter(cur_y, thr_db=-1.0)
        cur_sc = score_emotion(cur_y, sr)

        better = (
            cur_sc.confidence > best_sc.confidence
            or (_ok(cur_sc.valence, cur_sc.arousal) and not _ok(best_sc.valence, best_sc.arousal))
        )
        if better:
            best_y, best_sc = cur_y, cur_sc

    meta = {
        "final": dict(valence=best_sc.valence, arousal=best_sc.arousal,
                      label=best_sc.label, confidence=best_sc.confidence),
        "preset": p, "strength": strength, "iters": it,
        "exaggerate": exaggerate, "nuclear": nuclear,
    }
    return best_y, meta
