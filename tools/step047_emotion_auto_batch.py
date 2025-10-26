# -*- coding: utf-8 -*-
"""
tools/step047_emotion_auto_batch.py
Batch tuner for emotion with artifact guards and loudness normalization.
"""
from __future__ import annotations
import os, glob
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from loguru import logger
from scipy.signal import butter, sosfiltfilt

from .step045_emotion import auto_tune_emotion, _limiter_soft, _fade_edges, _db_to_lin

# Configurable loudness targets (keep in sync with step045)
import os
_TARGET_LUFS = float(os.getenv("EMO_TARGET_LUFS", "-16.0"))
_MAX_MAKEUP_DB = float(os.getenv("EMO_MAX_MAKEUP_DB", "9.0"))
_GATE_LUFS = float(os.getenv("EMO_GATE_LUFS", "-45.0"))

def _downmix_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2: y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)

def _xfade(a: np.ndarray, b: np.ndarray, xfade_samples: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if xfade_samples <= 0 or len(a) == 0: return np.concatenate([a,b]).astype(np.float32, copy=False)
    if len(b) == 0: return a
    x = min(int(xfade_samples), len(a), len(b))
    fo = np.linspace(1.0, 0.0, x, dtype=np.float32); fi = 1.0 - fo
    head = a[:-x] if x < len(a) else np.zeros(0, dtype=np.float32)
    tail = a[-x:] * fo + b[:x] * fi
    rest = b[x:]
    return np.concatenate([head, tail, rest]).astype(np.float32, copy=False)

def _segment_indices(n: int, sr: int, win_s: float, hop_s: float) -> List[Tuple[int,int]]:
    win = int(round(win_s*sr)); hop = int(round(hop_s*sr))
    if win <= 0 or hop <= 0: return [(0,n)]
    i=0; out=[]
    while i < n:
        j = min(n, i+win); out.append((i,j))
        if j >= n: break
        i += hop
    return out

def _butter_sos(sr, cutoff, btype, order=4):
    nyq = 0.5 * float(sr)
    wc = float(cutoff) / max(1e-9, nyq)
    wc = np.clip(wc, 1e-6, 0.999999)
    return butter(order, wc, btype=btype, output='sos')

# --- K-weighting (approx) & LUFS-like measurement ---
def _k_weighting_sos(sr: int):
    return _butter_sos(sr, 40.0, 'high', order=2)

def _lufs_like(y: np.ndarray, sr: int) -> float:
    if len(y) == 0: return -np.inf
    sos = _k_weighting_sos(sr)
    yk = sosfiltfilt(sos, y).astype(np.float32)
    ms = float(np.mean(yk**2) + 1e-12)
    return -0.691 + 10.0*np.log10(ms)

def _apply_loudness(y: np.ndarray, sr: int,
                    target_lufs: float = _TARGET_LUFS,
                    max_gain_db: float = _MAX_MAKEUP_DB,
                    gate_lufs: float = _GATE_LUFS) -> np.ndarray:
    cur = _lufs_like(y, sr)
    if not np.isfinite(cur) or cur < gate_lufs:
        return y  # don't lift near-silence
    delta = target_lufs - cur
    delta = float(np.clip(delta, -0.1, max_gain_db))  # only upward up to cap
    gain = _db_to_lin(delta)
    y2 = (y * gain).astype(np.float32)
    y2 = _limiter_soft(y2, thr_db=-1.5)
    return y2

def _hp_dc(y: np.ndarray, sr: int, cutoff: float = 20.0) -> np.ndarray:
    sos = _butter_sos(sr, cutoff, 'high', order=2)
    return sosfiltfilt(sos, y).astype(np.float32)

def _aa_lpf(y: np.ndarray, sr: int, cutoff: float = 11000.0) -> np.ndarray:
    sos = _butter_sos(sr, min(0.45*sr, cutoff), 'low', order=4)
    return sosfiltfilt(sos, y).astype(np.float32)

def _safe_write(path: str, y: np.ndarray, sr: int):
    import soundfile as sf
    # cleanup: DC and gentle AA
    y = _hp_dc(y, sr, cutoff=20.0)
    y = _aa_lpf(y, sr, cutoff=11000.0)
    # per-file loudness normalization (post-stitch)
    pre = _lufs_like(y, sr)
    y = _apply_loudness(y, sr, target_lufs=_TARGET_LUFS,
                        max_gain_db=_MAX_MAKEUP_DB, gate_lufs=_GATE_LUFS)
    post = _lufs_like(y, sr)
    logger.debug(f"[EmotionAutoBatch] {os.path.basename(path)} loudness {pre:.1f}→{post:.1f} LUFS")
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > 1.0: y = (y / peak).astype(np.float32)
    sf.write(path, y, sr)

def _parse_auto_preset(emotion: str) -> Optional[str]:
    if not emotion: return None
    e = emotion.strip().lower()
    if e == "auto": return "happy"
    if e.startswith("auto-"): return e.split("-",1)[1].strip() or "happy"
    return None

def auto_tune_emotion_all_wavs_under_folder(
    folder: str,
    emotion: str = "auto-angry",
    strength: float = 0.95,
    lang_hint: str = "en",
    win_s: float = 10.0,
    hop_s: float = 8.0,
    xfade_ms: int = 72,
    latency_budget_s: float = 1.2,
    min_confidence: float = 0.45,
    max_iters: int = 7,
    exaggerate: bool = True,
) -> tuple[bool, str]:
    target = _parse_auto_preset(emotion)
    if target is None: return False, f"Emotion '{emotion}' is not an auto-* mode"

    wav_dir = os.path.join(folder, "wavs")
    if not os.path.isdir(wav_dir): return False, f"No wavs dir: {wav_dir}"
    paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not paths: return False, f"No wav files in {wav_dir}"

    processed = 0
    xfade_cache = {}

    for p in paths:
        try:
            y, sr = sf.read(p, dtype="float32", always_2d=False)
            y = _downmix_mono(y)
            n = len(y)
            if n == 0:
                logger.warning(f"[EmotionAutoBatch] Empty file skipped: {p}")
                continue

            spans = _segment_indices(n, sr, win_s, hop_s)
            xfade = xfade_cache.get(sr)
            if xfade is None:
                xfade = max(0, int(round(xfade_ms * 1e-3 * sr)))
                xfade_cache[sr] = xfade

            out = np.zeros(0, dtype=np.float32)
            last_v, last_a, last_cf = 0.0, 0.0, 0.0

            for (i0,i1) in spans:
                seg = y[i0:i1]
                tuned, meta = auto_tune_emotion(
                    seg, sr,
                    target_preset=target,
                    strength=strength,
                    lang=lang_hint,
                    sentence_times=None,
                    latency_budget_s=latency_budget_s,
                    min_confidence=min_confidence,
                    max_iters=max_iters,
                    exaggerate=exaggerate,
                )
                tuned = _fade_edges(tuned, sr, ms=6.0)

                final = meta.get("final", {}) or {}
                v = float(final.get("valence", 0.0) or 0.0)
                a = float(final.get("arousal", 0.0) or 0.0)
                cf = float(final.get("confidence", 0.0) or 0.0)

                logger.debug(
                    f"[EmotionAutoBatch] {os.path.basename(p)} [{i0/sr:.2f}-{i1/sr:.2f}s] "
                    f"target={target}{' EXAG' if exaggerate else ''} → "
                    f"v={v:+.2f} a={a:+.2f} conf={cf:.2f}"
                )

                last_v, last_a, last_cf = v, a, cf
                out = _xfade(out, tuned, xfade) if len(out) else tuned

            _safe_write(p, out, sr)
            processed += 1
            logger.info(
                f"[EmotionAutoBatch] Auto-tuned {target} ({strength:.2f}) "
                f"{'[EXAG]' if exaggerate else ''} → "
                f"{os.path.basename(p)} | final: v={last_v:+.2f} a={last_a:+.2f} conf={last_cf:.2f}"
            )

        except Exception as e:
            logger.exception(f"[EmotionAutoBatch] Failed '{p}': {e}")

    return True, f"Auto-tuned {processed} file(s) to {target} ({strength:.2f}); loudness ≈ {_TARGET_LUFS:.1f} LUFS."
