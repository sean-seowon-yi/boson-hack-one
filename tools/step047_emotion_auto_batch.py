# -*- coding: utf-8 -*-
"""
step047_emotion_auto_batch.py
Auto emotion tuning (Higgs-understanding feedback) for all synthesized WAVs in a folder.

Strategy:
- Segment long WAVs into ~10s windows (with small overlap) to keep scoring fast and robust.
- Run auto_tune_emotion(...) per window.
- Recombine with short crossfades to avoid seams.

Public API:
    auto_tune_emotion_all_wavs_under_folder(folder, emotion="auto-happy", strength=0.6, lang_hint="en", ...)
"""

from __future__ import annotations
import os, glob, time
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from loguru import logger

from .step045_emotion import auto_tune_emotion

# ---------------- helpers ---------------- #

def _downmix_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)

def _xfade(a: np.ndarray, b: np.ndarray, xfade_samples: int) -> np.ndarray:
    """Linear crossfade 'a' into 'b' with overlap xfade_samples."""
    if xfade_samples <= 0 or len(a) == 0 or len(b) == 0:
        return np.concatenate([a, b], dtype=np.float32)
    n1 = len(a); n2 = len(b)
    x = min(xfade_samples, n1, n2)
    fade_out = np.linspace(1.0, 0.0, x, dtype=np.float32)
    fade_in  = 1.0 - fade_out
    head = a[:-x] if x < n1 else np.zeros(0, dtype=np.float32)
    tail = a[-x:] * fade_out + b[:x] * fade_in
    rest = b[x:]
    return np.concatenate([head, tail, rest], dtype=np.float32)

def _segment_indices(n_samples: int, sr: int, win_s: float, hop_s: float) -> List[Tuple[int,int]]:
    win = int(round(win_s * sr))
    hop = int(round(hop_s * sr))
    if win <= 0 or hop <= 0:
        return [(0, n_samples)]
    idx = []
    i = 0
    while i < n_samples:
        j = min(n_samples, i + win)
        idx.append((i, j))
        if j >= n_samples:
            break
        i += hop
    return idx

def _safe_write(path: str, y: np.ndarray, sr: int):
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    sf.write(path, y, sr)

def _parse_auto_preset(emotion: str) -> Optional[str]:
    """
    Accepts 'auto-happy', 'auto-sad', 'auto-angry', or just 'auto' (defaults to happy).
    Returns target preset string ('happy'/'sad'/'angry') or None if not auto.
    """
    if not emotion:
        return None
    e = emotion.strip().lower()
    if e == "auto":
        return "happy"
    if e.startswith("auto-"):
        return e.split("-", 1)[1].strip() or "happy"
    return None

# ---------------- main batch API ---------------- #

def auto_tune_emotion_all_wavs_under_folder(
    folder: str,
    emotion: str = "auto-happy",
    strength: float = 0.6,
    lang_hint: str = "en",
    win_s: float = 10.0,          # segment window (sec)
    hop_s: float = 9.0,           # hop (sec) -> 1s overlap
    xfade_ms: int = 28,           # overlap when stitching back
    latency_budget_s: float = 0.5,
    min_confidence: float = 0.50,
    max_iters: int = 2,
) -> tuple[bool, str]:
    """
    Auto-tune emotion for every WAV in <folder>/wavs/*.wav using Higgs understanding feedback.

    Returns:
        (ok, message)
    """
    target = _parse_auto_preset(emotion)
    if target is None:
        return False, f"Emotion '{emotion}' is not an auto-* mode"

    wav_dir = os.path.join(folder, "wavs")
    if not os.path.isdir(wav_dir):
        return False, f"No wavs dir: {wav_dir}"
    paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not paths:
        return False, f"No wav files in {wav_dir}"

    processed = 0
    xfade_samples_cache = {}

    for p in paths:
        try:
            y, sr = sf.read(p, dtype="float32", always_2d=False)
            y = _downmix_mono(y)
            n = len(y)
            if n == 0:
                logger.warning(f"[EmotionAutoBatch] Empty file skipped: {p}")
                continue

            # Segment indices
            spans = _segment_indices(n, sr, win_s=win_s, hop_s=hop_s)
            xfade_samples = xfade_samples_cache.get(sr)
            if xfade_samples is None:
                xfade_samples = max(0, int(round(xfade_ms * 1e-3 * sr)))
                xfade_samples_cache[sr] = xfade_samples

            out = np.zeros(0, dtype=np.float32)
            for (i0, i1) in spans:
                seg = y[i0:i1]

                # Auto-tune this window
                tuned, meta = auto_tune_emotion(
                    seg, sr,
                    target_preset=target,
                    strength=strength,
                    lang=lang_hint,
                    sentence_times=None,
                    latency_budget_s=latency_budget_s,
                    min_confidence=min_confidence,
                    max_iters=max_iters,
                )
                logger.debug(f"[EmotionAutoBatch] {os.path.basename(p)} [{i0/sr:.2f}-{i1/sr:.2f}s] "
                             f"target={target} → final={meta.get('final',{})}")

                # Stitch with crossfade
                out = _xfade(out, tuned, xfade_samples) if len(out) else tuned

            _safe_write(p, out, sr)
            processed += 1
            logger.info(f"[EmotionAutoBatch] Auto-tuned {target} ({strength:.2f}) → {os.path.basename(p)}")

        except Exception as e:
            logger.exception(f"[EmotionAutoBatch] Failed '{p}': {e}")

    return True, f"Auto-tuned {processed} file(s) to {target} ({strength:.2f})."
