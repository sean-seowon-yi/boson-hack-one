# -*- coding: utf-8 -*-
"""
tools/step047_emotion_auto_batch.py
Batch tuner that uses the rate-safe, DSP emotion (step045) and
rebuilds audio_tts.wav + audio_combined.wav for consistent loudness.

Key differences:
- Operates on wavs/*_adjusted.wav (the files actually used by the timeline)
- Raised-cosine crossfades between windowed emotion passes
- Per-clip LUFS makeup (gate + cap) after emotion DSP
- Rebuilds audio_tts.wav from translation.json timing
- Re-mixes instruments -> audio_combined.wav
- Final album-level loudness normalization + soft limiter
"""

from __future__ import annotations
import os, glob, json
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from loguru import logger

from .step045_emotion import (
    auto_tune_emotion,
    # reuse the same targets via env if present
)

# ======= Small local audio utils (no circular imports) =======

SR_DEFAULT = 24000  # must match your TTS SR

def _downmix_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)

def _safe_write(path: str, y: np.ndarray, sr: int):
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    sf.write(path, y, sr)

def _hann_xfade(a: np.ndarray, b: np.ndarray, xfade_samples: int) -> np.ndarray:
    """Raised-cosine (Hann) crossfade."""
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if xfade_samples <= 0 or len(a) == 0:
        return np.concatenate([a, b]).astype(np.float32, copy=False)
    if len(b) == 0:
        return a
    x = min(int(xfade_samples), len(a), len(b))
    # Hann half-windows
    t = np.linspace(0, np.pi, x, dtype=np.float32)
    fo = 0.5 * (1.0 + np.cos(t))   # fade-out (cosine from 1->0)
    fi = 1.0 - fo                  # complementary fade-in
    head = a[:-x] if x < len(a) else np.zeros(0, dtype=np.float32)
    tail = a[-x:] * fo + b[:x] * fi
    rest = b[x:]
    return np.concatenate([head, tail, rest]).astype(np.float32, copy=False)

# --- very light LUFS-like and limiter (copies, no import cycle) ---
def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def _limiter_soft(y: np.ndarray, thr_db: float = -1.5) -> np.ndarray:
    thr = _db_to_lin(thr_db)
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak <= thr:
        return y.astype(np.float32)
    return (y / peak * thr).astype(np.float32)

def _lufs_like(y: np.ndarray, sr: int) -> float:
    # K-weighting approx: 2nd-order HP @ ~40 Hz, then RMS->LUFS-ish
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * float(sr)
    wc = 40.0 / max(1e-9, nyq)
    wc = np.clip(wc, 1e-6, 0.999999)
    sos = butter(2, wc, btype='high', output='sos')
    yk = sosfiltfilt(sos, y).astype(np.float32)
    ms = float(np.mean(yk**2) + 1e-12)
    return float(-0.691 + 10.0*np.log10(ms))

def _loudness_normalize(y: np.ndarray, sr: int,
                        target_lufs: float = -16.0,
                        max_gain_db: float = 9.0,
                        gate_lufs: float = -45.0) -> np.ndarray:
    cur = _lufs_like(y, sr)
    if not np.isfinite(cur) or cur < gate_lufs:
        # very quiet/silent: avoid boosting noise
        return y.astype(np.float32)
    delta = target_lufs - cur
    delta = float(np.clip(delta, -0.1, max_gain_db))  # upward only, up to cap
    y2 = (y * _db_to_lin(delta)).astype(np.float32)
    return _limiter_soft(y2, thr_db=-1.5)

def _fade_edges(y: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    if len(y) == 0: return y
    n = int(sr * (ms/1000.0))
    n = max(8, min(n, len(y)//4))
    if n <= 0: return y
    env = np.ones_like(y)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    env[:n] *= ramp
    env[-n:] *= ramp[::-1]
    return (y * env).astype(np.float32)

# ============================================================

def _segment_indices(n: int, sr: int, win_s: float, hop_s: float) -> List[Tuple[int,int]]:
    win = int(round(win_s*sr)); hop = int(round(hop_s*sr))
    if win <= 0 or hop <= 0: return [(0,n)]
    i=0; out=[]
    while i < n:
        j = min(n, i+win); out.append((i,j))
        if j >= n: break
        i += hop
    return out

def _parse_auto_preset(emotion: str) -> Optional[str]:
    if not emotion: return None
    e = emotion.strip().lower()
    if e == "auto": return "happy"
    if e.startswith("auto-"): return e.split("-",1)[1].strip() or "happy"
    return None

def _load_translation(folder: str):
    p = os.path.join(folder, "translation.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"translation.json missing in {folder}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _rebuild_tts_and_combined(
    folder: str,
    sr: int = SR_DEFAULT,
    target_lufs: float = -16.0,
    max_gain_db: float = 9.0,
    gate_lufs: float = -45.0,
):
    """
    Reconstruct audio_tts.wav from wavs/*_adjusted.wav according to translation.json,
    then mix with audio_instruments.wav -> audio_combined.wav.
    Apply album-level loudness normalization on the combined.
    """
    tr = _load_translation(folder)
    if not tr:
        logger.warning(f"[EmotionAutoBatch] Empty translation in {folder}")
        return

    wav_dir = os.path.join(folder, "wavs")

    # Determine total length from translation
    total_len_s = 0.0
    for line in tr:
        total_len_s = max(total_len_s, float(line.get("end", 0.0)))
    total_n = int(round(total_len_s * sr))
    tts_buf = np.zeros(total_n, dtype=np.float32)

    for i, line in enumerate(tr):
        start = float(line.get("start", 0.0))
        end   = float(line.get("end",   0.0))
        dur   = max(0.0, end - start)
        if dur <= 0: continue

        # Prefer the adjusted clip, fall back to raw index if needed
        base = f"{i:04d}"
        p_adj = os.path.join(wav_dir, f"{base}_adjusted.wav")
        p_raw = os.path.join(wav_dir, f"{base}.wav")
        path  = p_adj if os.path.exists(p_adj) else p_raw
        if not os.path.exists(path):
            logger.warning(f"[EmotionAutoBatch] Missing segment {path}")
            continue

        y, sr_file = sf.read(path, dtype="float32", always_2d=False)
        y = _downmix_mono(y)
        if sr_file != sr:
            # cheap resample via librosa if available
            try:
                import librosa
                y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
            except Exception:
                logger.warning(f"[EmotionAutoBatch] Resample failed; placing as-is")
        # Trim/pad to slot length
        slot_n = int(round(dur * sr))
        if slot_n <= 0: continue
        if len(y) > slot_n:
            y = y[:slot_n]
        elif len(y) < slot_n:
            y = np.pad(y, (0, slot_n - len(y)), mode="constant")
        y = _fade_edges(y, sr, ms=5.0)
        i0 = int(round(start * sr)); i1 = min(total_n, i0 + slot_n)
        if i1 > i0:
            tts_buf[i0:i1] += y[:(i1 - i0)]

    # Write audio_tts.wav
    tts_path = os.path.join(folder, "audio_tts.wav")
    _safe_write(tts_path, tts_buf, sr)

    # Mix with instruments if present
    inst_path = os.path.join(folder, "audio_instruments.wav")
    if os.path.exists(inst_path):
        inst, sr_i = sf.read(inst_path, dtype="float32", always_2d=False)
        inst = _downmix_mono(inst)
        if sr_i != sr:
            try:
                import librosa
                inst = librosa.resample(inst, orig_sr=sr_i, target_sr=sr)
            except Exception:
                pass
        # Align lengths
        if len(inst) < len(tts_buf):
            inst = np.pad(inst, (0, len(tts_buf) - len(inst)), mode="constant")
        elif len(inst) > len(tts_buf):
            tts_buf = np.pad(tts_buf, (0, len(inst) - len(tts_buf)), mode="constant")
        mixed = (tts_buf + inst).astype(np.float32)
    else:
        mixed = tts_buf

    # Album-level loudness + limiter
    mixed = _loudness_normalize(mixed, sr, target_lufs=target_lufs, max_gain_db=max_gain_db, gate_lufs=gate_lufs)
    mixed = _limiter_soft(mixed, thr_db=-1.5)

    combined_path = os.path.join(folder, "audio_combined.wav")
    _safe_write(combined_path, mixed, sr)
    logger.info(f"[EmotionAutoBatch] Rebuilt audio_tts.wav and audio_combined.wav in {folder}")

# ============================================================

def auto_tune_emotion_all_wavs_under_folder(
    folder: str,
    emotion: str = "auto-angry",
    strength: float = 0.85,
    lang_hint: str = "en",
    win_s: float = 10.0,
    hop_s: float = 9.0,
    xfade_ms: int = 45,                 # longer, smoother crossfade
    latency_budget_s: float = 1.0,
    min_confidence: float = 0.40,
    max_iters: int = 6,
    exaggerate: bool = True,

    # Loudness knobs (clip-level & album-level)
    clip_target_lufs: float = -18.0,    # per-clip target (slightly lower to preserve headroom)
    clip_max_gain_db: float = 9.0,
    clip_gate_lufs: float = -45.0,
    album_target_lufs: float = -16.0,   # final combined target
    album_max_gain_db: float = 6.0,
    album_gate_lufs: float = -45.0,
) -> tuple[bool, str]:
    """
    Emotion-tune wavs/*_adjusted.wav in windowed manner and rebuild combined audio.
    """
    target = _parse_auto_preset(emotion)
    if target is None:
        return False, f"Emotion '{emotion}' is not an auto-* mode"

    wav_dir = os.path.join(folder, "wavs")
    if not os.path.isdir(wav_dir):
        return False, f"No wavs dir: {wav_dir}"

    # CRITICAL: operate on the adjusted clips used by the timeline
    paths = sorted(glob.glob(os.path.join(wav_dir, "*_adjusted.wav")))
    # if no adjusted clips, fall back to raw
    if not paths:
        paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not paths:
        return False, f"No wav files in {wav_dir}"

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
                final = meta.get("final", {}) or {}
                v = float(final.get("valence", 0.0) or 0.0)
                a = float(final.get("arousal", 0.0) or 0.0)
                cf = float(final.get("confidence", 0.0) or 0.0)

                # Per-window clip loudness makeup (keeps windows consistent)
                tuned = _loudness_normalize(
                    tuned, sr,
                    target_lufs=clip_target_lufs,
                    max_gain_db=clip_max_gain_db,
                    gate_lufs=clip_gate_lufs
                )

                logger.debug(
                    f"[EmotionAutoBatch] {os.path.basename(p)} [{i0/sr:.2f}-{i1/sr:.2f}s] "
                    f"target={target}{' EXAG' if exaggerate else ''} → "
                    f"v={v:+.2f} a={a:+.2f} conf={cf:.2f}"
                )

                last_v, last_a, last_cf = v, a, cf
                out = _hann_xfade(out, tuned, xfade) if len(out) else tuned

            # Tiny click-suppression fades on final clip
            out = _fade_edges(out, sr, ms=5.0)
            _safe_write(p, out, sr)
            processed += 1
            logger.info(
                f"[EmotionAutoBatch] Auto-tuned {target} ({strength:.2f}) "
                f"{'[EXAG]' if exaggerate else ''} → "
                f"{os.path.basename(p)} | final: v={last_v:+.2f} a={last_a:+.2f} conf={last_cf:.2f}"
            )

        except Exception as e:
            logger.exception(f"[EmotionAutoBatch] Failed '{p}': {e}")

    # After per-clip processing, rebuild tts + combined with ALBUM loudness
    try:
        _rebuild_tts_and_combined(
            folder,
            sr=SR_DEFAULT,
            target_lufs=album_target_lufs,
            max_gain_db=album_max_gain_db,
            gate_lufs=album_gate_lufs,
        )
    except Exception as e:
        logger.warning(f"[EmotionAutoBatch] Rebuild combined failed: {e}")

    return True, f"Auto-tuned {processed} file(s) to {target} ({strength:.2f}), rebuilt combined with album loudness."
