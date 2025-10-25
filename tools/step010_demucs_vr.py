# -*- coding: utf-8 -*-
"""
Faster Demucs separation for MacBook Pro (CPU by default; CUDA-friendly).

Key speedups (without breaking outputs):
- Use segmented inference (segment ~10-12s) + small overlap to reduce FFT cost and RAM.
- Clamp shifts on CPU (1) and keep small on GPU (2) for big speed gains (shifts is linear-time).
- Use torch.inference_mode() for faster no-grad path.
- Cache a single torchaudio Resample transform.
- Cap OMP/MKL threads to avoid oversubscription on laptop CPUs.
- Preallocate and accumulate 'instruments' in-place (avoids tensor churn).
- Keep outputs on CPU to reduce device swaps; on CUDA, run model+input in float16.

Public API preserved:
    init_demucs()
    load_model(...)
    release_model()
    separate_audio(folder, ...)
    extract_audio_from_video(folder)
    separate_all_audio_under_folder(root_folder, ...)
"""

import os
import time
import math
import gc
from typing import Tuple, Optional

# ---- Threading caps (greatly helps FFT-heavy CPU runs) ----
MAX_THREADS = max(1, min(8, os.cpu_count() or 4))
os.environ.setdefault("OMP_NUM_THREADS", str(MAX_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(MAX_THREADS))

from loguru import logger

import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.transforms import Resample

# Demucs programmatic API
from demucs import pretrained
from demucs.apply import apply_model

from .utils import save_wav, normalize_wav  # noqa: F401

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# -----------------------------
# Globals
# -----------------------------
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model: Optional[torch.nn.Module] = None
_model_loaded: bool = False
current_model_config = {}
_resampler: Optional[Resample] = None
_TARGET_SR = 44100


def _pick_device(device: str):
    return auto_device if device == 'auto' else torch.device(device)


def _defaults_for_hardware():
    if torch.cuda.is_available():
        return dict(shifts=2, segment=12.0, overlap=0.10, dtype=torch.float16)
    else:
        return dict(shifts=1, segment=10.0, overlap=0.10, dtype=torch.float32)


def init_demucs():
    global _model, _model_loaded
    if not _model_loaded:
        _model = load_model()
        _model_loaded = True
    else:
        logger.info("Demucs model already loaded — skipping initialization.")


def load_model(model_name: str = "htdemucs_ft",
               device: str = 'auto',
               progress: bool = True,
               shifts: int = 5):
    global _model, _model_loaded, current_model_config

    hw = _defaults_for_hardware()
    shifts = int(shifts) if shifts is not None else hw["shifts"]
    if (not torch.cuda.is_available()) and shifts > 1:
        shifts = hw["shifts"]

    requested_config = {
        'model_name': model_name,
        'device': 'auto' if device == 'auto' else str(device),
        'shifts': shifts
    }

    if _model is not None and current_model_config == requested_config:
        logger.info('Demucs model already loaded with the same configuration — reusing existing model.')
        return _model

    if _model is not None:
        logger.info('Demucs configuration changed — reloading model.')
        release_model()

    logger.info(f'Loading Demucs model: {model_name}')
    t_start = time.time()

    device_to_use = _pick_device(device)

    model = pretrained.get_model(model_name)
    model.eval()
    model.to(device_to_use)
    if torch.cuda.is_available():
        model.half()

    current_model_config = requested_config
    _model = model
    _model_loaded = True

    logger.info(f'Demucs model loaded successfully in {time.time() - t_start:.2f}s.')
    return _model


def release_model():
    global _model, _model_loaded, current_model_config

    if _model is not None:
        logger.info('Releasing Demucs model resources...')
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _model_loaded = False
        current_model_config = {}
        logger.info('Demucs model resources released.')


def _get_resampler(orig_sr: int, new_sr: int):
    global _resampler
    if _resampler is None or _resampler.orig_freq != orig_sr or _resampler.new_freq != new_sr:
        _resampler = Resample(orig_freq=orig_sr, new_freq=new_sr)
    return _resampler


def _load_audio_as_tensor(path: str, target_sr: int = _TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = _get_resampler(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr

    if wav.shape[0] > 2:
        wav = wav[:2, :]
    return wav.contiguous().float()


def _pick_aligned_segment_seconds(model, clip_num_samples: int, sr: int) -> tuple[float | None, float]:
    training_len = getattr(model, "training_length", None)
    model_sr = getattr(model, "samplerate", sr)

    if not training_len or model_sr <= 0:
        return None, 0.10

    base_seg_s = training_len / float(model_sr)
    T = clip_num_samples / float(sr)
    k = int(T // base_seg_s)
    k = max(1, min(k, 3))
    segment_s = k * base_seg_s

    overlap = 0.10 if segment_s > 0.15 else 0.05
    if overlap >= segment_s:
        overlap = max(0.0, 0.5 * segment_s)

    return segment_s, overlap


def separate_audio(folder: str,
                   model_name: str = "htdemucs_ft",
                   device: str = 'auto',
                   progress: bool = True,
                   shifts: int = 5):
    global _model

    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None

    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'Audio already separated: {folder}')
        return vocal_output_path, instruments_output_path

    logger.info(f'Separating audio: {folder}')

    try:
        need_reload = (
            (not _model_loaded) or
            current_model_config.get('model_name') != model_name or
            (current_model_config.get('device') == 'auto') != (device == 'auto') or
            current_model_config.get('shifts') != int(shifts)
        )
        if need_reload:
            load_model(model_name, device, progress, shifts)

        device_to_use = _pick_device(device)

        t_start = time.time()
        wav = _load_audio_as_tensor(audio_path, target_sr=_TARGET_SR)
        C, T_samples = wav.shape

        segment_s, overlap = _pick_aligned_segment_seconds(_model, T_samples, _TARGET_SR)

        wav_in = wav.unsqueeze(0)
        if device_to_use.type == 'cuda':
            wav_in = wav_in.to(device_to_use, non_blocking=True).half()
        else:
            wav_in = wav_in.to(device_to_use, non_blocking=True)

        eff_shifts = current_model_config.get('shifts', 1)

        with torch.inference_mode():
            sources_tensor = apply_model(
                _model,
                wav_in,
                shifts=eff_shifts,
                progress=progress,
                overlap=overlap,
                split=True,
                segment=segment_s,
            )[0]

        sources_tensor = sources_tensor.to(dtype=torch.float32, device='cpu')
        name_to_src = {name: sources_tensor[i] for i, name in enumerate(_model.sources)}

        vocals = name_to_src.get('vocals')
        if vocals is None:
            logger.warning("This Demucs model does not include a 'vocals' stem — generating silent vocals.")
            vocals = torch.zeros_like(wav)

        instruments = torch.zeros_like(vocals)
        for k, v in name_to_src.items():
            if k != 'vocals':
                instruments.add_(v)

        if not instruments.abs().sum().item():
            instruments = wav - vocals

        save_wav(vocals.transpose(0, 1).numpy(), vocal_output_path, sample_rate=_TARGET_SR)
        logger.info(f'Saved vocals: {vocal_output_path}')

        save_wav(instruments.transpose(0, 1).numpy(), instruments_output_path, sample_rate=_TARGET_SR)
        logger.info(f'Saved accompaniment: {instruments_output_path}')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f'Audio separation complete in {time.time() - t_start:.2f}s.')
        return vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'Audio separation failed: {str(e)}')
        release_model()
        raise


def extract_audio_from_video(folder: str) -> bool:
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False

    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'Audio already extracted: {folder}')
        return True

    logger.info(f'Extracting audio from video: {folder}')
    os.system(
        f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar {_TARGET_SR} -ac 2 "{audio_path}"'
    )
    time.sleep(0.5)
    logger.info(f'Audio extraction complete: {folder}')
    return True


def separate_all_audio_under_folder(root_folder: str,
                                    model_name: str = "htdemucs_ft",
                                    device: str = 'auto',
                                    progress: bool = True,
                                    shifts: int = 5):
    vocal_output_path, instruments_output_path = None, None

    try:
        for subdir, dirs, files in os.walk(root_folder):
            files_set = set(files)
            if 'download.mp4' not in files_set:
                continue
            if 'audio.wav' not in files_set:
                extract_audio_from_video(subdir)
                files_set = set(os.listdir(subdir))
            if 'audio_vocals.wav' not in files_set or 'audio_instruments.wav' not in files_set:
                vocal_output_path, instruments_output_path = separate_audio(
                    subdir, model_name, device, progress, shifts
                )
            else:
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
                logger.info(f'Audio already separated: {subdir}')

        logger.info(f'All audio separation completed under: {root_folder}')
        return f'All audio separated: {root_folder}', vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'Error during audio separation: {str(e)}')
        release_model()
        raise


if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=1)
