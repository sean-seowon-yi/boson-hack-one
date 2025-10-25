import os
import time
import gc
from loguru import logger

import torch
import torchaudio
import torchaudio.functional as AF

# Demucs programmatic API
from demucs import pretrained
from demucs.apply import apply_model

# If you actually use these elsewhere keep them; otherwise you can remove.
from .utils import save_wav, normalize_wav  # noqa: F401

# -----------------------------
# Globals
# -----------------------------
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = None
_model_loaded = False
current_model_config = {}  # tracks current model name/device/shifts


def _pick_device(device: str):
    """Resolve device string."""
    if device == 'auto':
        return auto_device
    # allow 'cpu', 'cuda', or 'cuda:0' etc.
    return torch.device(device)


def init_demucs():
    """
    Initialize Demucs model if not already loaded.
    """
    global _model, _model_loaded
    if not _model_loaded:
        _model = load_model()
        _model_loaded = True
    else:
        logger.info("Demucs模型已经加载，跳过初始化")


def load_model(model_name: str = "htdemucs_ft",
               device: str = 'auto',
               progress: bool = True,
               shifts: int = 5):
    """
    Load a Demucs model via the official pretrained registry.
    Reuses a cached model if config matches; otherwise reloads.
    """
    global _model, _model_loaded, current_model_config

    requested_config = {
        'model_name': model_name,
        'device': 'auto' if device == 'auto' else str(device),
        'shifts': int(shifts)
    }

    if _model is not None and current_model_config == requested_config:
        logger.info('Demucs模型已加载且配置相同，重用现有模型')
        return _model

    if _model is not None:
        logger.info('Demucs模型配置改变，需要重新加载')
        release_model()

    logger.info(f'加载Demucs模型: {model_name}')
    t_start = time.time()

    device_to_use = _pick_device(device)

    # Load pretrained model
    model = pretrained.get_model(model_name)
    model.to(device_to_use)
    model.eval()

    # Store config (note: progress isn’t a model property; keep it in config for reuse checks if desired)
    current_model_config = requested_config

    # Cache
    _model = model
    _model_loaded = True

    t_end = time.time()
    logger.info(f'Demucs模型加载完成，用时 {t_end - t_start:.2f} 秒')

    return _model


def release_model():
    """
    Free model resources to avoid memory leaks.
    """
    global _model, _model_loaded, current_model_config

    if _model is not None:
        logger.info('正在释放Demucs模型资源...')
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _model_loaded = False
        current_model_config = {}
        logger.info('Demucs模型资源已释放')


def _load_audio_as_tensor(path: str, target_sr: int = 44100) -> torch.Tensor:
    """
    Loads audio to a (channels, samples) float32 tensor in [-1, 1] at target_sr.
    If mono, keeps 1 channel. If >2 channels, mixes down to stereo.
    """
    wav, sr = torchaudio.load(path)  # shape: (C, T)
    # Resample if needed
    if sr != target_sr:
        wav = AF.resample(wav, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    # If more than 2 channels, mix down to stereo
    if wav.shape[0] > 2:
        wav = wav[:2, :]  # simple cut to first two channels; or do mean across channels if you prefer
    # Ensure float32
    wav = wav.float()
    return wav  # (C, T) in [-1, 1]


def separate_audio(folder: str,
                   model_name: str = "htdemucs_ft",
                   device: str = 'auto',
                   progress: bool = True,
                   shifts: int = 5):
    """
    Separate audio using Demucs apply_model pipeline.
    Returns paths of saved vocals and instruments, or (None, None) if audio missing.
    """
    global _model

    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None

    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'音频已分离: {folder}')
        return vocal_output_path, instruments_output_path

    logger.info(f'正在分离音频: {folder}')

    try:
        # Ensure model loaded with requested config
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

        # Load audio tensor
        wav = _load_audio_as_tensor(audio_path, target_sr=44100)  # (C, T)
        # Demucs expects shape: (batch, channels, samples)
        wav_in = wav.unsqueeze(0).to(device_to_use)

        with torch.no_grad():
            # apply_model returns: (batch, sources, channels, samples)
            # We keep default overlap/split; control via kwargs if needed.
            sources_tensor = apply_model(
                _model,
                wav_in,
                shifts=int(shifts),
                progress=progress,
                overlap=0.25,
                split=True
            )[0]  # (sources, C, T)

        # Map names -> tensors
        # _model.sources is a list like ['drums','bass','other','vocals'] (varies by model)
        name_to_src = {name: sources_tensor[i].cpu() for i, name in enumerate(_model.sources)}

        # Vocals
        vocals = name_to_src.get('vocals', None)
        if vocals is None:
            # if the model doesn’t provide a 'vocals' stem, fall back to leaving vocals empty
            logger.warning("该Demucs模型不包含'vocals'声部，将输出空白人声。")
            vocals = torch.zeros_like(wav)

        # Instruments = sum of all non-vocals stems
        instruments = None
        for k, v in name_to_src.items():
            if k == 'vocals':
                continue
            instruments = v if instruments is None else (instruments + v)

        if instruments is None:
            # If only vocals provided, instruments = original - vocals (fallback)
            instruments = wav - vocals

        # Convert to numpy (T, C)
        vocals_np = vocals.transpose(0, 1).numpy()
        instruments_np = instruments.transpose(0, 1).numpy()

        # Save
        save_wav(vocals_np, vocal_output_path, sample_rate=44100)
        logger.info(f'已保存人声: {vocal_output_path}')

        save_wav(instruments_np, instruments_output_path, sample_rate=44100)
        logger.info(f'已保存伴奏: {instruments_output_path}')

        t_end = time.time()
        logger.info(f'音频分离完成，用时 {t_end - t_start:.2f} 秒')

        return vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分离音频失败: {str(e)}')
        # In case of any error, free model and rethrow
        release_model()
        raise


def extract_audio_from_video(folder: str) -> bool:
    """
    Extracts audio.wav from download.mp4 using ffmpeg.
    """
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False

    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'音频已提取: {folder}')
        return True

    logger.info(f'正在从视频提取音频: {folder}')
    os.system(
        f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"'
    )
    time.sleep(1)
    logger.info(f'音频提取完成: {folder}')
    return True


def separate_all_audio_under_folder(root_folder: str,
                                    model_name: str = "htdemucs_ft",
                                    device: str = 'auto',
                                    progress: bool = True,
                                    shifts: int = 5):
    """
    Walk root_folder, extract audio from videos, and separate all.
    """
    vocal_output_path, instruments_output_path = None, None

    try:
        for subdir, dirs, files in os.walk(root_folder):
            if 'download.mp4' not in files:
                continue
            if 'audio.wav' not in files:
                extract_audio_from_video(subdir)
                # refresh files after extraction
                files = set(os.listdir(subdir))
            if 'audio_vocals.wav' not in files or 'audio_instruments.wav' not in files:
                vocal_output_path, instruments_output_path = separate_audio(
                    subdir, model_name, device, progress, shifts
                )
            else:
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
                logger.info(f'音频已分离: {subdir}')

        logger.info(f'已完成所有音频分离: {root_folder}')
        return f'所有音频分离完成: {root_folder}', vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分离音频过程中出错: {str(e)}')
        release_model()
        raise


if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)
