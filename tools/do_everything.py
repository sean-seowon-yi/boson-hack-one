# -*- coding: utf-8 -*-
"""
tools/do_everything.py

End-to-end pipeline with post-TTS Emotion control (auto-batch available).

Folder policy (restored original style):
  - Parent folder: <author>
  - Child folder:  <language>_<emotion>_<short_title(no dates)>_<ttsmodel>
  - If the child already exists, it is REMOVED and recreated (overwrite).

Supported languages (translation + TTS):
  - Simplified Chinese (zh-cn), English (en), Korean (ko), Spanish (es), French (fr)
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from loguru import logger

from .step000_video_downloader import (
    get_info_list_from_url,
    download_single_video,
    get_target_folder,
)
from .step010_demucs_vr import separate_all_audio_under_folder, init_demucs, release_model
from .step020_asr import transcribe_all_audio_under_folder
from .step030_translation import translate_all_transcript_under_folder
from .step040_tts import generate_all_wavs_under_folder
from .step042_tts_xtts import init_TTS
from .step043_tts_cosyvoice import init_cosyvoice
from .step050_synthesize_video import synthesize_all_video_under_folder
from .step047_emotion_auto_batch import auto_tune_emotion_all_wavs_under_folder

# ---------------------------------------------------------------------
# Model init flags
# ---------------------------------------------------------------------
models_initialized = {
    "demucs": False,
    "xtts": False,
    "cosyvoice": False,
    "diarize": False,
    "funasr": False,
}

# ---------------------------------------------------------------------
# Language normalization (restricted to 5)
# ---------------------------------------------------------------------
_TRANSLATION_ALIASES = {
    "simplified chinese (简体中文)": "zh-cn", "简体中文": "zh-cn", "chinese": "zh-cn", "zh-cn": "zh-cn", "zh": "zh-cn",
    "english": "en", "en": "en",
    "korean": "ko", "한국어": "ko", "ko": "ko",
    "spanish": "es", "español": "es", "es": "es",
    "french": "fr", "français": "fr", "fr": "fr",
}
_TTS_ALIASES = {
    "chinese (中文)": "zh-cn", "中文": "zh-cn", "chinese": "zh-cn", "zh": "zh-cn", "zh-cn": "zh-cn",
    "english": "en", "en": "en",
    "korean": "ko", "韩国语": "ko", "한국어": "ko", "ko": "ko",
    "spanish": "es", "español": "es", "es": "es",
    "french": "fr", "français": "fr", "fr": "fr",
}
_ALLOWED_SUB_LANGS = {"zh-cn", "en", "ko", "es", "fr"}
_ALLOWED_TTS_LANGS = {"zh-cn", "en", "ko", "es", "fr"}

def _canon(s: Optional[str]) -> Optional[str]:
    return None if s is None else str(s).strip().lower()

def _norm_translation_lang(v: str) -> str:
    code = _TRANSLATION_ALIASES.get(_canon(v), _canon(v))
    if code not in _ALLOWED_SUB_LANGS:
        raise ValueError(f"Unrecognized subtitle/translation language: {v}")
    return code

def _norm_tts_lang(v: str) -> str:
    code = _TTS_ALIASES.get(_canon(v), _canon(v))
    if code not in _ALLOWED_TTS_LANGS:
        raise ValueError(f"Unrecognized TTS language: {v}")
    return code

def _coerce_int_or_none(x):
    if x in (None, "", "None"):
        return None
    try:
        return int(x)
    except Exception:
        return None

def get_available_gpu_memory() -> float:
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            used = torch.cuda.memory_allocated(0)
            return (total - used) / (1024 ** 3)
        return 0.0
    except Exception:
        return 0.0

# ---------------------------------------------------------------------
# Folder helpers (author parent + language_emotion_title_ttsmodel child)
# ---------------------------------------------------------------------
def _slugify_component(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "-").replace("\\", "-").replace(":", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unknown"

def _strip_dates(title: str) -> str:
    if not title:
        return title
    s = str(title)
    s = re.sub(r"\b(?:19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2}\b", "", s)  # 2018-06-11 / 2018.06.11 / 2018/06/11
    s = re.sub(r"\b(?:19|20)\d{6}\b", "", s)                          # 20180611
    s = re.sub(r"\s{2,}", " ", s).strip(" -_")
    return s

def _truncate_reasonably(s: str, max_len: int) -> str:
    if not s:
        return s
    s = s.strip()
    if len(s) <= max_len:
        return s
    cut = s[:max_len+1]
    last_space = cut.rfind(" ")
    if last_space >= max_len // 2:
        return cut[:last_space].rstrip() + "…"
    return s[:max_len].rstrip() + "…"

def _shorten_title(raw_title: str) -> str:
    tmax = int(os.getenv("TITLE_MAX_LEN", "64"))  # room for added components
    return _truncate_reasonably(_strip_dates(raw_title or "untitled"), tmax) or "untitled"

def _extract_author(info, author_override: Optional[str]) -> str:
    if author_override:
        return author_override
    if isinstance(info, dict):
        for key in ("uploader", "channel", "author", "owner", "artist"):
            v = info.get(key)
            if v:
                return str(v)
    return "local"

def _compose_author_title_folder(
    root_folder: str,
    base_folder: str,
    title: str,
    author: str,
    lang_code: str,
    emotion: str,
    tts_model: str,
    overwrite: bool = True,   # always overwrite duplicates per request
) -> str:
    """
    Create: <root>/<author>/<language>_<emotion>_<short_title>_<ttsmodel>
    Move contents of base_folder into the child folder.
    """
    author_dir = os.path.join(root_folder, _slugify_component(author or "local"))
    os.makedirs(author_dir, exist_ok=True)

    child_name = "_".join([
        _slugify_component(lang_code or "en"),
        _slugify_component((emotion or "natural").lower()),
        _slugify_component(_shorten_title(title)),
        _slugify_component((tts_model or "tts").lower()),
    ])
    child_path = os.path.join(author_dir, child_name)

    if os.path.exists(child_path) and overwrite:
        logger.info(f"[Overwrite] Removing existing child folder: {child_path}")
        shutil.rmtree(child_path, ignore_errors=True)
    os.makedirs(child_path, exist_ok=True)

    # Move/copy everything from the temporary base folder into the final child folder
    if os.path.abspath(base_folder) != os.path.abspath(child_path) and os.path.isdir(base_folder):
        for name in os.listdir(base_folder):
            src = os.path.join(base_folder, name)
            dst = os.path.join(child_path, name)
            try:
                shutil.move(src, dst)
            except Exception:
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    shutil.rmtree(src, ignore_errors=True)
                else:
                    shutil.copy2(src, dst)
                    try:
                        os.remove(src)
                    except Exception:
                        pass
        try:
            os.rmdir(base_folder)
        except Exception:
            pass

    return child_path

# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------
def initialize_models(tts_method: str, asr_method: str, diarization: bool) -> None:
    global models_initialized
    futures = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            if not models_initialized["demucs"]:
                futures.append(executor.submit(init_demucs))
                models_initialized["demucs"] = True
                logger.info("Initialized Demucs")
            else:
                logger.info("Demucs already initialized — skipping")

            if tts_method == "xtts":
                if not models_initialized["xtts"]:
                    futures.append(executor.submit(init_TTS))
                    models_initialized["xtts"] = True
                    logger.info("Initialized XTTS")
            elif tts_method == "cosyvoice":
                if not models_initialized["cosyvoice"]:
                    futures.append(executor.submit(init_cosyvoice))
                    models_initialized["cosyvoice"] = True
                    logger.info("Initialized CosyVoice")
            elif tts_method == "Higgs":
                logger.info("TTS 'Higgs' selected — API-based")

            if asr_method == "FunASR" and not models_initialized.get("funasr", False):
                models_initialized["funasr"] = True
                logger.info("Initialized FunASR")
            elif asr_method == "Higgs":
                logger.info("ASR 'Higgs' selected — API-based")

            for fut in futures:
                fut.result()
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Failed to initialize models: {e}\n{stack_trace}")
        models_initialized = {k: False for k in models_initialized}
        release_model()
        raise

# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------
def process_video(
    info,
    root_folder,
    resolution,
    demucs_model,
    device,
    shifts,
    asr_method,
    whisper_model,
    batch_size,
    diarization,
    whisper_min_speakers,
    whisper_max_speakers,
    translation_method,
    translation_target_language,   # may be label or code
    tts_method,
    tts_target_language,           # may be label or code
    voice,
    subtitles,
    speed_up,
    fps,
    background_music,
    bgm_volume,
    video_volume,
    target_resolution,
    max_retries,
    progress_callback=None,
    *,
    emotion: str = "natural",
    emotion_strength: float = 0.6,
    overwrite_duplicates: bool = True,       # default overwrite
    author_override: Optional[str] = None,
):
    stages = [
        ("Downloading video...", 10),
        ("Separating vocals...", 15),
        ("Speech recognition...", 20),
        ("Translating subtitles...", 25),
        ("Synthesizing speech...", 20),
        ("Compositing video...", 10),
    ]
    current_stage = 0
    progress_base = 0

    if progress_callback:
        progress_callback(0, "Preparing...")

    for retry in range(max_retries):
        try:
            # Stage: Download
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            if isinstance(info, str) and info.endswith(".mp4"):
                original_file_name = os.path.basename(info)
                base_folder_name = os.path.splitext(original_file_name)[0]
                base_folder = os.path.join(root_folder, base_folder_name)
                os.makedirs(base_folder, exist_ok=True)
                dest_path = os.path.join(base_folder, "download.mp4")
                shutil.copy2(info, dest_path)
            else:
                base_folder = get_target_folder(info, root_folder)
                if base_folder is None:
                    error_msg = f'Unable to derive target folder: {info.get("title") if isinstance(info, dict) else info}'
                    logger.warning(error_msg)
                    return False, None, error_msg
                base_folder = download_single_video(info, root_folder, resolution)
                if base_folder is None:
                    error_msg = f'Download failed: {info.get("title") if isinstance(info, dict) else info}'
                    logger.warning(error_msg)
                    return False, None, error_msg

            # Compose final working folder:
            # <root>/<author>/<language>_<emotion>_<short_title>_<ttsmodel>
            author = _extract_author(info, author_override)
            raw_title = info.get("title") if isinstance(info, dict) else os.path.basename(base_folder)
            tts_lang_code = _norm_tts_lang(tts_target_language)

            folder = _compose_author_title_folder(
                root_folder=root_folder,
                base_folder=base_folder,
                title=raw_title,
                author=author,
                lang_code=tts_lang_code,
                emotion=emotion,
                tts_model=tts_method,
                overwrite=True,  # enforce overwrite on duplicates
            )
            logger.info(
                f"Processing folder: {folder} "
                f"(author='{author}', lang={tts_lang_code}, emotion={emotion}, tts={tts_method})"
            )

            # Stage: Vocal separation
            current_stage += 1; progress_base += stage_weight
            if progress_callback: progress_callback(progress_base, stages[current_stage][0])
            status, vocals_path, _ = separate_all_audio_under_folder(
                folder, model_name=demucs_model, device=device, progress=True, shifts=shifts
            )
            logger.info(f"Vocal separation complete: {vocals_path}")

            # Stage: ASR
            current_stage += 1; progress_base += stages[current_stage-1][1]
            if progress_callback: progress_callback(progress_base, stages[current_stage][0])
            status, result_json = transcribe_all_audio_under_folder(
                folder,
                asr_method=asr_method,
                whisper_model_name=whisper_model,
                device=device,
                batch_size=batch_size,
                diarization=diarization,
                min_speakers=_coerce_int_or_none(whisper_min_speakers),
                max_speakers=_coerce_int_or_none(whisper_max_speakers),
            )
            logger.info(f"ASR completed: {status}")

            # Stage: Translation
            current_stage += 1; progress_base += stages[current_stage-1][1]
            if progress_callback: progress_callback(progress_base, stages[current_stage][0])
            translation_target_language = _norm_translation_lang(translation_target_language)
            msg, summary, translation = translate_all_transcript_under_folder(
                folder, method=translation_method, target_language=translation_target_language
            )
            logger.info(f"Translation completed: {msg}")

            # Stage: TTS (with in-TTS emotion injection)
            current_stage += 1; progress_base += stages[current_stage-1][1]
            if progress_callback: progress_callback(progress_base, stages[current_stage][0])
            tts_target_language = _norm_tts_lang(tts_target_language)
            status, synth_path, _ = generate_all_wavs_under_folder(
                folder,
                method=tts_method,
                target_language=tts_target_language,
                voice=voice,
                emotion=emotion,
                emotion_strength=emotion_strength,
            )
            logger.info(f"TTS completed: {synth_path}")

            # Optional emotion batch pass (only if emotion == natural)
            try:
                if (emotion or "natural").strip().lower() == "natural":
                    _lang_hint = tts_target_language or "en"
                    ok, emsg = auto_tune_emotion_all_wavs_under_folder(
                        folder,
                        emotion="auto-neutral",
                        strength=float(emotion_strength),
                        lang_hint=_lang_hint,
                        win_s=10.0,
                        hop_s=9.0,
                        xfade_ms=int(os.getenv("HIGGS_TTS_XFADE_MS", "28")),
                        latency_budget_s=0.5,
                        min_confidence=0.50,
                        max_iters=2,
                    )
                    logger.info(f"Emotion (AUTO) shaping: {emsg}")
            except Exception as e:
                logger.warning(f"Emotion shaping step failed but continuing: {e}")

            # Stage: Video synthesis
            current_stage += 1; progress_base += stages[current_stage-1][1]
            if progress_callback: progress_callback(progress_base, stages[current_stage][0])
            status, output_video = synthesize_all_video_under_folder(
                folder,
                subtitles=subtitles,
                speed_up=speed_up,
                fps=fps,
                resolution=target_resolution,
                background_music=background_music,
                bgm_volume=bgm_volume,
                video_volume=video_volume,
            )
            if progress_callback: progress_callback(100, "Completed!")
            logger.info(f"Video composition completed: {output_video}")
            return True, output_video, "Success"

        except Exception as e:
            stack_trace = traceback.format_exc()
            title = info.get("title") if isinstance(info, dict) else info
            error_msg = f"Error while processing {title}: {e}\n{stack_trace}"
            logger.error(error_msg)
            if retry < max_retries - 1:
                logger.info(f"Retrying {retry + 2}/{max_retries}...")
            else:
                return False, None, error_msg

    return False, None, f"Max retries reached: {max_retries}"

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def do_everything(
    root_folder,
    url,
    num_videos=5,
    resolution="1080p",
    demucs_model="htdemucs_ft",
    device="auto",
    shifts=5,
    asr_method="Higgs",
    whisper_model="large",
    batch_size=32,
    diarization=False,
    whisper_min_speakers=None,
    whisper_max_speakers=None,
    translation_method="LLM",
    translation_target_language="zh-cn",
    tts_method="Higgs",
    tts_target_language="zh-cn",
    voice="zh-CN-XiaoxiaoNeural",
    subtitles=True,
    speed_up=1.00,
    fps=30,
    background_music=None,
    bgm_volume=0.5,
    video_volume=1.0,
    target_resolution="1080p",
    max_workers=3,
    max_retries=5,
    progress_callback=None,
    *,
    emotion: str = "natural",
    emotion_strength: float = 0.6,
    overwrite_duplicates: bool = True,     # default overwrite for this design
    author_override: Optional[str] = None,
):
    try:
        success_list, fail_list, error_details = [], [], []

        # Normalize input languages early (UI labels → codes)
        try:
            translation_target_language = _norm_translation_lang(translation_target_language)
            tts_target_language = _norm_tts_lang(tts_target_language)
        except Exception as e:
            logger.error(f"Language normalization error: {e}")
            return f"Language normalization error: {e}", None

        logger.info("-" * 50)
        logger.info(f"Starting job: {url}")
        logger.info(f"Output root={root_folder}, videos={num_videos}, download_res={resolution}")
        logger.info(f"ASR: {asr_method}/{whisper_model} | Demucs: {demucs_model} shifts={shifts} device={device}")
        logger.info(f"Translate → {translation_method} ({translation_target_language}) | TTS → {tts_method} ({tts_target_language}) voice={voice}")
        logger.info(f"Emotion preset={emotion} strength={emotion_strength:.2f} | overwrite_duplicates={overwrite_duplicates}")
        logger.info(f"Render: subtitles={subtitles} speed={speed_up} fps={fps} target_res={target_resolution}")
        logger.info("-" * 50)

        # Warm up
        if progress_callback:
            progress_callback(5, "Initializing models...")
        initialize_models(tts_method, asr_method, diarization)

        # URLs
        normalized = (url or "").replace(" ", "").replace(" ", "").replace("，", "\n").replace(",", "\n")
        urls = [u for u in normalized.split("\n") if u]
        out_video: Optional[str] = None

        # Local file path
        if url.endswith(".mp4"):
            success, output_video, error_msg = process_video(
                url,
                root_folder,
                resolution,
                demucs_model,
                device,
                shifts,
                asr_method,
                whisper_model,
                batch_size,
                diarization,
                whisper_min_speakers,
                whisper_max_speakers,
                translation_method,
                translation_target_language,
                tts_method,
                tts_target_language,
                voice,
                subtitles,
                speed_up,
                fps,
                background_music,
                bgm_volume,
                video_volume,
                target_resolution,
                max_retries,
                progress_callback,
                emotion=emotion,
                emotion_strength=emotion_strength,
                overwrite_duplicates=True,      # enforce overwrite
                author_override=author_override,
            )
            return ("Success", output_video) if success else (f"Failed: {error_msg}", None)

        # Remote URLs
        if progress_callback:
            progress_callback(10, "Fetching video info...")
        try:
            videos_info = list(get_info_list_from_url(urls, num_videos))
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Failed to get video list: {e}\n{stack_trace}")
            return f"Failed to get video list: {e}", None

        if not videos_info:
            return "Failed to retrieve video info. Please check the URL(s).", None

        for info in videos_info:
            try:
                success, output_video, error_msg = process_video(
                    info,
                    root_folder,
                    resolution,
                    demucs_model,
                    device,
                    shifts,
                    asr_method,
                    whisper_model,
                    batch_size,
                    diarization,
                    whisper_min_speakers,
                    whisper_max_speakers,
                    translation_method,
                    translation_target_language,
                    tts_method,
                    tts_target_language,
                    voice,
                    subtitles,
                    speed_up,
                    fps,
                    background_music,
                    bgm_volume,
                    video_volume,
                    target_resolution,
                    max_retries,
                    progress_callback,
                    emotion=emotion,
                    emotion_strength=emotion_strength,
                    overwrite_duplicates=True,  # enforce overwrite
                    author_override=author_override,
                )
                if success:
                    success_list.append(info)
                    out_video = output_video
                    logger.info(f"Processed: {info['title'] if isinstance(info, dict) else info}")
                else:
                    fail_list.append(info)
                    error_details.append(f"{info['title'] if isinstance(info, dict) else info}: {error_msg}")
                    logger.error(f"Failed: {error_msg}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                fail_list.append(info)
                error_details.append(f"{info['title'] if isinstance(info, dict) else info}: {e}")
                logger.error(f"Error: {e}\n{stack_trace}")

        logger.info("-" * 50)
        logger.info(f"Done. success={len(success_list)}, failed={len(fail_list)}")
        if error_details:
            logger.info("Failure details:")
            for detail in error_details:
                logger.info(f"  - {detail}")

        return f"Success: {len(success_list)}\nFailed: {len(fail_list)}", out_video

    except Exception as e:
        stack_trace = traceback.format_exc()
        error_msg = f"Pipeline error: {e}\n{stack_trace}"
        logger.error(error_msg)
        return error_msg, None
