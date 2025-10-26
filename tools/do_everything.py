# -*- coding: utf-8 -*-
"""
tools/do_everything.py

End-to-end pipeline with post-TTS Emotion control automated by
Higgs-understanding (windowed + crossfaded), using ONLY the auto batch.

UI values supported:
  - "natural"  -> skip emotion shaping
  - "happy"    -> treated as "auto-happy"
  - "sad"      -> treated as "auto-sad"
  - "angry"    -> treated as "auto-angry"
  - "auto-*"   -> respected as-is (e.g., "auto-happy", "auto-sad", "auto-angry")

Requires:
  tools/step045_emotion_auto_batch.py
"""

import json
import os
import time
import traceback
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
# from .step021_asr_whisperx import init_whisperx, init_diarize
from .step022_asr_funasr import init_funasr
from .step030_translation import translate_all_transcript_under_folder
from .step040_tts import generate_all_wavs_under_folder
from .step042_tts_xtts import init_TTS
from .step043_tts_cosyvoice import init_cosyvoice
from .step050_synthesize_video import synthesize_all_video_under_folder

# ONLY import the auto emotion batch
from .step047_emotion_auto_batch import auto_tune_emotion_all_wavs_under_folder

# Track which heavy models were initialized (process lifetime)
models_initialized = {
    "demucs": False,
    "xtts": False,
    "cosyvoice": False,
    "diarize": False,
    "funasr": False,
    # Higgs ASR/TTS are API-based; kept out of init gating intentionally
}

# === UI → internal normalization (keep EXACTLY these UI labels) ===
_UI_TO_TRANSLATION_LANG = {
    "Simplified Chinese (简体中文)": "简体中文",
    "Traditional Chinese (繁体中文)": "繁体中文",
    "English": "English",
    "Korean": "Korean",
    "Spanish": "Spanish",
}

# For TTS, dropdown is: ["Chinese (中文)", "English", "Korean", "Spanish", "French"]
_UI_TO_TTS_LANG = {
    "Chinese (中文)": "中文",
    "English": "English",
    "Korean": "Korean",
    "Spanish": "Spanish",
    "French": "French",
}

def _norm_translation_lang(ui_label: str) -> str:
    return _UI_TO_TRANSLATION_LANG.get(ui_label, ui_label)

def _norm_tts_lang(ui_label: str) -> str:
    return _UI_TO_TTS_LANG.get(ui_label, ui_label)

def _coerce_int_or_none(x):
    if x in (None, "", "None"):
        return None
    try:
        return int(x)
    except Exception:
        return None

def get_available_gpu_memory() -> float:
    """Return available GPU memory in GiB (0 if CUDA is unavailable or an error occurs)."""
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            used = torch.cuda.memory_allocated(0)
            return (total - used) / (1024 ** 3)
        return 0.0
    except Exception:
        return 0.0


def initialize_models(tts_method: str, asr_method: str, diarization: bool) -> None:
    """
    Initialize required models exactly once per process.
    Uses a thread pool for parallel cold-start, then waits for completion.
    """
    global models_initialized
    futures = []

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Demucs
            if not models_initialized["demucs"]:
                futures.append(executor.submit(init_demucs))
                models_initialized["demucs"] = True
                logger.info("Initialized Demucs")
            else:
                logger.info("Demucs already initialized — skipping")

            # TTS
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
                # API-based; nothing to init locally
                logger.info("TTS 'Higgs' selected — API-based")

            # ASR (local initializers when applicable)
            # if asr_method == "WhisperX":
            #     if not models_initialized["whisperx"]:
            #         futures.append(executor.submit(init_whisperx))
            #         models_initialized["whisperx"] = True
            #         logger.info("Initialized WhisperX")
            #     if diarization and not models_initialized["diarize"]:
            #         futures.append(executor.submit(init_diarize))
            #         models_initialized["diarize"] = True
            #         logger.info("Initialized diarization")
            if asr_method == "FunASR":
                if not models_initialized["funasr"]:
                    futures.append(executor.submit(init_funasr))
                    models_initialized["funasr"] = True
                    logger.info("Initialized FunASR")
            elif asr_method == "Higgs":
                # API-based; no local model to init
                logger.info("ASR 'Higgs' selected — API-based, no local initialization required")

            # Ensure any init exception gets raised here
            for fut in futures:
                fut.result()

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Failed to initialize models: {e}\n{stack_trace}")
        # Reset flags to allow retry and free any partially loaded state
        models_initialized = {k: False for k in models_initialized}
        release_model()
        raise


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
    progress_callback=None,
    *,
    emotion: str = "natural",
    emotion_strength: float = 0.6,
):
    """
    Process a single video end-to-end with optional progress callback.

    progress_callback(progress_percent: int, status_message: str) -> None
    """
    # Progress stages: (label, weight_total_percent)
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
                # Local file mode: place it under <root_folder>/<basename>/download.mp4
                import shutil
                original_file_name = os.path.basename(info)
                folder_name = os.path.splitext(original_file_name)[0]
                folder = os.path.join(root_folder, folder_name)
                os.makedirs(folder, exist_ok=True)
                dest_path = os.path.join(folder, "download.mp4")
                shutil.copy(info, dest_path)
            else:
                folder = get_target_folder(info, root_folder)
                if folder is None:
                    error_msg = f'Unable to derive target folder: {info.get("title") if isinstance(info, dict) else info}'
                    logger.warning(error_msg)
                    return False, None, error_msg

                folder = download_single_video(info, root_folder, resolution)
                if folder is None:
                    error_msg = f'Download failed: {info.get("title") if isinstance(info, dict) else info}'
                    logger.warning(error_msg)
                    return False, None, error_msg

            logger.info(f"Processing video folder: {folder}")

            # Stage: Vocal separation
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                status, vocals_path, _ = separate_all_audio_under_folder(
                    folder, model_name=demucs_model, device=device, progress=True, shifts=shifts
                )
                logger.info(f"Vocal separation complete: {vocals_path}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f"Vocal separation failed: {e}\n{stack_trace}"
                logger.error(error_msg)
                return False, None, error_msg

            # Stage: ASR
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                # Coerce radios to int/None if needed
                whisper_min_speakers_c = _coerce_int_or_none(whisper_min_speakers)
                whisper_max_speakers_c = _coerce_int_or_none(whisper_max_speakers)

                status, result_json = transcribe_all_audio_under_folder(
                    folder,
                    asr_method=asr_method,
                    whisper_model_name=whisper_model,  # ignored by Higgs path if implemented that way
                    device=device,
                    batch_size=batch_size,
                    diarization=diarization,
                    min_speakers=whisper_min_speakers_c,
                    max_speakers=whisper_max_speakers_c,
                )
                logger.info(f"ASR completed: {status}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f"ASR failed: {e}\n{stack_trace}"
                logger.error(error_msg)
                return False, None, error_msg

            # Stage: Translation
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                # Normalize translation language label
                translation_target_language = _norm_translation_lang(translation_target_language)
                msg, summary, translation = translate_all_transcript_under_folder(
                    folder, method=translation_method, target_language=translation_target_language
                )
                logger.info(f"Translation completed: {msg}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f"Translation failed: {e}\n{stack_trace}"
                logger.error(error_msg)
                return False, None, error_msg

            # Stage: TTS
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                # Normalize TTS language label
                tts_target_language = _norm_tts_lang(tts_target_language)
                status, synth_path, _ = generate_all_wavs_under_folder(
                    folder, method=tts_method, target_language=tts_target_language, voice=voice
                )
                logger.info(f"TTS completed: {synth_path}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f"TTS failed: {e}\n{stack_trace}"
                logger.error(error_msg)
                return False, None, error_msg

            # NEW Stage: Emotion shaping (auto via Higgs-understanding)
            try:
                # Map "happy"|"sad"|"angry" to "auto-happy"|... ; keep "natural" as skip
                _emotion = (emotion or "natural").strip().lower()
                if _emotion in ("happy", "sad", "angry"):
                    _emotion = f"auto-{_emotion}"

                if _emotion.startswith("auto"):
                    _lang_hint = tts_target_language or "en"
                    ok, emsg = auto_tune_emotion_all_wavs_under_folder(
                        folder,
                        emotion=_emotion,                     # "auto-happy"/"auto-sad"/"auto-angry"/"auto"
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
                else:
                    logger.info("Emotion preset is natural — skipping.")
            except Exception as e:
                logger.warning(f"Emotion shaping step failed but continuing: {e}")

            # Stage: Synthesis (video)
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
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
                logger.info(f"Video composition completed: {output_video}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f"Video composition failed: {e}\n{stack_trace}"
                logger.error(error_msg)
                return False, None, error_msg

            # Done
            if progress_callback:
                progress_callback(100, "Completed!")
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


def do_everything(
    root_folder,
    url,
    num_videos=5,
    resolution="1080p",
    demucs_model="htdemucs_ft",
    device="auto",
    shifts=5,
    asr_method="Higgs",          # <-- matches UI default
    whisper_model="large",
    batch_size=32,
    diarization=False,
    whisper_min_speakers=None,
    whisper_max_speakers=None,
    translation_method="LLM",
    translation_target_language="Simplified Chinese (简体中文)",  # <-- exact UI label
    tts_method="Higgs",          # <-- matches UI default
    tts_target_language="Chinese (中文)",                         # <-- exact UI label
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
    emotion: str = "natural",          # "natural" | "happy" | "sad" | "angry" | "auto-*" | "auto"
    emotion_strength: float = 0.6,     # 0..1
):
    """
    Full pipeline entrypoint with an optional progress callback.

    Returns:
        (summary_text: str, last_output_video_path: Optional[str])
    """
    try:
        success_list = []
        fail_list = []
        error_details = []

        logger.info("-" * 50)
        logger.info(f"Starting job: {url}")
        logger.info(f"Output folder={root_folder}, videos={num_videos}, download_res={resolution}")
        logger.info(f"Vocal separation: model={demucs_model}, device={device}, shifts={shifts}")
        logger.info(f"ASR: method={asr_method}, model={whisper_model}, batch_size={batch_size}, diarization={diarization}")
        logger.info(f"Translate: method={translation_method}, target_lang={translation_target_language}")
        logger.info(f"TTS: method={tts_method}, target_lang={tts_target_language}, voice={voice}")
        logger.info(f"Emotion(AUTO): preset={emotion}, strength={emotion_strength:.2f}")
        logger.info(f"Video compose: subtitles={subtitles}, speed={speed_up}, FPS={fps}, render_res={target_resolution}")
        logger.info("-" * 50)

        # Normalize multiline URL list; allow comma/Chinese comma separators
        normalized = (url or "").replace(" ", "").replace("，", "\n").replace(",", "\n")
        urls = [u for u in normalized.split("\n") if u]

        # Warm up models once
        try:
            if progress_callback:
                progress_callback(5, "Initializing models...")
            initialize_models(tts_method, asr_method, diarization)
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Model initialization failed: {e}\n{stack_trace}")
            return f"Model initialization failed: {e}", None

        out_video: Optional[str] = None

        # Local file convenience: handle a single .mp4 path
        if url.endswith(".mp4"):
            try:
                success, output_video, error_msg = process_video(
                    url,  # pass the actual file path
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
                    # NEW
                    emotion=emotion,
                    emotion_strength=emotion_strength,
                )

                if success:
                    logger.info(f"Local video processed successfully: {url}")
                    return "Success", output_video
                else:
                    logger.error(f"Local video failed: {url}, error: {error_msg}")
                    return f"Failed: {error_msg}", None

            except Exception as e:
                stack_trace = traceback.format_exc()
                logger.error(f"Failed to process local video: {e}\n{stack_trace}")
                return f"Failed to process local video: {e}", None

        # Remote URLs
        try:
            videos_info = []
            if progress_callback:
                progress_callback(10, "Fetching video info...")

            for video_info in get_info_list_from_url(urls, num_videos):
                videos_info.append(video_info)

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
                        # NEW
                        emotion=emotion,
                        emotion_strength=emotion_strength,
                    )

                    if success:
                        success_list.append(info)
                        out_video = output_video
                        logger.info(f"Processed: {info['title'] if isinstance(info, dict) else info}")
                    else:
                        fail_list.append(info)
                        error_details.append(
                            f"{info['title'] if isinstance(info, dict) else info}: {error_msg}"
                        )
                        logger.error(
                            f"Failed: {info['title'] if isinstance(info, dict) else info}, error: {error_msg}"
                        )
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    fail_list.append(info)
                    error_details.append(
                        f"{info['title'] if isinstance(info, dict) else info}: {e}"
                    )
                    logger.error(
                        f"Error: {info['title'] if isinstance(info, dict) else info}, error: {e}\n{stack_trace}"
                    )
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Failed to get video list: {e}\n{stack_trace}")
            return f"Failed to get video list: {e}", None

        # Summary
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
