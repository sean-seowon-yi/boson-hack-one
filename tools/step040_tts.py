# -*- coding: utf-8 -*-
import os
import re
import json
import librosa
import numpy as np
from loguru import logger

from .utils import save_wav, save_wav_norm
from .cn_tx import TextNorm
from audiostretchy.stretch import stretch_audio

# TTS backends (each must expose: tts(text, output_path, speaker_wav, ...))
from .step042_tts_xtts import tts as xtts_tts
from .step043_tts_cosyvoice import tts as cosyvoice_tts
from .step044_tts_edge_tts import tts as edge_tts
# NEW: Higgs/Boson TTS (OpenAI-compatible)
from .step041_tts_higgs import tts as higgs_tts  # ensure this file exists

normalizer = TextNorm()

def preprocess_text(text: str) -> str:
    # keep your prior behavior
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    # insert space between letters and digits
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text


def adjust_audio_length(
    wav_path: str,
    desired_length: float,
    sample_rate: int = 24000,
    min_speed_factor: float = 0.6,
    max_speed_factor: float = 1.1
):
    # Load (fallback to .mp3 if needed)
    try:
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    except Exception:
        if wav_path.endswith('.wav'):
            alt = wav_path.replace('.wav', '.mp3')
        elif wav_path.endswith('.mp3'):
            alt = wav_path
        else:
            alt = wav_path
        wav, sample_rate = librosa.load(alt, sr=sample_rate)

    current_length = len(wav) / sample_rate
    if current_length <= 1e-6:
        # avoid division by zero; return silence
        return np.zeros(0, dtype=np.float32), 0.0

    speed_factor = max(min(desired_length / current_length, max_speed_factor), min_speed_factor)
    logger.info(f"Speed Factor {speed_factor:.3f}")

    # output path for stretched version
    if wav_path.endswith('.wav'):
        target_path = wav_path.replace('.wav', '_adjusted.wav')
    elif wav_path.endswith('.mp3'):
        target_path = wav_path.replace('.mp3', '_adjusted.wav')
    else:
        target_path = wav_path + '_adjusted.wav'

    # stretch + reload
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)

    new_len = min(desired_length, len(wav) / sample_rate)
    return wav[:int(new_len * sample_rate)].astype(np.float32), new_len


# Language capability map per backend (UI enforces these)
tts_support_languages = {
    # XTTS: keep your surfaced set (XTTS supports many more; we keep parity with UI)
    'xtts':     ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish'],
    # EdgeTTS via Azure voices
    'EdgeTTS':  ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish'],
    # CosyVoice zero-shot markers (as you had)
    'cosyvoice':['中文', '粤语', 'English', 'Japanese', 'Korean', 'French'],
    # NEW: Higgs (voice cloning via reference) – support the same set you expose in UI
    'Higgs':    ['中文', 'English', 'Japanese', 'Korean', 'French', 'Spanish', 'Polish'],
}


def _synthesize_one_line(method: str, text: str, out_path: str, speaker_wav: str,
                         target_language: str, voice: str):
    """
    Dispatch to the selected backend. Backends write WAV to out_path.
    """
    if method == 'xtts':
        xtts_tts(text, out_path, speaker_wav, target_language=target_language)
    elif method == 'cosyvoice':
        cosyvoice_tts(text, out_path, speaker_wav, target_language=target_language)
    elif method == 'EdgeTTS':
        edge_tts(text, out_path, target_language=target_language, voice=voice)
    elif method == 'Higgs':
        # Higgs TTS (OpenAI-compatible). Reference speaker is optional but recommended.
        higgs_tts(text, out_path, speaker_wav)
    else:
        raise ValueError(f"Unknown TTS method: {method}")


def generate_wavs(method: str, folder: str, target_language: str = '中文', voice: str = 'zh-CN-XiaoxiaoNeural'):
    """
    Generate per-line WAVs and the combined track for one video's folder.

    RETURNS (strictly two values):
        (combined_wav_path, original_audio_path)
    """
    # Validate method & language support deterministically
    supported = tts_support_languages.get(method, [])
    if supported and target_language not in supported:
        raise ValueError(f"TTS method '{method}' does not support target language '{target_language}'")

    transcript_path = os.path.join(folder, 'translation.json')
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"translation.json not found in {folder}")

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    # Create output directory
    output_folder = os.path.join(folder, 'wavs')
    os.makedirs(output_folder, exist_ok=True)

    # Collect speakers (for info)
    speakers = {line.get('speaker', 'SPEAKER_00') for line in transcript}
    logger.info(f'Found {len(speakers)} speakers')

    # Build combined wav progressively
    full_wav = np.zeros((0,), dtype=np.float32)

    for i, line in enumerate(transcript):
        speaker = line.get('speaker', 'SPEAKER_00')
        text = preprocess_text(line.get('translation', '').strip())
        if not text:
            # If empty translation, keep timing with silence
            logger.warning(f'Empty translation for line {i}, inserting silence.')
            text = ""

        out_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')

        # Synthesize (idempotent: backends skip if out_path exists)
        _synthesize_one_line(method, text, out_path, speaker_wav, target_language, voice)

        # Timing adjustment
        start = float(line['start'])
        end = float(line['end'])
        length = max(0.0, end - start)
        last_end = len(full_wav) / 24000.0

        # Pad gap if any
        if start > last_end:
            pad_len = int((start - last_end) * 24000)
            if pad_len > 0:
                full_wav = np.concatenate((full_wav, np.zeros((pad_len,), dtype=np.float32)))

        # Update start to the current end of full_wav
        start = len(full_wav) / 24000.0
        line['start'] = start

        # Avoid overlap with next line
        if i < len(transcript) - 1:
            next_end = float(transcript[i + 1]['end'])
            end = min(start + length, next_end)

        # Stretch/crop synthesized line to fit the slot
        wav_seg, adj_len = adjust_audio_length(out_path, end - start)
        full_wav = np.concatenate((full_wav, wav_seg.astype(np.float32)))
        line['end'] = start + adj_len

    # Match energy with original vocals
    vocal_path = os.path.join(folder, 'audio_vocals.wav')
    if os.path.exists(vocal_path):
        vocal_wav, _sr = librosa.load(vocal_path, sr=24000)
        peak = float(np.max(np.abs(vocal_wav))) if vocal_wav.size else 1.0
        if peak > 0 and np.max(np.abs(full_wav)) > 0:
            full_wav = full_wav / np.max(np.abs(full_wav)) * peak

    # Save TTS-only track and write back timing updates
    tts_path = os.path.join(folder, 'audio_tts.wav')
    save_wav(full_wav, tts_path)
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # Mix with instruments
    inst_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(inst_path):
        instruments_wav, _sr = librosa.load(inst_path, sr=24000)
    else:
        instruments_wav = np.zeros_like(full_wav)

    # Length align
    len_full = len(full_wav)
    len_inst = len(instruments_wav)
    if len_full > len_inst:
        instruments_wav = np.pad(instruments_wav, (0, len_full - len_inst), mode='constant')
    elif len_inst > len_full:
        full_wav = np.pad(full_wav, (0, len_inst - len_full), mode='constant')

    combined = full_wav + instruments_wav
    combined_path = os.path.join(folder, 'audio_combined.wav')
    save_wav_norm(combined, combined_path)
    logger.info(f'Generated {combined_path}')

    # Return strictly two values (EXPECTED by callers)
    return combined_path, os.path.join(folder, 'audio.wav')


def generate_all_wavs_under_folder(root_folder: str, method: str,
                                   target_language: str = '中文',
                                   voice: str = 'zh-CN-XiaoxiaoNeural'):
    """
    Walk `root_folder`, generate TTS where needed.

    RETURNS (strictly three values):
        (status_text, combined_wav_path_or_None, original_audio_path_or_None)
    """
    wav_combined, wav_ori = None, None
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            # always EXACTLY two returns here
            wav_combined, wav_ori = generate_wavs(method, root, target_language, voice)
        elif 'audio_combined.wav' in files:
            wav_combined = os.path.join(root, 'audio_combined.wav')
            wav_ori = os.path.join(root, 'audio.wav')
            logger.info(f'Wavs already generated in {root}')

    return f'Generated all wavs under {root_folder}', wav_combined, wav_ori


if __name__ == '__main__':
    # Example quick test
    # folder = r'videos/ExampleUploader/20240805 Demo Video'
    # print(generate_wavs('xtts', folder))
    pass
