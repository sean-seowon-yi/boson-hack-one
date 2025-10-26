# -*- coding: utf-8 -*-
"""
TTS synthesis pipeline (per-line => stitched track)
- Language-aware text preprocessing (Chinese-only normalizations gated by target language)
- Backend dispatch to XTTS / CosyVoice / EdgeTTS / Higgs
- Precise timing via time-stretch with bounds
- Deterministic language support checks with unified language codes
"""

import os
import re
import json
import librosa
import numpy as np
from functools import lru_cache
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

# -----------------------
# Constants / globals
# -----------------------
SR = 24000
EPS = 1e-8  # tiny guard for divides
normalizer = TextNorm()

# Precompiled regexes
_RE_CAP_SPLIT = re.compile(r'(?<!^)([A-Z])')
_RE_ALNUM_GAP = re.compile(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])')

# -----------------------
# Unified language normalization
#   Accepts labels or codes; returns canonical codes:
#   'zh-cn','zh-tw','en','ko','ja','es','fr','pl'
# -----------------------
_LANG_ALIASES = {
    # Simplified Chinese
    "zh-cn": "zh-cn", "zh_cn": "zh-cn", "cn": "zh-cn",
    "chinese (中文)": "zh-cn", "chinese": "zh-cn", "中文": "zh-cn",
    "simplified chinese (简体中文)": "zh-cn", "simplified chinese": "zh-cn", "简体中文": "zh-cn",

    # Traditional Chinese
    "zh-tw": "zh-tw", "zh_tw": "zh-tw", "tw": "zh-tw",
    "traditional chinese (繁体中文)": "zh-tw", "traditional chinese": "zh-tw", "繁体中文": "zh-tw",

    # English
    "en": "en", "english": "en",

    # Korean
    "ko": "ko", "korean": "ko", "한국어": "ko",

    # Japanese
    "ja": "ja", "japanese": "ja", "日本語": "ja",

    # Spanish
    "es": "es", "spanish": "es", "español": "es",

    # French
    "fr": "fr", "french": "fr", "français": "fr",

    # Polish (XTTS supports it)
    "pl": "pl", "polish": "pl",
}

_ALLOWED_CODES = {"zh-cn", "zh-tw", "en", "ko", "ja", "es", "fr", "pl"}

@lru_cache(maxsize=128)
def normalize_lang_to_code(lang: str) -> str:
    if not lang:
        raise ValueError("target_language is empty/None")
    key = str(lang).strip().lower()
    code = _LANG_ALIASES.get(key, key)
    if code not in _ALLOWED_CODES:
        raise ValueError(f"Unrecognized/unsupported language: {lang} -> {code}")
    return code

def is_chinese_code(code: str) -> bool:
    return code in ("zh-cn", "zh-tw")


# -----------------------
# Preprocessing
# -----------------------
@lru_cache(maxsize=4096)
def preprocess_text(text: str, target_lang_code: str) -> str:
    """
    Minimal, language-aware text normalization.
    Only apply Chinese-specific rules when target is Chinese (zh-cn/zh-tw).
    """
    t = text or ""

    if is_chinese_code(target_lang_code):
        t = t.replace('AI', '人工智能')            # legacy preference
        t = _RE_CAP_SPLIT.sub(r' \1', t)         # split camel-case-ish caps
        t = normalizer(t)                        # Chinese text normalization

    # Language-agnostic: space between letters and digits
    t = _RE_ALNUM_GAP.sub(' ', t)
    return t


# -----------------------
# Time & audio helpers
# -----------------------
def adjust_audio_length(
    wav_path: str,
    desired_length: float,
    sample_rate: int = SR,
    min_speed_factor: float = 0.5,
    max_speed_factor: float = 1.2
):
    """
    Load synthesized audio (wav or mp3), time-stretch to fit desired_length,
    then crop to the exact slot if needed. Returns (audio, new_length_sec).
    """
    # Load (fallback to .mp3 if needed)
    try:
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    except Exception:
        alt = wav_path.replace('.wav', '.mp3') if wav_path.endswith('.wav') else wav_path
        wav, sample_rate = librosa.load(alt, sr=sample_rate)

    current_length = len(wav) / max(sample_rate, 1)
    if current_length <= 1e-6 or desired_length <= 0:
        return np.zeros(0, dtype=np.float32), 0.0

    speed_factor = max(min(desired_length / (current_length + EPS), max_speed_factor), min_speed_factor)
    logger.info(f"[TTS] stretch ratio={speed_factor:.3f}")

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

    new_len = min(desired_length, len(wav) / max(sample_rate, 1))
    return wav[:int(new_len * sample_rate)].astype(np.float32), new_len


# -----------------------
# Backend support map (codes)
# -----------------------
tts_support_languages = {
    # XTTS supports many; we keep a safe subset used in your project
    'xtts':      {'zh-cn', 'zh-tw', 'en', 'ja', 'ko', 'fr', 'pl', 'es'},
    # EdgeTTS: voices primarily determine exact locale, but these codes are fine as hints
    'EdgeTTS':   {'zh-cn', 'zh-tw', 'en', 'ja', 'ko', 'fr', 'es', 'pl'},
    # CosyVoice (common distributions): no Spanish/Polish typically
    'cosyvoice': {'zh-cn', 'zh-tw', 'en', 'ja', 'ko', 'fr'},
    # Higgs (per your notes): includes Spanish, French, etc.
    'Higgs':     {'zh-cn', 'zh-tw', 'en', 'ja', 'ko', 'fr', 'es'},
}

# If a backend needs a specific token instead of the unified code, adapt here.
_BACKEND_LANG_ADAPTER = {
    'xtts': {
        # XTTS is happy with codes as below (common TTS community convention)
        # Keeping identity mapping; override here if your xtts expects different tokens.
    },
    'EdgeTTS': {
        # EdgeTTS typically uses the voice to pick locale, but we pass the code for completeness.
        # Identity mapping is fine; voice wins in Edge backend.
    },
    'cosyvoice': {
        # Identity for supported codes; Cantonese not used here.
    },
    'Higgs': {
        # Higgs/OpenAI-compatible endpoints are fine with ISO-ish codes per your prior usage.
    }
}

def _adapt_lang_for_backend(method: str, code: str) -> str:
    # If adapter table has a mapping, use it; otherwise default to the code itself.
    table = _BACKEND_LANG_ADAPTER.get(method, {})
    return table.get(code, code)


# -----------------------
# Backend dispatcher
# -----------------------
def _synthesize_one_line(method: str, text: str, out_path: str, speaker_wav: str,
                         target_lang_code: str, voice: str):
    """
    Dispatch to the selected backend. Backends write WAV to out_path.
    target_lang_code is one of: 'zh-cn','zh-tw','en','ko','ja','es','fr','pl'
    """
    lang = _adapt_lang_for_backend(method, target_lang_code)

    if method == 'xtts':
        xtts_tts(text, out_path, speaker_wav, target_language=lang)
    elif method == 'cosyvoice':
        cosyvoice_tts(text, out_path, speaker_wav, target_language=lang)
    elif method == 'EdgeTTS':
        edge_tts(text, out_path, target_language=lang, voice=voice)
    elif method == 'Higgs':
        higgs_tts(text, out_path, speaker_wav, voice_type=voice, target_language=lang)
    else:
        raise ValueError(f"Unknown TTS method: {method}")


# -----------------------
# Small I/O helper
# -----------------------
def _atomic_write_json(path: str, obj):
    tmp = f"{path}.tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# -----------------------
# Main per-folder synthesis
# -----------------------
def generate_wavs(method: str, folder: str, target_language: str = "en", voice: str = 'zh-CN-XiaoxiaoNeural'):
    """
    Generate per-line WAVs and the combined track for one video's folder.

    RETURNS (strictly two values):
        (combined_wav_path, original_audio_path)
    """
    # Normalize & validate language for this backend (to code)
    lang_code = normalize_lang_to_code(target_language)
    supported = tts_support_languages.get(method, set())
    if supported and lang_code not in supported:
        raise ValueError(
            f"TTS method '{method}' does not support target language '{target_language}' "
            f"(normalized code='{lang_code}')"
        )

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
    logger.info(f'[TTS] Found {len(speakers)} speakers')

    # Build combined wav via chunk list to avoid repeated reallocations
    chunks: list[np.ndarray] = []
    current_time = 0.0  # in seconds

    for i, line in enumerate(transcript):
        speaker = line.get('speaker', 'SPEAKER_00')
        raw_text = (line.get('translation') or '').strip()

        if not raw_text:
            logger.warning(f'[TTS] Empty translation for line {i}, inserting silence.')
            text = ""
        else:
            text = preprocess_text(raw_text, lang_code)

        out_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')

        # Optional idempotency: skip synthesis if file already exists & non-empty
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 1024):
            _synthesize_one_line(method, text, out_path, speaker_wav, lang_code, voice)

        # Desired slot timing from transcript
        start = float(line['start'])
        end = float(line['end'])
        length = max(0.0, end - start)

        # Pad any gap between current timeline and desired start
        if start > current_time:
            pad_len = int((start - current_time) * SR)
            if pad_len > 0:
                chunks.append(np.zeros((pad_len,), dtype=np.float32))
                current_time = start

        # Avoid overlap with next line
        if i < len(transcript) - 1:
            next_start = float(transcript[i + 1]['start'])
            end = min(current_time + length, next_start)
        else:
            end = current_time + length

        # Stretch/crop synthesized line to fit the slot
        wav_seg, adj_len = adjust_audio_length(out_path, end - current_time, sample_rate=SR)
        chunks.append(wav_seg.astype(np.float32))

        # Write back updated timing
        line['start'] = current_time
        line['end'] = current_time + adj_len
        current_time = line['end']

    # Concatenate once
    full_wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    # Match energy with original vocals
    vocal_path = os.path.join(folder, 'audio_vocals.wav')
    if os.path.exists(vocal_path):
        vocal_wav, _sr = librosa.load(vocal_path, sr=SR)
        peak_vocal = float(np.max(np.abs(vocal_wav))) if vocal_wav.size else 1.0
        peak_tts = float(np.max(np.abs(full_wav))) if full_wav.size else 0.0
        if peak_vocal > 0 and peak_tts > 0:
            full_wav = full_wav / (peak_tts + EPS) * peak_vocal

    # Save TTS-only track and write back timing updates
    tts_path = os.path.join(folder, 'audio_tts.wav')
    save_wav(full_wav, tts_path)
    _atomic_write_json(transcript_path, transcript)

    # Mix with instruments
    inst_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(inst_path):
        instruments_wav, _sr = librosa.load(inst_path, sr=SR)
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
    logger.info(f'[TTS] Generated {combined_path}')

    # Return strictly two values (EXPECTED by callers)
    return combined_path, os.path.join(folder, 'audio.wav')


def generate_all_wavs_under_folder(root_folder: str, method: str,
                                   target_language: str = 'en',
                                   voice: str = 'zh-CN-XiaoxiaoNeural'):
    """
    Walk `root_folder`, generate TTS where needed.

    RETURNS (strictly three values):
        (status_text, combined_wav_path_or_None, original_audio_path_or_None)
    """
    wav_combined, wav_ori = None, None
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            wav_combined, wav_ori = generate_wavs(method, root, target_language, voice)
        elif 'audio_combined.wav' in files:
            wav_combined = os.path.join(root, 'audio_combined.wav')
            wav_ori = os.path.join(root, 'audio.wav')
            logger.info(f'[TTS] Wavs already generated in {root}')

    return f'Generated all wavs under {root_folder}', wav_combined, wav_ori


if __name__ == '__main__':
    # Example quick test
    # folder = r'videos/ExampleUploader/20240805 Demo Video'
    # print(generate_wavs('xtts', folder))
    pass
