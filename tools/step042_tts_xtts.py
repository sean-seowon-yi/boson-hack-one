# -*- coding: utf-8 -*-
import os
from TTS.api import TTS
from loguru import logger
import numpy as np
import torch
import time
from .utils import save_wav

model = None

"""
Supported by XTTS-v2 (subset commonly used):
Arabic: ar, Brazilian Portuguese: pt, Mandarin Chinese: zh-cn,
Czech: cs, Dutch: nl, English: en, French: fr, German: de,
Italian: it, Polish: pl, Russian: ru, Spanish: es, Turkish: tr,
Japanese: ja, Korean: ko, Hungarian: hu, Hindi: hi
"""

# -----------------------------
# Unified language normalization
# -----------------------------
# Accept labels OR codes; return canonical code
_LANG_ALIASES = {
    # Chinese
    "zh-cn": "zh-cn", "zh_cn": "zh-cn", "cn": "zh-cn",
    "chinese (中文)": "zh-cn", "chinese": "zh-cn", "中文": "zh-cn",
    "simplified chinese (简体中文)": "zh-cn", "simplified chinese": "zh-cn", "简体中文": "zh-cn",
    # Allow Traditional input but map to zh-cn (XTTS lacks explicit zh-tw token)
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

    # German
    "de": "de", "german": "de", "deutsch": "de",

    # Italian
    "it": "it", "italian": "it",

    # Portuguese
    "pt": "pt", "portuguese": "pt",

    # Polish
    "pl": "pl", "polish": "pl",

    # Russian
    "ru": "ru", "russian": "ru",

    # Turkish
    "tr": "tr", "turkish": "tr",

    # Hungarian
    "hu": "hu", "hungarian": "hu",

    # Hindi
    "hi": "hi", "hindi": "hi",
}

# Exact set XTTS accepts (codes)
_XTTS_ALLOWED = {
    'ar', 'pt', 'zh-cn', 'cs', 'nl', 'en', 'fr', 'de',
    'it', 'pl', 'ru', 'es', 'tr', 'ja', 'ko', 'hu', 'hi'
}

# Env flag: allow reading ASCII-English text with non-English language without forcing
_ALLOW_MISMATCH = bool(int(os.getenv("XTTS_ALLOW_MISMATCH", "0")))


def _canon(s: str) -> str:
    return "" if s is None else str(s).strip().lower()


def _normalize_to_code(lang: str) -> str:
    """
    Return canonical code from labels/codes; allow 'zh-tw' input but adapt to 'zh-cn' for XTTS.
    """
    key = _canon(lang)
    code = _LANG_ALIASES.get(key, key or "en")
    # XTTS does not expose 'zh-tw' token; fall back to 'zh-cn' while logging
    if code == "zh-tw":
        logger.warning("[XTTS] No explicit 'zh-tw' support; using 'zh-cn' for synthesis.")
        code = "zh-cn"
    if code not in _XTTS_ALLOWED:
        raise ValueError(f"[XTTS] Unsupported language: {lang} -> {code}")
    return code


def _looks_ascii_english(text: str) -> bool:
    if not text:
        return False
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return any(c.isalpha() for c in text)


def init_TTS():
    load_model()


def load_model(model_path="models/TTS/XTTS-v2", device='auto'):
    global model
    if model is not None:
        return

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Loading TTS model from {model_path}')
    t_start = time.time()
    if os.path.isdir(model_path):
        print(f"Loading TTS model from {model_path}")
        model = TTS(
            model_path=model_path,
            config_path=os.path.join(model_path, 'config.json'),
        ).to(device)
    else:
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    t_end = time.time()
    logger.info(f'TTS model loaded in {t_end - t_start:.2f}s')


def tts(
    text,
    output_path,
    speaker_wav,
    model_name="models/TTS/XTTS-v2",
    device='auto',
    target_language='en'  # safer default than Chinese
):
    """
    Synthesize `text` with XTTS to `output_path`, cloning timbre from `speaker_wav`.
    `target_language` can be a UI label or a code; it will be normalized to a code for XTTS.
    """
    global model

    # Normalize language to code + optional ASCII guard
    try:
        lang_code = _normalize_to_code(target_language)
    except Exception as e:
        logger.error(str(e))
        raise

    if not _ALLOW_MISMATCH and lang_code != "en" and _looks_ascii_english(text or ""):
        logger.warning(f"[XTTS] ASCII-looking text but lang={lang_code}; forcing 'en'. "
                       f"Set XTTS_ALLOW_MISMATCH=1 to disable.")
        lang_code = "en"

    assert lang_code in _XTTS_ALLOWED, f"[XTTS] Language code not allowed: {lang_code}"

    if os.path.exists(output_path):
        logger.info(f'TTS output for "{text}" already exists.')
        return

    if model is None:
        load_model(model_name, device)

    for retry in range(3):
        try:
            wav = model.tts(text or "", speaker_wav=speaker_wav, language=lang_code)
            wav = np.array(wav, dtype=np.float32)
            save_wav(wav, output_path)
            logger.info(f'TTS synthesis succeeded (lang={lang_code}) for: {text}')
            break
        except Exception as e:
            logger.warning(f'TTS synthesis failed (attempt {retry+1}/3) for: {text}')
            logger.warning(e)
            if retry == 2:
                raise


if __name__ == '__main__':
    pass
