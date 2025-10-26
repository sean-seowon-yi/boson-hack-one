# -*- coding: utf-8 -*-
"""
tools.step033_translation_translator
Thin wrapper around `translators` with retries, language label normalization,
and small caching. API unchanged:

    translator_response(messages, to_language='zh-CN', translator_server='bing') -> str
"""

from __future__ import annotations

import os
from functools import lru_cache
import translators as ts  # pip: translators
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Normalize user-friendly labels to ISO-ish codes used by translators
# Restricted to: zh-CN, en, ko, es, fr
_LANG_MAP = {
    # Simplified Chinese
    '中文': 'zh-CN', '简体中文': 'zh-CN', 'chinese': 'zh-CN', 'zh': 'zh-CN', 'zh-cn': 'zh-CN',

    # English
    'English': 'en', 'english': 'en', 'en': 'en',

    # Korean
    '한국어': 'ko', 'korean': 'ko', 'ko': 'ko',

    # Spanish
    'español': 'es', 'spanish': 'es', 'es': 'es',

    # French
    'français': 'fr', 'french': 'fr', 'fr': 'fr',
}

def _norm_lang(label: str) -> str:
    if not label:
        return 'zh-CN'
    l = label.strip()
    return _LANG_MAP.get(l, _LANG_MAP.get(l.lower(), l))

@lru_cache(maxsize=8192)
def _translate_once(text: str, to_language: str, server: str) -> str:
    # Use translators with explicit auto source detection
    return ts.translate_text(
        query_text=text,
        translator=server,
        from_language='auto',
        to_language=to_language
    )

def translator_response(messages, to_language: str = 'zh-CN', translator_server: str = 'bing') -> str:
    """
    messages: str (kept as your original usage)
    """
    to_language = _norm_lang(to_language)
    server = (translator_server or 'bing').strip().lower()

    # Single-shot fast path with small fallback cascade (bing -> google)
    for attempt in range(3):
        try:
            out = _translate_once(str(messages), to_language, server).strip()
            return out
        except Exception as e:
            logger.info(f'[MT] translate failed on {server} (attempt {attempt+1}): {e}')
            if attempt == 0:
                server = 'google' if server != 'google' else 'bing'
            continue

    logger.warning("[MT] translation fell back to input text due to repeated failures.")
    return str(messages)

if __name__ == '__main__':
    print(translator_response('Hello, how are you?', '中文', 'bing'))
    print(translator_response('你好，最近怎么样？', 'en', 'google'))
