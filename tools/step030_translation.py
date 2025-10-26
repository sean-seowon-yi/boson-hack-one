# -*- coding: utf-8 -*-
"""
Step030 — Translation pipeline (robust + language-aware + enforcement) [streamlined]

Goal: Proper translation + storage to JSON quickly, without breaking existing usage.

Key tweaks:
- Early skip for non-speech tokens (e.g., "[LAUGHTER]") to avoid wasted calls.
- Normalized de-dup (spacing/case) so repeated lines translate once.
- Optional FAST mode to prefer MT path automatically (env toggle; default off).
- Parallel MT path preserved; safer caching; tighter sleeps/backoff.
- Stricter "absolute translation" enforcement (rejects same-language paraphrases) with smart relaxations.
- Progressive validation (strict → relaxed) + faster MT fallback.
- Atomic writes for JSON outputs.
- NEW: Strip <t>...</t> wrappers from all final outputs (no <t> in translation.json).

Public APIs preserved:
    summarize(...)
    translate(...)
    translate_all_transcript_under_folder(...)
"""

from __future__ import annotations

import json
import os
import re
import time
import string
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from loguru import logger

# Backends (keep your existing modules/paths)
from tools.step032_translation_llm import llm_response
from tools.step033_translation_translator import translator_response

load_dotenv()

# ============================================================
# Tunables (perf+behavior knobs; defaults conservative)
# ============================================================
ENABLE_BACKTRANSLATE_VERIFY = os.getenv("TRANSLATION_BACKTRANSLATE_VERIFY", "0") == "1"
ENABLE_DEDUP_SAME_LINES = os.getenv("TRANSLATION_DEDUP", "1") == "1"
MT_MAX_WORKERS = max(1, int(os.getenv("TRANSLATION_MT_MAX_WORKERS", "4")))  # only used on MT path
RETRY_SLEEP_S = float(os.getenv("TRANSLATION_RETRY_SLEEP", "0.2"))
SMALL_SLEEP_S = float(os.getenv("TRANSLATION_SMALL_SLEEP", "0.03"))
LLM_MAX_RETRIES = max(1, int(os.getenv("TRANSLATION_LLM_MAX_RETRIES", "3")))  # default 3
LLM_HISTORY_WINDOW = max(12, int(os.getenv("TRANSLATION_LLM_HISTORY_WINDOW", "14")))
SUMMARY_TEXT_LIMIT = max(800, int(os.getenv("TRANSLATION_SUMMARY_TEXT_LIMIT", "1600")))
FAST_TRANSLATION_MODE = os.getenv("TRANSLATION_FAST_MODE", "0") == "1"  # prefer MT path automatically

# Non-speech pattern (skip heavy translation path)
_NON_SPEECH = re.compile(
    r'^\s*\[(?:music|applause|laughter|silent|silence|noise|beat|pause|inaudible|coughs|cough|breath|breathing)[^\]]*\]\s*$',
    re.I
)

# ============================================================
# Precompiled regexes
# ============================================================
_RE_FW_PARENS = re.compile(r'\（[^）]*\）')
_RE_NUM_COMMA = re.compile(r'(?<=\d),(?=\d)')
_RE_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_RE_PREFIXES = [
    re.compile(pat, re.IGNORECASE | re.DOTALL)
    for pat in [
        r'^\s*translated\s*text\s*:\s*(.+)$',
        r'^\s*translation\s*:\s*(.+)$',
        r'^\s*译文\s*[:：]\s*(.+)$',
        r'^\s*翻译\s*[:：]\s*(.+)$',
        r'^\s*resultado\s*[:：]\s*(.+)$',
        r'^\s*traducci[oó]n\s*[:：]\s*(.+)$',
    ]
]
_RE_CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
_RE_HIRA = re.compile(r'[\u3040-\u309f]')
_RE_KATA = re.compile(r'[\u30a0-\u30ff]')
_RE_HANG = re.compile(r'[\uac00-\ud7af]')
_RE_LATN = re.compile(r'[A-Za-z]')
_PUNC_TABLE = str.maketrans('', '', string.punctuation + "，。！？；：、“”‘’—…《》·")

# Strip <t>...</t> safely (NEW)
_RE_T_WRAPPER = re.compile(r'^\s*<t\s*>(.*?)</t\s*>\s*$', re.IGNORECASE | re.DOTALL)
_RE_T_TAGS    = re.compile(r'</?t\s*>', re.IGNORECASE)

def _strip_t_tags(s: str) -> str:
    if not s:
        return s
    m = _RE_T_WRAPPER.match(s)
    if m:
        return m.group(1).strip()
    # If it's not a perfect single wrapper, remove any loose <t> / </t> occurrences
    return _RE_T_TAGS.sub('', s).strip()

# For CN sentence splitting
_RE_CN_SPLIT_1 = re.compile(r'([。！？\?])([^，。！？\?”’》])')
_RE_CN_SPLIT_2 = re.compile(r'(\.{6})([^，。！？\?”’》])')  # ......
_RE_CN_SPLIT_3 = re.compile(r'(\…{2})([^，。！？\?”’》])')   # ……
_RE_CN_SPLIT_4 = re.compile(r'([。！？\?][”’])([^ ，。！？\?”’》])')
_RE_LAT_SPLIT = re.compile(r'(?<=[.!?])\s+')

# ============================================================
# Utilities & small helpers
# ============================================================
def get_necessary_info(info: dict) -> dict:
    return {
        'title': info.get('title', ''),
        'uploader': info.get('uploader', ''),
        'description': info.get('description', ''),
        'upload_date': info.get('upload_date', ''),
        'tags': info.get('tags', []),
    }

def ensure_transcript_length(transcript: str, max_length: int = 4000) -> str:
    if len(transcript) <= max_length:
        return transcript
    mid = len(transcript) // 2
    half = max_length // 2
    return transcript[:mid][:half] + transcript[mid:][-half:]

def _is_chinese_target(lang: str) -> bool:
    lang = (lang or "").lower()
    return any(k in lang for k in ["zh", "简体", "繁体", "中文", "chinese"])

def translation_postprocess(result: str, target_language: str = "简体中文") -> str:
    result = (result or "").strip()
    result = _strip_t_tags(result)  # ensure <t> never survives
    result = _RE_FW_PARENS.sub('', result)
    result = _RE_NUM_COMMA.sub('', result)
    result = result.replace('²', '^2')
    if _is_chinese_target(target_language):
        result = (result
                  .replace('...', '，')
                  .replace('————', '：')
                  .replace('——', '：')
                  .replace('°', '度')
                  .replace('变压器', 'Transformer')
                  .replace('AI', '人工智能'))
    return result

def _extract_first_json_object(text: str) -> dict:
    if not text:
        raise ValueError("Empty text")
    m = _RE_JSON_FENCE.search(text)
    if m:
        return json.loads(m.group(1).strip())

    # Brace-balance scan
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = text.find("{", start + 1)
    raise ValueError("No valid JSON object found in text")

def _pluck_translation_payload(raw: str) -> str:
    if not raw:
        return ""
    t = raw.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    try:
        obj = _extract_first_json_object(t)
        for key in ("translation", "译文", "resultado", "traducción", "traduccion"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return _strip_t_tags(val.strip())  # strip <t> here
    except Exception:
        pass
    for rex in _RE_PREFIXES:
        m = rex.match(t)
        if m:
            t = m.group(1).strip()
            break
    m = (re.search(r'“([^”]+)”', t) or
         re.search(r'"([^"]+)"', t) or
         re.search(r'‘([^’]+)’', t) or
         re.search(r"'([^']+)'", t))
    if m and len(m.group(1).strip()) >= 1:
        return _strip_t_tags(m.group(1).strip())  # strip <t> here
    wrappers = ['“', '”', '"', '‘', '’', "'", '《', '》', '「', '」', '『', '』']
    while len(t) >= 2 and t[0] in wrappers and t[-1] in wrappers:
        t = t[1:-1].strip()
    return _strip_t_tags(t.strip())  # strip <t> here

# ============================================================
# Language normalization & detection
# ============================================================
def _norm_lang_label(label: str) -> str:
    if not label:
        return "unknown"
    s = label.strip().lower()
    mapping = {
        "chinese": "zh", "simplified chinese": "zh", "zh": "zh", "zh-cn": "zh", "zh_cn": "zh",
        "简体中文": "zh", "中文": "zh",
        "english": "en", "en": "en", "en-us": "en", "en_gb": "en",
        "japanese": "ja", "ja": "ja", "日本語": "ja",
        "korean": "ko", "ko": "ko", "韩国语": "ko", "한국어": "ko",
        "spanish": "es", "es": "es", "español": "es",
        "french": "fr", "fr": "fr", "français": "fr",
    }
    return mapping.get(s, "unknown")

def _heuristic_lang(text: str) -> str:
    t = text or ""
    cjk = len(_RE_CJK.findall(t))
    hira = len(_RE_HIRA.findall(t))
    kata = len(_RE_KATA.findall(t))
    hang = len(_RE_HANG.findall(t))
    latin = len(_RE_LATN.findall(t))
    if (hira + kata) > 0:
        return "ja"
    if hang > 0:
        return "ko"
    if cjk > 0 and (hira + kata + hang) == 0:
        return "zh"
    if latin > 0 and (cjk + hira + kata + hang) == 0:
        return "en"
    return "unknown"

try:
    import cld3  # type: ignore
    def _detect_lang(text: str) -> str:
        # heuristic first (cheap), CLD3 if needed
        h = _heuristic_lang(text)
        if h != "unknown":
            return h
        res = cld3.get_language(text or "")
        if res and res.language:
            code = res.language.lower()
            if code.startswith("zh"): return "zh"
            if code.startswith("en"): return "en"
            if code.startswith("ja"): return "ja"
            if code.startswith("ko"): return "ko"
            if code.startswith("es"): return "es"
            if code.startswith("fr"): return "fr"
            if code.startswith("pl"): return "pl"
        return h
except Exception:
    def _detect_lang(text: str) -> str:
        return _heuristic_lang(text)

# ============================================================
# Similarity / overlap guards
# ============================================================
def _token_set(s: str) -> set:
    s = (s or "").lower().translate(_PUNC_TABLE)
    return set(s.split())

def _too_similar_to_source(src: str, tgt: str, threshold: float = 0.92) -> bool:
    ts, tt = _token_set(src), _token_set(tgt)
    if not ts or not tt:
        return False
    overlap = len(ts & tt) / max(1, len(ts | tt))
    return overlap >= threshold

# ============================================================
# Tiny / numeric inputs helpers
# ============================================================
_MICRO_MAX = 3
_RE_NUMERICISH = re.compile(r'^[\d\W_]+$')  # digits/punct/underscore only (no letters)

def _is_micro_utterance(s: str) -> bool:
    return len((s or "").strip()) <= _MICRO_MAX

def _is_numericish(s: str) -> bool:
    return bool(_RE_NUMERICISH.fullmatch((s or "").strip()))

# ============================================================
# Back-translation verification (optional)
# ============================================================
def _verify_by_backtranslation(src_text: str, tgt_text: str, target_language: str) -> bool:
    # Skip noisy verification for tiny/numeric content
    if _is_micro_utterance(src_text) or _is_numericish(src_text):
        return True
    try:
        src_code = _detect_lang(src_text)
        src_label = {
            "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
            "es": "Spanish", "fr": "French"
        }.get(src_code, "English")
        bt = translator_response(tgt_text, to_language=src_label, translator_server='google')
        ts, tb = _token_set(src_text), _token_set(bt)
        if not ts or not tb:
            return True
        jacc = len(ts & tb) / max(1, len(ts | tb))
        return jacc >= 0.25
    except Exception:
        return True

# ============================================================
# Validation — enforces absolute translation (with progressive strictness)
# ============================================================
def valid_translation(
    text: str,
    translation: str,
    target_language: str = "简体中文",
    *,
    strict: bool = True
) -> Tuple[bool, str]:
    t = _pluck_translation_payload(translation)
    if not t:
        return False, 'Only translate the following sentence and give me the result.'

    # Postprocess early (also strips <t> if any)
    t = translation_postprocess(t, target_language)

    src_len = len(text or "")
    out_len = len(t)
    # Allow a bit more expansion; looser when strict=False
    limit = max(24, int(src_len * (3.0 if strict else 3.6)))
    if src_len > 10 and out_len > limit:
        return False, 'The translation is too long. Only translate the sentence and give me the result.'
    if src_len <= 10 and out_len > (50 if not strict else 40):
        return False, 'Only translate the sentence and give me the result.'

    target_code = _norm_lang_label(target_language)
    trans_code  = _detect_lang(t)
    src_code    = _detect_lang(text)

    # Micro-utterance fast path: only enforce language
    if _is_micro_utterance(text):
        if target_code != "unknown" and trans_code != "unknown" and trans_code != target_code:
            return False, f'Output must be in {target_language}. Only output the translation (no explanations).'
        return True, t

    # Must be in target language
    if target_code != "unknown" and trans_code != "unknown" and trans_code != target_code:
        return False, f'Output must be in {target_language}. Only output the translation (no explanations).'

    # Hard reject same-language paraphrase (threshold slightly stricter)
    if trans_code != "unknown" and src_code != "unknown" and trans_code == src_code:
        if _too_similar_to_source(text, t, threshold=0.92):
            return False, f'The output is not a translation. Translate into {target_language} and output only the translated text.'

    # Script coverage guards (RELAXED)
    if target_code == "zh":
        cjk = len(_RE_CJK.findall(t))
        min_ratio = 0.30 if strict else 0.25
        if out_len > 0 and (cjk / out_len) < min_ratio:
            return False, 'Output must be in Chinese. Only output the translation.'
    if target_code == "ja":
        kana = len(_RE_HIRA.findall(t)) + len(_RE_KATA.findall(t))
        min_ratio = 0.12 if strict else 0.10
        if out_len > 0 and (kana / out_len) < min_ratio and len(_RE_CJK.findall(t)) < 2:
            return False, 'Output must be in Japanese. Only output the translation.'
    if target_code == "ko":
        hang = len(_RE_HANG.findall(t))
        min_ratio = 0.25 if strict else 0.20
        if out_len > 0 and (hang / out_len) < min_ratio:
            return False, 'Output must be in Korean. Only output the translation.'

    # Some visible text required
    if not re.search(r'\w', t, flags=re.UNICODE) and not _RE_CJK.search(t):
        return False, 'Only output the translation text.'

    return True, t

# ============================================================
# Sentence splitting & timing
# ============================================================
def split_text_into_sentences(para: str, target_language: str = "简体中文") -> List[str]:
    para = (para or "").strip()
    if not para:
        return []
    if _is_chinese_target(target_language):
        para = _RE_CN_SPLIT_1.sub(r"\1\n\2", para)
        para = _RE_CN_SPLIT_2.sub(r"\1\n\2", para)
        para = _RE_CN_SPLIT_3.sub(r"\1\n\2", para)
        para = _RE_CN_SPLIT_4.sub(r'\1\n\2', para)
        return [s.strip() for s in para.rstrip().split("\n") if s.strip()]
    return [p.strip() for p in _RE_LAT_SPLIT.split(para) if p.strip()]

def split_sentences(translation_items: List[Dict], target_language: str = "简体中文", use_char_based_end: bool = True) -> List[Dict]:
    output = []
    for item in translation_items:
        start = float(item['start'])
        end = float(item['end'])
        text = item['text']
        speaker = item['speaker']
        translation_text = (item.get('translation') or "").strip()

        if not translation_text:
            output.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "speaker": speaker,
                "translation": translation_text
            })
            continue

        sentences = split_text_into_sentences(translation_text, target_language) or [translation_text]

        if use_char_based_end:
            total_chars = max(1, sum(len(s) for s in sentences))
            duration = end - start
            acc = start
            for i, s in enumerate(sentences):
                if i < len(sentences) - 1:
                    seg = duration * (len(s) / total_chars)
                    seg_end = acc + seg
                else:
                    seg_end = end
                output.append({
                    "start": round(acc, 3),
                    "end": round(seg_end, 3),
                    "text": text,
                    "speaker": speaker,
                    "translation": s
                })
                acc = seg_end
        else:
            for s in sentences:
                output.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "speaker": speaker,
                    "translation": s
                })
    return output

# ============================================================
# Summarization + summary translate (kept; fast limit)
# ============================================================
def summarize(info: dict, transcript: List[dict], target_language: str = '简体中文', method: str = 'LLM') -> dict:
    transcript_text = ' '.join(line.get('text', '') for line in transcript)
    transcript_text = ensure_transcript_length(transcript_text, max_length=SUMMARY_TEXT_LIMIT)
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '

    if method in ['Google Translate', 'Bing Translate']:
        full_description = f'{info_message}\n{transcript_text}\n{info_message}\n'
        translation = translator_response(full_description, target_language)
        return {
            'title': translator_response(info['title'], target_language),
            'author': info['uploader'],
            'summary': translation,
            'language': target_language,
            'tags': info.get('tags', [])
        }

    schema_hint = (
        'Return ONLY JSON with the keys "title" and "summary". '
        'Example: {"title": "t", "summary": "s"}'
    )
    messages = [
        {'role': 'system',
         'content': f'You are an expert in the field of this video. {schema_hint}'},
        {'role': 'user',
         'content': f'The following is the full content of the video:\n'
                    f'{info_message}\n{transcript_text}\n{info_message}\n'
                    f'Please summarize the video in JSON only.'},
    ]

    summary_obj = None
    for attempt in range(6):
        try:
            response = llm_response(messages) if method == 'LLM' else None
            logger.debug(f"[summarize] raw response (attempt {attempt+1}): {str(response)[:300]}...")
            summary_obj = _extract_first_json_object(response)
            t = (summary_obj.get('title') or '').strip()
            s = (summary_obj.get('summary') or '').strip()
            if not t or not s or 'title' in t.lower():
                raise ValueError("Invalid summary fields")
            break
        except Exception as e:
            logger.debug(f"[summarize] parse error: {e}")
            time.sleep(RETRY_SLEEP_S)
    if summary_obj is None:
        # graceful fallback: a minimal summary using info
        summary_obj = {"title": info.get("title", "Untitled"), "summary": info.get("description", "")}

    safe_title = summary_obj["title"].replace('"', '\\"')
    safe_summary = summary_obj["summary"].replace('"', '\\"')
    safe_tags = json.dumps(info.get("tags", []), ensure_ascii=False)

    trans_messages = [
        {'role': 'system',
         'content': (
             f'You are a native speaker of {target_language}. '
             f'Return ONLY JSON: {{"title": "...", "summary": "...", "tags": ["..."]}}'
         )},
        {'role': 'user',
         'content': (
             f'Please translate the following into {target_language} and return JSON only:\n'
             f'{{"title": "{safe_title}", "summary": "{safe_summary}", "tags": {safe_tags} }}'
         )}
    ]

    trans = None
    for attempt in range(5):
        try:
            resp = llm_response(trans_messages)
            resp = resp.strip()
            logger.debug(f"[summarize-translate] raw response (attempt {attempt+1}): {resp[:300]}...")
            trans = _extract_first_json_object(resp)
            if not trans.get('title') or not trans.get('summary'):
                raise ValueError("Missing fields")
            break
        except Exception as e:
            logger.debug(f"[summarize-translate] parse error: {e}")
            time.sleep(RETRY_SLEEP_S)

    if trans is None:
        trans = {
            'title': summary_obj['title'],
            'summary': summary_obj['summary'],
            'tags': info.get('tags', [])
        }

    title = (trans.get('title', '')).strip().strip('“”"‘’\'《》')
    return {
        'title': title,
        'author': info.get('uploader', ''),
        'summary': (trans.get('summary', '')).strip(),
        'tags': trans.get('tags', info.get('tags', [])),
        'language': target_language
    }

# ============================================================
# Line-by-line translation (LLM path kept; MT path fast/parallel)
# ============================================================

@lru_cache(maxsize=4096)
def _mt_cached(text: str, target_language: str, server: str) -> str:
    return translator_response(text, to_language=target_language, translator_server=server)

def _norm_key(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip().lower())

def _translate_llm_path(summary: dict, transcript: List[dict], target_language: str) -> List[str]:
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    full_translation: List[str] = []

    fixed_message = [
        {
            'role': 'system',
            'content': (
                f'You are a professional translator.\n'
                f'Context (terminology only): {info}\n'
                f'RULES (must obey exactly):\n'
                f'1) Translate the quoted sentence into {target_language}.\n'
                f'2) Output ONLY inside tags: <t>...translation...</t>\n'
                f'3) No other text, no quotes, no markdown, no explanations.\n'
                f'4) Do NOT paraphrase in the original language; output MUST be in {target_language}.\n'
                f'5) Preserve numbers and technical terms faithfully.\n'
            )
        },
        {'role': 'user', 'content': 'Translate: "Original Text"'},
        {'role': 'assistant', 'content': '<t>Example translation</t>'}
    ]

    history: List[Dict[str, Any]] = []
    dedup_cache: Dict[str, str] = {}

    for line_idx, line in enumerate(transcript):
        text = line.get('text', '')
        if not text or _NON_SPEECH.match(text):
            full_translation.append('')
            continue

        key = _norm_key(text)
        if ENABLE_DEDUP_SAME_LINES and key in dedup_cache:
            full_translation.append(dedup_cache[key])
            history = history[-LLM_HISTORY_WINDOW:]
            history += [
                {'role': 'user', 'content': f'Translate: "{text}"'},
                {'role': 'assistant', 'content': dedup_cache[key]},
            ]
            time.sleep(SMALL_SLEEP_S)
            continue

        retry_hint = ''
        success = False
        last_err = None

        for attempt in range(LLM_MAX_RETRIES):
            strict = (attempt == 0)  # first attempt strict, later attempts relaxed
            messages = fixed_message + history[-LLM_HISTORY_WINDOW:] + [
                {'role': 'user',
                 'content': f'{retry_hint}Translate the following and output ONLY <t>...</t>:\n"{text}"'}
            ]
            try:
                resp = llm_response(messages)
                ok, t_clean = valid_translation(text, resp, target_language, strict=strict)
                do_bt = ENABLE_BACKTRANSLATE_VERIFY and not (_is_micro_utterance(text) or _is_numericish(text))
                if ok and do_bt:
                    if not _verify_by_backtranslation(text, t_clean, target_language):
                        ok = False
                        retry_hint = "Ensure the output is a faithful translation into the target language. "
                        raise ValueError("Back-translation verification failed")
                if not ok:
                    retry_hint = "Only output the translation. No quotes. No markdown. "
                    raise ValueError("Invalid translation output")

                full_translation.append(t_clean)
                if ENABLE_DEDUP_SAME_LINES:
                    dedup_cache[key] = t_clean
                success = True
                break
            except Exception as e:
                last_err = e
                logger.debug(f"[translate-LLM] retryable issue at idx={line_idx}: {e}")
                time.sleep(RETRY_SLEEP_S)

        if not success:
            try:
                mt_fallback = _mt_cached(text, target_language, 'google')
                ok, t_clean = valid_translation(text, mt_fallback, target_language, strict=False)
                if ok and ENABLE_BACKTRANSLATE_VERIFY and not (_is_micro_utterance(text) or _is_numericish(text)):
                    if not _verify_by_backtranslation(text, t_clean, target_language):
                        ok = False
                full_translation.append(t_clean if ok else text)
                if ok and ENABLE_DEDUP_SAME_LINES:
                    dedup_cache[key] = t_clean
                logger.warning(f"[translate-line] fell back to MT for a line due to: {last_err}")
            except Exception as ee:
                logger.warning(f"[translate-line] MT fallback failed: {ee}")
                full_translation.append(text)

        history = history[-LLM_HISTORY_WINDOW:]
        history += [
            {'role': 'user', 'content': f'Translate: "{text}"'},
            {'role': 'assistant', 'content': full_translation[-1]},
        ]
        time.sleep(SMALL_SLEEP_S)

    return full_translation

def _translate_mt_path(transcript: List[dict], target_language: str, server: str) -> List[str]:
    texts = [(i, line.get('text', '')) for i, line in enumerate(transcript)]
    results = [''] * len(texts)

    if MT_MAX_WORKERS <= 1:
        for i, t in texts:
            if not t or _NON_SPEECH.match(t):
                results[i] = ''
                continue
            mt = _mt_cached(t, target_language, server)
            ok, t_clean = valid_translation(t, mt, target_language)  # strict default
            if ok and ENABLE_BACKTRANSLATE_VERIFY and not _is_micro_utterance(t) and not _is_numericish(t):
                if not _verify_by_backtranslation(t, t_clean, target_language):
                    ok = False
            results[i] = t_clean if ok else t
            time.sleep(SMALL_SLEEP_S)
        return results

    with ThreadPoolExecutor(max_workers=MT_MAX_WORKERS) as ex:
        futs = {}
        for i, t in texts:
            if not t or _NON_SPEECH.match(t):
                results[i] = ''
                continue
            futs[ex.submit(_mt_cached, t, target_language, server)] = (i, t)

        for fut in as_completed(futs):
            i, src = futs[fut]
            try:
                mt = fut.result()
                ok, t_clean = valid_translation(src, mt, target_language)  # strict default
                if ok and ENABLE_BACKTRANSLATE_VERIFY and not _is_micro_utterance(src) and not _is_numericish(src):
                    if not _verify_by_backtranslation(src, t_clean, target_language):
                        ok = False
                results[i] = t_clean if ok else src
            except Exception as e:
                logger.debug(f"[translate-mt] worker error: {e}")
                results[i] = src
    return results

def _translate(summary: dict, transcript: List[dict], target_language: str = '简体中文', method: str = 'LLM') -> List[str]:
    # FAST mode: prefer MT path unless explicitly forced to LLM
    if FAST_TRANSLATION_MODE and method not in ['Google Translate', 'Bing Translate', 'LLM']:
        method = 'Google Translate'
    if method in ['Google Translate', 'Bing Translate']:
        server = 'google' if method == 'Google Translate' else 'bing'
        return _translate_mt_path(transcript, target_language, server)
    return _translate_llm_path(summary, transcript, target_language)

# ============================================================
# Public entry points
# ============================================================
def _atomic_write_json(path: str, obj: Any):
    tmp = f"{path}.tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def translate(method: str, folder: str, target_language: str = '简体中文'):
    """
    Translate a single video folder w/ transcript.json.
    Writes/updates summary.json and translation.json (time-aligned).
    """
    translation_path = os.path.join(folder, 'translation.json')
    if os.path.exists(translation_path):
        logger.info(f'Translation already exists in {folder}')
        return True

    info_path = os.path.join(folder, 'download.info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info_raw = json.load(f)
        info = get_necessary_info(info_raw)
    else:
        info = {
            'title': os.path.basename(folder),
            'uploader': 'Unknown',
            'description': 'Unknown',
            'upload_date': 'Unknown',
            'tags': []
        }

    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = summarize(info, transcript, target_language, method)
        _atomic_write_json(summary_path, summary)

    translations = _translate(summary, transcript, target_language, method)

    # Attach and split
    for i, line in enumerate(transcript):
        line['translation'] = translations[i]
    transcript_split = split_sentences(transcript, target_language=target_language, use_char_based_end=True)

    _atomic_write_json(translation_path, transcript_split)
    return summary, transcript_split

def translate_all_transcript_under_folder(folder: str, method: str, target_language: str):
    """
    Walk directory; translate each subfolder that has transcript.json but not translation.json.
    Returns (message, last_summary_json, last_translation_json)
    """
    summary_json, translate_json = None, None
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            summary_json, translate_json = translate(method, root, target_language)
        elif 'translation.json' in files:
            sum_p = os.path.join(root, 'summary.json')
            trn_p = os.path.join(root, 'translation.json')
            if os.path.exists(sum_p):
                with open(sum_p, 'r', encoding='utf-8') as f:
                    summary_json = json.load(f)
            if os.path.exists(trn_p):
                with open(trn_p, 'r', encoding='utf-8') as f:
                    translate_json = json.load(f)
    print(summary_json, translate_json)
    return f'Translated all videos under {folder}', summary_json, translate_json
