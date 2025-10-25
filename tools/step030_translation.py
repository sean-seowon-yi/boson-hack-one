# -*- coding: utf-8 -*-
"""
Step030 — Translation pipeline (robust + language-aware + enforcement)

Highlights
- Summarizes video (title/summary) in source, then translates summary to target_language (LLM path).
- Translates transcript line-by-line with strict validation:
    * target language enforcement (script/CLD3)
    * same-language paraphrase rejection (overlap)
    * optional back-translation gate
    * tolerant payload plucking from messy LLM outputs
    * retries with format nudges, then MT fallback if needed
- Splits translated text into timed sentences using language-aware rules.
- Writes summary.json and translation.json.

Dependencies expected in your project:
  - tools.step032_translation_llm.llm_response
  - tools.step033_translation_translator.translator_response
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
# Tunables (do not alter prompts/logic; only perf knobs)
# ============================================================
ENABLE_BACKTRANSLATE_VERIFY = False  # per-line MT verification (slow)
ENABLE_DEDUP_SAME_LINES = True       # reuse translation if identical line text repeats
MT_MAX_WORKERS = max(1, int(os.getenv("TRANSLATION_MT_MAX_WORKERS", "4")))  # only used on pure MT path
RETRY_SLEEP_S = float(os.getenv("TRANSLATION_RETRY_SLEEP", "0.2"))
SMALL_SLEEP_S = float(os.getenv("TRANSLATION_SMALL_SLEEP", "0.05"))

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

# For CN sentence splitting
_RE_CN_SPLIT_1 = re.compile(r'([。！？\?])([^，。！？\?”’》])')
_RE_CN_SPLIT_2 = re.compile(r'(\.{6})([^，。！？\?”’》])')  # ......
_RE_CN_SPLIT_3 = re.compile(r'(\…{2})([^，。！？\?”’》])')   # ……
_RE_CN_SPLIT_4 = re.compile(r'([。！？\?][”’])([^，。！？\?”’》])')
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
                return val.strip()
    except Exception:
        pass
    for rex in _RE_PREFIXES:
        m = rex.match(t)
        if m:
            t = m.group(1).strip()
            break
    m = re.search(r'“([^”]+)”', t) or re.search(r'"([^"]+)"', t) or re.search(r'‘([^’]+)’', t) or re.search(r"'([^']+)'", t)
    if m and len(m.group(1).strip()) >= 1:
        return m.group(1).strip()
    wrappers = ['“', '”', '"', '‘', '’', "'", '《', '》', '「', '」', '『', '』']
    while len(t) >= 2 and t[0] in wrappers and t[-1] in wrappers:
        t = t[1:-1].strip()
    return t.strip()

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
        "korean": "ko", "ko": "ko", "한국어": "ko",
        "spanish": "es", "es": "es", "español": "es",
        "french": "fr", "fr": "fr", "français": "fr",
        "polish": "pl", "pl": "pl", "polski": "pl",
    }
    return mapping.get(s, "unknown")

def _heuristic_lang(text: str) -> str:
    t = text or ""
    cjk = len(_RE_CJK.findall(t))
    hira = len(_RE_HIRA.findall(t))
    kata = len(_RE_KATA.findall(t))
    hang = len(_RE_HANG.findall(t))
    latin = len(_RE_LATN.findall(t))
    if (hira + kata) > 0 and cjk >= 0:
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
        return _heuristic_lang(text)
except Exception:
    def _detect_lang(text: str) -> str:
        return _heuristic_lang(text)

# ============================================================
# Similarity / overlap guards
# ============================================================
def _token_set(s: str) -> set:
    s = (s or "").lower().translate(_PUNC_TABLE)
    return set(s.split())

def _too_similar_to_source(src: str, tgt: str) -> bool:
    ts, tt = _token_set(src), _token_set(tgt)
    if not ts or not tt:
        return False
    overlap = len(ts & tt) / max(1, len(ts | tt))
    return overlap >= 0.85

# ============================================================
# Back-translation verification (optional)
# ============================================================
def _verify_by_backtranslation(src_text: str, tgt_text: str, target_language: str) -> bool:
    try:
        src_code = _detect_lang(src_text)
        src_label = {
            "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
            "es": "Spanish", "fr": "French", "pl": "Polish"
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
# Validation
# ============================================================
def valid_translation(text: str, translation: str, target_language: str = "简体中文") -> Tuple[bool, str]:
    t = _pluck_translation_payload(translation)
    if not t:
        return False, 'Only translate the following sentence and give me the result.'

    src_len = len(text)
    out_len = len(t)
    limit = max(24, int(src_len * 2.5))
    if src_len > 10 and out_len > limit:
        return False, 'The translation is too long. Only translate the sentence and give me the result.'
    if src_len <= 10 and out_len > 40:
        return False, 'Only translate the sentence and give me the result.'

    t = translation_postprocess(t, target_language)

    target_code = _norm_lang_label(target_language)
    trans_code = _detect_lang(t)
    src_code = _detect_lang(text)

    if target_code != "unknown" and trans_code != "unknown" and trans_code != target_code:
        return False, f'Output must be in {target_language}. Only output the translation (no explanations).'

    if trans_code != "unknown" and src_code != "unknown" and trans_code == src_code:
        if _too_similar_to_source(text, t):
            return False, f'The output is not a translation. Translate into {target_language} and output only the translated text.'

    if target_code == "zh":
        cjk = len(_RE_CJK.findall(t))
        if out_len > 0 and (cjk / out_len) < 0.4:
            return False, 'Output must be in Chinese. Only output the translation.'
    if target_code == "ja":
        kana = len(_RE_HIRA.findall(t)) + len(_RE_KATA.findall(t))
        if out_len > 0 and (kana / out_len) < 0.2 and len(_RE_CJK.findall(t)) < 2:
            return False, 'Output must be in Japanese. Only output the translation.'
    if target_code == "ko":
        hang = len(_RE_HANG.findall(t))
        if out_len > 0 and (hang / out_len) < 0.3:
            return False, 'Output must be in Korean. Only output the translation.'

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
# Summarization + summary translate
# ============================================================
def summarize(info: dict, transcript: List[dict], target_language: str = '简体中文', method: str = 'LLM') -> dict:
    transcript_text = ' '.join(line.get('text', '') for line in transcript)
    transcript_text = ensure_transcript_length(transcript_text, max_length=2000)
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
    for attempt in range(9):
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
        raise Exception('Failed to summarize')

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
    for attempt in range(6):
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
# Line-by-line translation
# ============================================================

@lru_cache(maxsize=4096)
def _mt_cached(text: str, target_language: str, server: str) -> str:
    """Cache MT results only (LLM intentionally not cached to preserve context)."""
    return translator_response(text, to_language=target_language, translator_server=server)

def _translate_llm_path(summary: dict, transcript: List[dict], target_language: str) -> List[str]:
    """Sequential LLM path (preserves behavior, history, and prompts)."""
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    full_translation: List[str] = []

    fixed_message = [
        {'role': 'system',
         'content': (
             f'You are a professional translator.\n'
             f'Context (for terminology only): {info}\n'
             f'TRANSLATION RULES (must obey):\n'
             f'1) Translate the quoted sentence into {target_language}.\n'
             f'2) Output ONLY the translation text — no quotes, no markdown, no prefixes.\n'
             f'3) Do NOT paraphrase in the original language; the output MUST be in {target_language}.\n'
             f'4) Preserve technical terms and numbers faithfully.\n'
         )},
        {'role': 'user', 'content': 'Translate: "Original Text"'},
        {'role': 'assistant',
         'content': '示例译文（仅文本，无引号）' if _is_chinese_target(target_language) else 'Example translation (text only)'},
    ]

    history: List[Dict[str, Any]] = []
    dedup_cache: Dict[str, str] = {}

    for line in transcript:
        text = line.get('text', '')
        if not text:
            full_translation.append('')
            continue

        # Optional dedup for repeated lines
        if ENABLE_DEDUP_SAME_LINES and text in dedup_cache:
            full_translation.append(dedup_cache[text])
            history = history[-30:]
            history += [
                {'role': 'user', 'content': f'Translate: "{text}"'},
                {'role': 'assistant', 'content': dedup_cache[text]},
            ]
            time.sleep(SMALL_SLEEP_S)
            continue

        retry_hint = ''
        success = False
        last_err = None

        for _ in range(10):
            messages = fixed_message + history[-30:] + [
                {'role': 'user', 'content': f'{retry_hint} Return only the translation. Translate: "{text}"'}
            ]
            try:
                resp = llm_response(messages)
                ok, t_clean = valid_translation(text, resp, target_language)
                if ok and ENABLE_BACKTRANSLATE_VERIFY:
                    if not _verify_by_backtranslation(text, t_clean, target_language):
                        ok = False
                        retry_hint = "Ensure the output is a faithful translation into the target language. "
                        raise ValueError("Back-translation verification failed")
                if not ok:
                    retry_hint = "Only output the translation. No quotes. No markdown. "
                    raise ValueError("Invalid translation output")

                full_translation.append(t_clean)
                if ENABLE_DEDUP_SAME_LINES:
                    dedup_cache[text] = t_clean
                success = True
                break
            except Exception as e:
                last_err = e
                logger.debug(f"[translate-line] retryable issue: {e}")
                time.sleep(RETRY_SLEEP_S)

        if not success:
            try:
                mt_fallback = _mt_cached(text, target_language, 'google')
                ok, t_clean = valid_translation(text, mt_fallback, target_language)
                if ok and ENABLE_BACKTRANSLATE_VERIFY and not _verify_by_backtranslation(text, t_clean, target_language):
                    ok = False
                full_translation.append(t_clean if ok else text)
                if ok and ENABLE_DEDUP_SAME_LINES:
                    dedup_cache[text] = t_clean
                logger.warning(f"[translate-line] fell back to MT for a line due to: {last_err}")
            except Exception as ee:
                logger.warning(f"[translate-line] MT fallback failed: {ee}")
                full_translation.append(text)

        history = history[-30:]
        history += [
            {'role': 'user', 'content': f'Translate: "{text}"'},
            {'role': 'assistant', 'content': full_translation[-1]},
        ]
        time.sleep(SMALL_SLEEP_S)

    return full_translation

def _translate_mt_path(transcript: List[dict], target_language: str, server: str) -> List[str]:
    """
    MT path with optional parallelism. Order is preserved when assembling results.
    """
    texts = [(i, line.get('text', '')) for i, line in enumerate(transcript)]
    results = [''] * len(texts)

    # Short-circuit if single-thread
    if MT_MAX_WORKERS <= 1:
        for i, t in texts:
            if not t:
                results[i] = ''
                continue
            mt = _mt_cached(t, target_language, server)
            ok, t_clean = valid_translation(t, mt, target_language)
            if ok and ENABLE_BACKTRANSLATE_VERIFY and not _verify_by_backtranslation(t, t_clean, target_language):
                ok = False
            results[i] = t_clean if ok else t
            time.sleep(SMALL_SLEEP_S)
        return results

    # Parallel MT
    with ThreadPoolExecutor(max_workers=MT_MAX_WORKERS) as ex:
        futs = {}
        for i, t in texts:
            if not t:
                results[i] = ''
                continue
            futs[ex.submit(_mt_cached, t, target_language, server)] = (i, t)

        for fut in as_completed(futs):
            i, src = futs[fut]
            try:
                mt = fut.result()
                ok, t_clean = valid_translation(src, mt, target_language)
                if ok and ENABLE_BACKTRANSLATE_VERIFY and not _verify_by_backtranslation(src, t_clean, target_language):
                    ok = False
                results[i] = t_clean if ok else src
            except Exception as e:
                logger.debug(f"[translate-mt] worker error: {e}")
                results[i] = src
    return results

def _translate(summary: dict, transcript: List[dict], target_language: str = '简体中文', method: str = 'LLM') -> List[str]:
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    # Keep behavior: If explicitly MT, use MT (with parallel). Else LLM path.
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
    Translates one video folder that contains transcript.json (and optionally download.info.json).
    Produces/updates summary.json and translation.json (sentence-split + time-aligned).
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
        if summary is None:
            logger.error(f'Failed to summarize {folder}')
            return False
        _atomic_write_json(summary_path, summary)

    translations = _translate(summary, transcript, target_language, method)

    # Attach translations back to original line structure
    for i, line in enumerate(transcript):
        line['translation'] = translations[i]

    # Language-aware sentence splitting for timing
    transcript_split = split_sentences(transcript, target_language=target_language, use_char_based_end=True)

    _atomic_write_json(translation_path, transcript_split)

    return summary, transcript_split

def translate_all_transcript_under_folder(folder: str, method: str, target_language: str):
    """
    Walk the directory; translate each subfolder that has transcript.json and lacks translation.json.
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
