# -*- coding: utf-8 -*-
"""
Boson Hackathon TTS (multilingual, native-like, consistent) — enhanced + audio hotfix
- OpenAI-compatible Higgs Audio Generation
- Multilingual verbatim: EN / KO / ZH / ES / TA (extensible)
- Language lock + anti code-switch
- Clean timbre extraction from speaker_wav (2s best-voice segment)
- Anchor-conditioning for consistent voice/tone/pitch across chunks
- Previous-chunk prosody context (not read aloud)
- Parallel chunks (after anchor), crossfaded joins, loudness normalize, soft limiter
- Low-variance decoding (deterministic by default)
- HTTP/2 pooling if httpx is available
- Optional minimal SSML pauses via TTS_USE_SSML=1 (OFF by default)
- Optional few-shot anti-translation via TTS_USE_FEWSHOT=1 (OFF by default)
- Forces WAV output from server; validates WAV header; emergency gain for near-silence
- .env-configurable; drop-in signature: tts(text, output_path, speaker_wav, voice_type=None)
"""

import os
import io
import re
import time
import math
import base64
import unicodedata
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

import numpy as np
import soundfile as sf

# Optional httpx for HTTP/2 connection pooling
try:
    import httpx
    _HAS_HTTPX = True
except Exception:
    _HAS_HTTPX = False

# Optional g2p (silent fallback if missing)
try:
    from pypinyin import pinyin, Style  # Mandarin tone hints (optional)
    _HAS_PYPINYIN = True
except Exception:
    _HAS_PYPINYIN = False

# -----------------------------
# Environment & client helpers
# -----------------------------
load_dotenv()

def _load_boson_cfg():
    """
    Load required + optional settings from .env.
    Required: BOSON_API_KEY, BOSON_BASE_URL, HIGGS_TTS_MODEL
    Optional: TTS_WORKERS, TTS_TARGET_SR, TTS_CHUNK_CHARS, TTS_MAX_RETRIES,
              TTS_TIMEOUT_S, TTS_SPEED, TTS_LANG_OVERRIDE, TTS_REF_*, TTS_STYLE_*,
              TTS_USE_SSML, TTS_USE_FEWSHOT, TTS_DEBUG_SAVE_FIRST
    """
    cfg = {
        # required
        "api_key":  os.getenv("BOSON_API_KEY"),
        "base_url": os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1"),
        "model":    os.getenv("HIGGS_TTS_MODEL"),

        # optional (provide safe defaults)
        "workers":      int(os.getenv("TTS_WORKERS", "6")),
        "target_sr":    int(os.getenv("TTS_TARGET_SR", "24000")),
        "chunk_chars":  int(os.getenv("TTS_CHUNK_CHARS", "200")),
        "max_retries":  int(os.getenv("TTS_MAX_RETRIES", "2")),
        "timeout_s":    float(os.getenv("TTS_TIMEOUT_S", "45")),
        "speed_ratio":  float(os.getenv("TTS_SPEED", "1.00")),
        "use_ssml":     (os.getenv("TTS_USE_SSML", "0") == "1"),         # OFF by default
        "use_fewshot":  (os.getenv("TTS_USE_FEWSHOT", "0") == "1"),      # OFF by default
        "debug_save":   (os.getenv("TTS_DEBUG_SAVE_FIRST", "0") == "1"), # OFF by default

        # OPTIONAL language override; empty string means “no override”
        "lang_override": (os.getenv("TTS_LANG_OVERRIDE") or "").strip().lower(),
    }

    missing = [k for k in ("api_key", "base_url", "model") if not cfg[k]]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

    return cfg

_client = None
_model = None
_cfg = None

def _get_client_and_model():
    global _client, _model, _cfg
    if _client is None or _model is None or _cfg is None:
        _cfg = _load_boson_cfg()

        http_client = None
        if _HAS_HTTPX:
            try:
                limits = httpx.Limits(max_keepalive_connections=24, max_connections=48)
                http_client = httpx.Client(http2=True, timeout=_cfg["timeout_s"], limits=limits)
            except Exception as e:
                logger.warning(f"[Higgs TTS] httpx client init failed; falling back: {e}")
                http_client = None

        try:
            _client = OpenAI(
                api_key=_cfg["api_key"],
                base_url=_cfg["base_url"],
                http_client=http_client  # newer OpenAI SDKs accept this; ignored otherwise
            )
        except TypeError:
            _client = OpenAI(api_key=_cfg["api_key"], base_url=_cfg["base_url"])

        _model = _cfg["model"]
        logger.info(f"[Higgs TTS] Client initialized @ {_cfg['base_url']} | model={_model}")
    return _client, _model, _cfg

def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -----------------------------
# Language detection & splitting
# -----------------------------
_SCRIPT_PAT = {
    "zh": re.compile(r"[\u4e00-\u9fff]"),
    "ko": re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"),
    "ta": re.compile(r"[\u0B80-\u0BFF]"),
    "es": re.compile(r"[¿¡ñáéíóúÁÉÍÓÚ]"),
}

_LANG_NAME_NATIVE = {
    "en": "English",
    "zh": "中文（普通话）",
    "ko": "한국어",
    "es": "Español",
    "ta": "தமிழ்",
}

def _detect_lang(txt: str, override: str | None = None) -> str:
    if override in {"en", "zh", "ko", "es", "ta"}:
        return override
    for code, pat in _SCRIPT_PAT.items():
        if pat.search(txt):
            return code
    return "en"

_SENT_SEP = {
    "zh": r"(?<=[。！？…])\s+",
    "ko": r"(?<=[\.!?…])\s+",
    "es": r"(?<=[\.\!\?…])\s+",
    "ta": r"(?<=[\.!?…])\s+",
    "en": r"(?<=[\.\!\?…])\s+",
}

def _normalize_text(txt: str) -> str:
    txt = unicodedata.normalize("NFC", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if txt and txt[-1] not in ".!?。！？…":
        txt += "."
    return txt

def _inject_soft_breaks(text: str, lang: str) -> str:
    """
    Insert subtle, language-appropriate soft breaks to improve rhythm,
    without altering words or translating. Keeps characters intact.
    """
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("...", "…")

    if lang in {"zh"}:
        # Add a slight list-break '、' in very long runs without punctuation
        out, run = [], 0
        for ch in t:
            out.append(ch)
            if ch in "，。！？；：、…":
                run = 0
            else:
                run += 1
                if run >= 18:
                    out.append("、")
                    run = 0
        t = "".join(out)
    else:
        # Ensure a single space after commas for tiny pause
        t = re.sub(r",\s*", ", ", t)
        # Assist very long sentences with mild spacing
        if len(t) > 100:
            t = re.sub(r"(\w{16,})(\s+)", r"\1 \2", t)
    return t

def _split_sentences(txt: str, lang: str, max_chars: int):
    sep = _SENT_SEP.get(lang, _SENT_SEP["en"])
    raw = [s.strip() for s in re.split(sep, txt) if s.strip()]
    if not raw:
        return []

    # Language-aware budget: characters are denser in CJK; permit larger chunks safely.
    budget = max_chars
    if lang in {"zh", "ko", "ta"}:
        budget = int(max_chars * 1.8)
    elif lang in {"es"}:
        budget = int(max_chars * 1.3)

    chunks, cur = [], ""
    for s in raw:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= budget:
            cur = f"{cur} {s}"
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    # Merge stragglers: avoid final tiny pieces
    if len(chunks) >= 2 and len(chunks[-1]) < budget // 3:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks

# -----------------------------
# Optional native refs & style
# -----------------------------
def _load_lang_refs_from_env():
    env_map = {
        "en": os.getenv("TTS_REF_EN"),
        "zh": os.getenv("TTS_REF_ZH"),
        "ko": os.getenv("TTS_REF_KO"),
        "es": os.getenv("TTS_REF_ES"),
        "ta": os.getenv("TTS_REF_TA"),
    }
    ref_audio = {}
    for lang, path in env_map.items():
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    ref_audio[lang] = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                logger.warning(f"[Higgs TTS] Failed loading native ref {lang}: {e}")
    return ref_audio

def _style_note(lang: str) -> str:
    fallback = {
        "en": "Neutral North American narration, clear articulation.",
        "zh": "普通话（中国大陆），自然停顿，保持地道声调与轻重缓急。",
        "ko": "표준 발음, 자연스러운 억양, 문장 끝을 부드럽게 처리.",
        "es": "Español latino neutro, ritmo natural, entonación clara y fluida.",
        "ta": "நெறிப்படுத்தப்பட்ட தமிழ் உச்சரிப்பு, இயல்பான தாளம் மற்றும் இடைவெளிகள்.",
    }
    env_key = {
        "en": "TTS_STYLE_EN", "zh": "TTS_STYLE_ZH", "ko": "TTS_STYLE_KO",
        "es": "TTS_STYLE_ES", "ta": "TTS_STYLE_TA"
    }.get(lang)
    return os.getenv(env_key) or fallback.get(lang, "")

def _make_zh_tone_hint(text: str) -> str:
    if not _HAS_PYPINYIN:
        return ""
    snippet = re.sub(r"\s+", "", text)[:30]
    if not snippet:
        return ""
    py = pinyin(snippet, style=Style.TONE3, strict=False)
    py_str = " ".join(s[0] for s in py if s and s[0])
    return f"【内注音提示（请勿朗读）: {py_str}】"

# -----------------------------
# DSP helpers
# -----------------------------
def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    y = y.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(np.maximum(y * y, 1e-12))))

def _loudness_normalize(y: np.ndarray, target_dbfs: float = -16.0) -> np.ndarray:
    if y.size == 0:
        return y
    y = y.astype(np.float32, copy=False)
    rms = _rms(y)
    cur_dbfs = 20.0 * math.log10(max(rms, 1e-9))
    gain = 10.0 ** ((target_dbfs - cur_dbfs) / 20.0)
    y = y * gain
    thr = 10 ** (-1.0 / 20.0)  # -1 dBFS
    y = np.tanh(y / thr) * thr
    return y.astype(np.float32, copy=False)

def _concat_with_crossfade(y_list: list[np.ndarray], sr: int, xf_ms: int = 30) -> np.ndarray:
    if not y_list:
        return np.zeros(0, dtype=np.float32)
    if len(y_list) == 1:
        return y_list[0].astype(np.float32, copy=False)
    xf = max(1, int(sr * xf_ms / 1000.0))
    out = y_list[0].astype(np.float32, copy=True)
    for seg in y_list[1:]:
        if seg is None or seg.size == 0:
            continue
        seg = seg.astype(np.float32, copy=False)
        a_len = min(out.size, xf)
        b_len = min(seg.size, xf)
        n = min(a_len, b_len)
        if n > 0:
            fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
            fade_in  = 1.0 - fade_out
            mixed = out[-n:] * fade_out + seg[:n] * fade_in
            out = np.concatenate([out[:-n], mixed, seg[n:]]).astype(np.float32, copy=False)
        else:
            out = np.concatenate([out, seg]).astype(np.float32, copy=False)
    return out

def _resample_linear(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or y.size == 0:
        return y.astype(np.float32, copy=False)
    ratio = sr_to / sr_from
    x_old = np.linspace(0.0, 1.0, num=y.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=int(round(y.size * ratio)), endpoint=False, dtype=np.float64)
    y_new = np.interp(x_new, x_old, y.astype(np.float64)).astype(np.float32)
    return y_new

# --- Clean reference selection (best ~2s voiced segment) ---
def _frame_energy(x: np.ndarray, win: int = 1024, hop: int = 256) -> np.ndarray:
    if x.size < win:
        return np.array([float(np.mean(x * x))], dtype=np.float32)
    n = 1 + (x.size - win) // hop
    E = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i*hop:i*hop+win]
        E[i] = float(np.mean(seg * seg))
    return E

def _select_clean_ref_segment(path: str, target_sr: int = 24000,
                              seg_seconds: float = 2.0) -> np.ndarray | None:
    """Return the cleanest ~2s mono segment from `path` or None on failure."""
    try:
        y, sr = sf.read(path, always_2d=False, dtype="float32")
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1)
        if sr != target_sr and y.size:
            y = _resample_linear(y, sr, target_sr)
        y = y.astype(np.float32, copy=False)

        if y.size < int(0.6 * target_sr):
            return y if y.size else None

        win, hop = 1024, 256
        E = _frame_energy(y, win, hop)
        if E.size < 8:
            seg = y[: int(seg_seconds * target_sr)]
        else:
            clip_mask = (np.abs(y) > 0.98).astype(np.float32)
            if clip_mask.any():
                clip_frames = np.convolve(clip_mask, np.ones(win, dtype=np.float32), mode="valid")[::hop]
                clip_frames = clip_frames[:E.size]
                E = E / (1.0 + 5.0 * clip_frames)
            best_i = int(np.argmax(E))
            center_samp = best_i * hop + win // 2
            half = int(seg_seconds * target_sr // 2)
            start = max(0, center_samp - half)
            end = min(y.size, start + int(seg_seconds * target_sr))
            seg = y[start:end]

        nfade = min(256, seg.size // 8)
        if nfade > 0:
            seg[:nfade] *= np.linspace(0.2, 1.0, nfade, dtype=np.float32)
            seg[-nfade:] *= np.linspace(1.0, 0.2, nfade, dtype=np.float32)
        return seg
    except Exception:
        return None

# -----------------------------
# Audio extraction / decoding (WAV forced)
# -----------------------------
def _extract_audio_b64(resp) -> str:
    # Primary: OpenAI-style audio field
    try:
        return resp.choices[0].message.audio.data
    except Exception:
        pass
    # Secondary: content list with output_audio / audio dicts
    try:
        content = resp.choices[0].message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "output_audio" and "audio" in item and "data" in item["audio"]:
                        return item["audio"]["data"]
                    if "audio" in item and "data" in item["audio"]:
                        return item["audio"]["data"]
                    if item.get("type") == "audio" and "data" in item:
                        return item["data"]
    except Exception:
        pass
    raise ValueError("No audio base64 found in response")

def _read_wav_bytes(raw: bytes):
    # Guard: make sure this looks like WAV (RIFF/WAVE)
    if not (len(raw) > 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"):
        raise ValueError("Returned audio is not WAV. Ensure audio.format='wav' is requested.")
    data, sr = sf.read(io.BytesIO(raw), always_2d=False, dtype="float32")
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)
    y = data.astype(np.float32, copy=False)

    # Emergency gain if the server produced near-silence
    rms = float(np.sqrt(np.mean(np.maximum(y * y, 1e-12)))) if y.size else 0.0
    if rms < 1e-4 and y.size > 0:  # ~ -80 dBFS
        y *= 30.0  # +29.5 dB
    return y, int(sr)

def _wav_bytes_from_array(y: np.ndarray, sr: int) -> bytes:
    """Encode a float32 mono array as WAV bytes (for anchor/reference)."""
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32, copy=False), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

# -----------------------------
# Caching for synthesized chunks
# -----------------------------
class _ChunkCache:
    """Simple bounded dict (LRU-ish) for raw WAV bytes keyed by deterministic strings."""
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._store: OrderedDict[str, bytes] = OrderedDict()

    def get(self, key: str) -> bytes | None:
        val = self._store.get(key)
        if val is not None:
            self._store.move_to_end(key)
        return val

    def put(self, key: str, value: bytes):
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)

_chunk_cache = _ChunkCache(max_items=512)

def _hash_str(s: str | None) -> str:
    if not s:
        return "0"
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()

# -----------------------------
# Synthesis
# -----------------------------
def _system_prompt(lang: str, style_text: str, zh_tone_hint: str) -> str:
    lang_name = _LANG_NAME_NATIVE.get(lang, "target language")
    return (
        f"You must speak ONLY in {lang_name}. Do not speak any words in other languages. "
        "Read the user text verbatim with native pronunciation and prosody. "
        "Do NOT translate, paraphrase, or add words. Ignore any [SPEAKER*] tags, and do not read any hints aloud. "
        "Use the accent/prosody of reference audios if provided; keep the same voice across all sentences. "
        f"Target style: {style_text} {zh_tone_hint}"
    )

def _fewshot_anti_translate(lang: str) -> str:
    name = _LANG_NAME_NATIVE.get(lang, "target language")
    return (
        f"【Example — DO NOT READ IN OUTPUT】\n"
        f"Instruction: Speak only in {name} without translating the input. Read verbatim.\n"
        f"Input: Hello AI.\n"
        f"Output should be: Hello AI.\n"
    )

def _wrap_ssml_if_enabled(text: str, use_ssml: bool) -> str:
    if not use_ssml:
        return text
    # Minimal, non-intrusive SSML-like wrapper
    ssml = (
        text.replace("，", "，<break time=\"180ms\"/>")
            .replace(",", ", <break time=\"160ms\"/>")
            .replace("。", "。<break time=\"220ms\"/>")
            .replace(".", ". <break time=\"200ms\"/>")
    )
    return f"<speak><p>{ssml}</p></speak>"

def _synthesize_chunk(client, model, lang: str, text_chunk: str,
                      ref_b64_user: str | None, ref_b64_native: str | None,
                      ref_b64_anchor: str | None,
                      style_text: str,
                      zh_tone_hint: str,
                      prev_text: str | None,
                      use_ssml: bool,
                      use_fewshot: bool,
                      # Deterministic defaults to avoid silent failures
                      temperature=0.0, top_p=1.0, top_k=1, timeout_s=45) -> bytes:

    messages = [{"role": "system", "content": _system_prompt(lang, style_text, zh_tone_hint)}]

    # Few-shot nudge against translation/paraphrase (disabled by default)
    if use_fewshot:
        messages.append({"role": "user", "content": _fewshot_anti_translate(lang)})

    # Prosody continuity context (not to be read aloud)
    if prev_text:
        messages.append({
            "role": "assistant",
            "content": f"[Context for prosody only — DO NOT READ ALOUD]\n{prev_text}"
        })

    # Native accent reference (steer accent/prosody)
    if ref_b64_native:
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": ref_b64_native, "format": "wav"}
            }],
        })

    # User timbre reference (keep identity)
    if ref_b64_user:
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": ref_b64_user, "format": "wav"}
            }],
        })

    # Anchor reference from already-generated audio (stabilize voice/pitch)
    if ref_b64_anchor:
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": ref_b64_anchor, "format": "wav"}
            }],
        })

    # Final input
    text_payload = _wrap_ssml_if_enabled(text_chunk, use_ssml)
    messages.append({"role": "user", "content": text_payload})

    # >>> HOTFIX: force WAV format, include in both top-level arg and extra_body
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=2048,
        temperature=temperature,
        top_p=top_p,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        audio={"format": "wav"},
        extra_body={"top_k": top_k, "audio": {"format": "wav"}},
        timeout=timeout_s,
    )

    audio_b64 = _extract_audio_b64(resp)
    # Optional: save first raw chunk for debugging
    try:
        raw = base64.b64decode(audio_b64)
        if os.getenv("TTS_DEBUG_SAVE_FIRST", "0") == "1":
            with open("tts_first_chunk.raw.wav", "wb") as dbg:
                dbg.write(raw)
    except Exception as e:
        logger.error(f"[Higgs TTS] base64 decode failed: {e}")
        raise
    return raw

def _synthesize_chunk_with_retry(client, model, lang, chunk,
                                 ref_user_b64, ref_native_b64, ref_anchor_b64,
                                 style_text, zh_tone_hint,
                                 prev_text, use_ssml, use_fewshot,
                                 max_retries, timeout_s):
    # Deterministic cache key
    key = "|".join([
        model, lang,
        str(len(chunk)), hashlib.blake2b(chunk.encode("utf-8"), digest_size=8).hexdigest(),
        _hash_str(prev_text),
        _hash_str(ref_user_b64), _hash_str(ref_native_b64), _hash_str(ref_anchor_b64),
        _hash_str(style_text), _hash_str(zh_tone_hint),
        "ssml1" if use_ssml else "ssml0",
        "few1" if use_fewshot else "few0",
    ])

    cached = _chunk_cache.get(key)
    if cached:
        try:
            y, sr = _read_wav_bytes(cached)
            if y.size >= 1000:
                return y, sr
        except Exception:
            pass

    delay = 0.15
    for attempt in range(max_retries):
        try:
            raw = _synthesize_chunk(
                client, model, lang, chunk,
                ref_user_b64, ref_native_b64, ref_anchor_b64,
                style_text=style_text,
                zh_tone_hint=zh_tone_hint,
                prev_text=prev_text,
                use_ssml=use_ssml,
                use_fewshot=use_fewshot,
                temperature=0.0,  # deterministic
                top_p=1.0,
                top_k=1,
                timeout_s=timeout_s
            )
            y, sr = _read_wav_bytes(raw)
            logger.debug(f"[Higgs TTS] chunk len={y.size} sr={sr} peak={float(np.max(np.abs(y))) if y.size else 0.0:.3f}")
            if y.size < 1000:
                raise ValueError("too-short audio")
            _chunk_cache.put(key, raw)
            return y, sr
        except Exception as e:
            logger.warning(f"[Higgs TTS] Chunk retry {attempt+1}/{max_retries} failed: {e}")
            if attempt + 1 == max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.8)  # backoff

# -----------------------------
# Main TTS (drop-in)
# -----------------------------
def tts(text, output_path, speaker_wav, voice_type=None):
    """
    Generate speech for `text` and write to `output_path` (wav).
    - User timbre ref: `speaker_wav` (optional; we auto-extract a clean 2s reference)
    - Native accent refs: .env TTS_REF_{EN,ZH,KO,ES,TA} (optional)
    - Language lock: auto-detect or .env TTS_LANG_OVERRIDE
    """
    if os.path.exists(output_path):
        logger.info(f"[Higgs TTS] File already exists, skipping: {output_path}")
        return

    client, model, cfg = _get_client_and_model()
    # Safety: disable SSML & few-shot if you want hard guarantee of audio generation
    # (You can re-enable by setting env TTS_USE_SSML=1 or TTS_USE_FEWSHOT=1)
    use_ssml = bool(cfg["use_ssml"])
    use_fewshot = bool(cfg["use_fewshot"])

    # Text prep
    text_norm = _normalize_text(text)
    lang = _detect_lang(text_norm, cfg.get("lang_override", ""))
    text_norm = _inject_soft_breaks(text_norm, lang)

    # Load per-language native accent references (once)
    native_refs = _load_lang_refs_from_env()
    ref_native_b64 = native_refs.get(lang)

    # User timbre reference (cleaned segment preferred)
    ref_user_b64, ref_key = None, "no-ref"
    clean_ref = None
    if speaker_wav and os.path.exists(speaker_wav):
        try:
            clean_ref = _select_clean_ref_segment(speaker_wav, target_sr=cfg["target_sr"], seg_seconds=2.0)
            if clean_ref is not None and clean_ref.size > 0:
                ref_user_b64 = base64.b64encode(_wav_bytes_from_array(clean_ref, cfg["target_sr"])).decode("utf-8")
                ref_key = f"{os.path.basename(speaker_wav)}:{clean_ref.size}"
                logger.info(f"[Higgs TTS] Using CLEANED user reference from: {speaker_wav}")
            else:
                with open(speaker_wav, "rb") as f:
                    ref_user_b64 = base64.b64encode(f.read()).decode("utf-8")
                ref_key = f"{os.path.basename(speaker_wav)}:{len(ref_user_b64)}"
                logger.warning("[Higgs TTS] Clean reference selection failed; using raw file")
        except Exception as e:
            logger.warning(f"[Higgs TTS] Failed to prepare user reference, proceeding without it: {e}")
            ref_user_b64, ref_key = None, "no-ref"

    # Chunking
    chunks = _split_sentences(text_norm, lang, cfg["chunk_chars"])
    if not chunks:
        raise RuntimeError("No text to synthesize")

    style_text = _style_note(lang)
    zh_tone_anchor = _make_zh_tone_hint(chunks[0]) if lang == "zh" else ""

    # ---- 1) Generate ANCHOR chunk serially (first chunk) ----
    t0 = time.time()
    y_anchor, sr_anchor = _synthesize_chunk_with_retry(
        client, model, lang, chunks[0],
        ref_user_b64, ref_native_b64, None,      # no anchor yet
        style_text=style_text,
        zh_tone_hint=zh_tone_anchor,             # tone hint only on anchor
        prev_text=None,
        use_ssml=use_ssml,
        use_fewshot=use_fewshot,
        max_retries=cfg["max_retries"],
        timeout_s=cfg["timeout_s"]
    )

    # Build anchor reference: first ~2s to bias voice across all remaining chunks
    target_sr = cfg["target_sr"]
    if sr_anchor != target_sr:
        y_anchor_rs = _resample_linear(y_anchor, sr_anchor, target_sr)
    else:
        y_anchor_rs = y_anchor

    anchor_seconds = min(2.0, max(0.8, y_anchor_rs.size / max(target_sr, 1)))
    anchor_samps = int(anchor_seconds * target_sr)
    anchor_clip = y_anchor_rs[:anchor_samps]
    anchor_b64 = base64.b64encode(_wav_bytes_from_array(anchor_clip, target_sr)).decode("utf-8")

    y_list = [None] * len(chunks)
    sr_list = [None] * len(chunks)
    y_list[0], sr_list[0] = y_anchor, sr_anchor

    # ---- 2) Generate remaining chunks in parallel WITH ANCHOR ----
    rest = [(i, c) for i, c in enumerate(chunks) if i != 0]
    workers = min(cfg["workers"], max(1, len(rest)))

    if rest:
        def _work(pair):
            i, ck = pair
            prev_txt = chunks[i-1] if i > 0 else None
            y, sr = _synthesize_chunk_with_retry(
                client, model, lang, ck,
                ref_user_b64, ref_native_b64, anchor_b64,
                style_text=style_text,
                zh_tone_hint="",               # only anchor uses tone hint
                prev_text=prev_txt,            # continuity cue
                use_ssml=use_ssml,
                use_fewshot=use_fewshot,
                max_retries=cfg["max_retries"],
                timeout_s=cfg["timeout_s"]
            )
            return i, y, sr

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, y_sr in zip((i for i, _ in rest), ex.map(_work, rest)):
                i2, y, sr = y_sr
                y_list[i2], sr_list[i2] = y, sr

    # ---- 3) Align SR, stitch, normalize, optional speed ----
    for i in range(len(y_list)):
        if y_list[i] is None:
            raise RuntimeError(f"Missing audio for chunk {i}")
    for i in range(len(y_list)):
        if sr_list[i] != target_sr:
            y_list[i] = _resample_linear(y_list[i], sr_list[i], target_sr)
            sr_list[i] = target_sr

    # order preserved; crossfade hides joins, anchor keeps tone/pitch stable
    y_all = _concat_with_crossfade(y_list, target_sr, xf_ms=28)
    y_all = _loudness_normalize(y_all, target_dbfs=-16.0)

    if abs(cfg.get("speed_ratio", 1.0) - 1.0) > 1e-3:
        r = cfg["speed_ratio"]
        y_all = _resample_linear(y_all, target_sr, int(round(target_sr * r)))
        target_sr = int(round(target_sr * r))

    n = int(0.01 * target_sr)
    if y_all.size > 2 * n:
        y_all[:n] *= np.linspace(0.2, 1.0, n, dtype=np.float32)
        y_all[-n:] *= np.linspace(1.0, 0.2, n, dtype=np.float32)

    sf.write(output_path, y_all.astype(np.float32, copy=False), target_sr)
    dur = y_all.size / max(target_sr, 1)
    logger.info(f"[Higgs TTS] Saved: {output_path} | lang={lang} | chunks={len(chunks)} "
                f"(anchor+{len(rest)}) | workers={workers} | dur={dur:.2f}s | wall={time.time()-t0:.2f}s")
    return

if __name__ == '__main__':
    pass
