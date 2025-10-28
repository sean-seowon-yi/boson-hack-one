# -*- coding: utf-8 -*-
import os
from typing import Any, Optional, Tuple
import re
import time
import urllib.parse
import threading
import webbrowser
import socket

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# =========================
# CSS (unchanged)
# =========================
CUSTOM_CSS = """
/* ---------- Palette (additive; doesn't alter existing rules) ---------- */
:root {
  --accent: #ff7a00;
  --accent-600: #e86d00;
  --accent-700: #cc6000;
  --success: #107832;
  --success-bg: #e7f7ea;
  --danger: #b32626;
  --danger-bg: #fde9e9;
  --warn-bg: #fff4e5;
  --ink-900: #2a2a2a;
  --ink-700: #3b3b3b;
  --ink-600: #4a4a4a;
  --soft-blue-50: #eef5ff;
  --soft-blue-100: #e6f0ff;
}

/* Orange border around the ENTIRE LEFT INPUT PANEL (Interface's left column) */
.gradio-container .main > .container > .grid > :first-child {
  border: 3px solid #ff7a00 !important;
  border-radius: 10px !important;
  padding: 12px !important;
  box-shadow: 0 0 0 2px rgba(255,122,0,0.08), 0 6px 22px rgba(0,0,0,0.06);
}

/* Compact, elegant status indicator */
#status input {
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.9rem;
  padding: 4px 10px !important;
  border-radius: 10px !important;
  width: 150px !important;
  min-height: 32px !important;
}
#status input[value="Success"] {
  color: #107832 !important;
  background: #e7f7ea !important;
  border: 1px solid #a9e0b7 !important;
}
#status input[value="Fail"] {
  color: #b32626 !important;
  background: #fde9e9 !important;
  border: 1px solid #f4b2b2 !important;
}

/* ---------- Visible warning callouts ---------- */
#subtitle_lang_box::after {
  content: "Warning: Subtitle language and TTS target language should match for best alignment.";
  display: block;
  background: #fff4e5;
  border-left: 4px solid #ff7a00;
  padding: 6px 10px;
  border-radius: 6px;
  margin-top: 6px;
  color: #5b3600;
  font-weight: 600;
  font-size: 0.9rem;
}
#emotion_box::after {
  content: "Warning: Emotion is still in development — output may sound choppy.";
  display: block;
  background: #fff4e5;
  border-left: 4px solid #ff7a00;
  padding: 6px 10px;
  border-radius: 6px;
  margin-top: 6px;
  color: #5b3600;
  font-weight: 600;
  font-size: 0.9rem;
}
"""

# =========================
# Language normalization
# =========================
def _normalize_lang(label: Optional[str]) -> str:
    if not isinstance(label, str):
        return "en"
    s = label.strip().lower().replace("_", "-")
    if "chinese" in s or "中文" in s or "简体" in s or s.startswith("zh"):
        return "zh-cn"
    if "english" in s or s.startswith("en"):
        return "en"
    if "korean" in s or "한국" in s or s.startswith("ko"):
        return "ko"
    if "spanish" in s or "español" in s or s.startswith("es"):
        return "es"
    if "french" in s or "français" in s or s.startswith("fr"):
        return "fr"
    return "en"

# =========================
# YouTube helpers
# =========================
def _extract_youtube_id(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{6,})",
        r"youtu\.be/([A-Za-z0-9_-]{6,})",
        r"/embed/([A-Za-z0-9_-]{6,})",
        r"/shorts/([A-Za-z0-9_-]{6,})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def _build_youtube_embed(video_url: str) -> str:
    vid = _extract_youtube_id(video_url or "")
    if not vid:
        return "<p>Invalid or unsupported YouTube link.</p>"
    return (
        "<div style='text-align:center'>"
        f"<iframe width='640' height='360' "
        f"src='https://www.youtube.com/embed/{vid}' "
        "title='YouTube player' frameborder='0' "
        "allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' "
        "allowfullscreen></iframe></div>"
    )

# =========================
# Extract translated video URL (/media or absolute) from pipeline result
# =========================
def _pluck_translated_video(res: Any) -> Tuple[str, Optional[str]]:
    status = "Fail"
    translated = None
    try:
        if isinstance(res, tuple) and len(res) == 2:
            status_text, maybe_url = res
            if isinstance(maybe_url, str) and (maybe_url.startswith("/media/") or maybe_url.startswith("http")):
                translated = maybe_url
                status = "Success" if "Success" in str(status_text) else "Success"
        elif isinstance(res, dict):
            for k in ("output", "output_video", "translated_video", "final_video", "url"):
                v = res.get(k)
                if isinstance(v, str) and (v.startswith("/media/") or v.startswith("http")):
                    translated = v
                    status = "Success"
                    break
        elif isinstance(res, str):
            if res.startswith("/media/") or res.startswith("http"):
                translated = res
                status = "Success"
    except Exception:
        status = "Fail"
    return status, translated

# =========================
# Main callback (adds request: gr.Request)
# =========================
def run_with_emotion(
    output_folder: str,
    video_url: str,
    translation_method: str,
    subtitle_language: str,
    tts_method: str,
    tts_target_language: str,
    emotion: str,
    emotion_strength: float,
    request: gr.Request,   # <--- important
):
    subtitle_lang_code = _normalize_lang(subtitle_language)
    tts_lang_code = _normalize_lang(tts_target_language)

    if subtitle_lang_code != tts_lang_code:
        print("Note: Subtitle language and TTS language differ; alignment may suffer.")

    from tools.do_everything import do_everything

    # Keep your previous defaults
    num_videos = 1
    download_resolution = "1080p"
    demucs_model = "htdemucs_ft"
    device = "auto"
    num_shifts = 5
    asr_backend = "Higgs"
    whisperx_size = "large"
    batch_size = 32
    enable_diar = True
    min_speakers = None
    max_speakers = None
    subtitles = True
    playback_speed = 1.0
    fps = 30
    bgm_path = None
    bgm_vol = 0.5
    video_vol = 1.0
    output_resolution = "1080p"
    max_workers = 1
    max_retries = 3
    edgetts_voice = None

    try:
        res = do_everything(
            output_folder, video_url, num_videos, download_resolution, demucs_model,
            device, num_shifts, asr_backend, whisperx_size, batch_size, enable_diar,
            min_speakers, max_speakers, translation_method, subtitle_lang_code,
            tts_method, tts_lang_code, edgetts_voice, subtitles, playback_speed, fps,
            bgm_path, bgm_vol, video_vol, output_resolution, max_workers, max_retries,
            emotion=emotion, emotion_strength=float(emotion_strength),
        )
    except TypeError:
        # Backward compatibility
        os.environ["EMOTION_PRESET"] = str(emotion)
        os.environ["EMOTION_STRENGTH"] = str(emotion_strength)
        res = do_everything(
            output_folder, video_url, num_videos, download_resolution, demucs_model,
            device, num_shifts, asr_backend, whisperx_size, batch_size, enable_diar,
            min_speakers, max_speakers, translation_method, subtitle_lang_code,
            tts_method, tts_lang_code, edgetts_voice, subtitles, playback_speed, fps,
            bgm_path, bgm_vol, video_vol, output_resolution, max_workers, max_retries,
        )
    except Exception:
        return "Fail", None, _build_youtube_embed(video_url)

    status, translated = _pluck_translated_video(res)

    # If it's a relative /media/... path, convert to absolute to avoid Gradio's /file= rewrite (403)
    if isinstance(translated, str) and translated.startswith("/"):
        base = f"{request.url.scheme}://{request.headers.get('host')}"
        translated = base + translated

    return status, translated, _build_youtube_embed(video_url)

# =========================
# Interface (unchanged layout)
# =========================
demo = gr.Interface(
    title="Intelligent Multilingual Video Dubbing/Translation",
    fn=run_with_emotion,
    css=CUSTOM_CSS,
    inputs=[
        gr.Textbox(label="Output folder", value="videos"),
        gr.Textbox(
            label="Video URL",
            placeholder="Enter a YouTube/Bilibili link",
            value="https://www.youtube.com/watch?v=VowXFWlAXIU",
        ),
        gr.Dropdown(["LLM"], label="Translation method", value="LLM"),
        gr.Dropdown(
            ["Chinese (中文)", "English", "Korean", "Spanish", "French"],
            label="Subtitle language",
            value="Chinese (中文)",
            elem_id="subtitle_lang_box",
        ),
        gr.Dropdown(["Higgs", "xtts"], label="TTS method", value="xtts"),
        gr.Dropdown(
            ["Chinese (中文)", "English", "Korean", "Spanish", "French"],
            label="TTS target language",
            value="Chinese (中文)",
        ),
        gr.Dropdown(
            ["natural", "happy", "sad", "angry"],
            label="Emotion",
            value="natural",
            elem_id="emotion_box",
        ),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.6, label="Emotion strength"),
    ],
    outputs=[
        gr.Text(label="Status", elem_id="status"),
        gr.Video(label="Translated video"),
        gr.HTML(label="Original video (YouTube)"),
    ],
    allow_flagging="never",
)

# =========================
# FastAPI + Static /media + mount Gradio
# =========================
app = FastAPI()

class NoStoreMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/media/"):
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

app.add_middleware(NoStoreMiddleware)

os.makedirs("videos", exist_ok=True)
app.mount("/media", StaticFiles(directory="videos", check_dir=True), name="media")

# Mount Gradio app at root
app = gr.mount_gradio_app(app, demo, path="/")

# =========================
# Auto-open browser when running with uvicorn
# =========================
def _get_uvicorn_host_port() -> Tuple[str, int]:
    """
    Try to infer host/port when launched via `uvicorn webui:app --host ... --port ...`.
    Falls back to 127.0.0.1:6006 to match your examples.
    You can override via env:
      UVICORN_HOST / UVICORN_PORT
      or HOST / PORT
    """
    host = (
        os.getenv("UVICORN_HOST")
        or os.getenv("HOST")
        or "127.0.0.1"
    )
    port_str = (
        os.getenv("UVICORN_PORT")
        or os.getenv("PORT")
        or "6006"
    )
    try:
        port = int(port_str)
    except ValueError:
        port = 6006
    return host, port

def _wait_and_open_browser(url: str, timeout_s: float = 30.0, poll_interval: float = 0.2) -> None:
    """
    Poll the TCP port until it's accepting connections, then open the browser.
    Runs in a background thread so it doesn't block startup.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                # tiny grace period to ensure the app router is ready
                time.sleep(0.25)
                webbrowser.open_new_tab(url)
                return
        except OSError:
            time.sleep(poll_interval)

@app.on_event("startup")
async def _open_browser_on_startup():
    # Respect opt-out
    if os.getenv("AUTO_OPEN_BROWSER", "1") in ("0", "false", "False", "no", "No"):
        return
    host, port = _get_uvicorn_host_port()
    scheme = os.getenv("UVICORN_SCHEME", "http")
    path = os.getenv("APP_ROOT_PATH", "/")  # if you mount elsewhere, set this
    url = f"{scheme}://{host}:{port}{path}"
    # Fire-and-forget thread
    threading.Thread(target=_wait_and_open_browser, args=(url,), daemon=True).start()

# Run with:
#   uvicorn webui:app --host 127.0.0.1 --port 6006
# If you use different host/port, either pass env:
#   UVICORN_HOST=0.0.0.0 UVICORN_PORT=8080 uvicorn webui:app
# or override PORT/HOST. Set AUTO_OPEN_BROWSER=0 to disable.
