# -*- coding: utf-8 -*-
import os
from typing import Any, Optional, Tuple
import gradio as gr

# =========================
# CSS: Left panel border + visible warnings + compact status
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

/* =========================
   ADDITIVE THEME ENHANCEMENTS (color-only)
   ========================= */

/* Inputs/textarea/select: accent focus ring */
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 3px rgba(255,122,0,0.18) !important;
}
/* Dropdown hover/open */
.gradio-container select:hover { background: #fffaf3; border-color: var(--accent); }

/* Slider accent */
.gradio-container input[type="range"] { accent-color: var(--accent); }
.gradio-container input[type="range"]::-webkit-slider-thumb { background: var(--accent); border: 2px solid #fff; }
.gradio-container input[type="range"]::-moz-range-thumb { background: var(--accent); border: 2px solid #fff; }

/* Buttons: soft accent hover/focus */
.gradio-container button:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 3px rgba(255,122,0,0.18) !important;
}
.gradio-container button:hover {
  box-shadow: 0 6px 18px rgba(255,122,0,0.12);
  border-color: var(--accent) !important;
}

/* Output cards */
.gradio-container .gr-video,
.gradio-container .gr-html {
  border: 2px solid var(--soft-blue-100);
  border-radius: 10px;
  box-shadow: 0 2px 14px rgba(20,60,140,0.06);
}
.gradio-container .gr-html > div {
  background: var(--soft-blue-50);
  border-radius: 8px;
  padding: 6px;
}

/* Labels */
.gradio-container label,
.gradio-container .label-wrap label { color: var(--ink-700); }

/* Scrollbars */
.gradio-container ::-webkit-scrollbar { height: 10px; width: 10px; }
.gradio-container ::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--accent), var(--accent-600));
  border-radius: 8px;
}
.gradio-container ::-webkit-scrollbar-track { background: #fff6ec; }

/* Checkboxes/radios focus */
.gradio-container input[type="checkbox"]:focus-visible,
.gradio-container input[type="radio"]:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
}

/* Right-panel subtle card */
.gradio-container .main > .container > .grid > :last-child {
  box-shadow: inset 0 0 0 1px #f1f1f1, 0 8px 24px rgba(0,0,0,0.05);
  border-radius: 12px;
  background: #ffffff;
}

/* =========================
   New additive components
   ========================= */

/* TABS (Gradio v4 uses .gr-tabs / .gr-tab / [data-selected]) */
.gradio-container .gr-tabs { border-bottom: 2px solid #f3f3f3; }
.gradio-container .gr-tabs .gr-tab {
  position: relative;
  color: var(--ink-600);
}
.gradio-container .gr-tabs .gr-tab:hover { background: #fff8ef; }
.gradio-container .gr-tabs .gr-tab[aria-selected="true"] {
  color: var(--ink-900);
  font-weight: 600;
}
.gradio-container .gr-tabs .gr-tab[aria-selected="true"]::after {
  content: "";
  position: absolute; left: 8px; right: 8px; bottom: -2px; height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent-600));
  border-radius: 3px;
}

/* ACCORDIONS */
.gradio-container .gr-accordion > .label {
  border-left: 4px solid var(--accent);
  background: #fffaf3;
}
.gradio-container .gr-accordion > .label:hover { background: #fff4e5; }
.gradio-container .gr-accordion[open] > .label {
  background: #fff4e5;
  box-shadow: inset 0 0 0 1px #ffe1c7;
}

/* TOOLTIPS */
.gradio-container .gr-tooltip,
.gradio-container [role="tooltip"] {
  background: #1d1d1d !important;
  color: #fff !important;
  border: 1px solid var(--accent) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

/* TABLES (Markdown & Dataframes) */
.gradio-container .prose table thead th {
  background: var(--soft-blue-100);
  color: #1b2a4b;
  border-bottom: 2px solid #cddfff;
}
.gradio-container .prose table tbody tr:hover {
  background: #f7fbff;
}
.gradio-container .prose table td, 
.gradio-container .prose table th {
  border-color: #e9eefb;
}

/* FILE UPLOADS */
.gradio-container .gr-file {
  border: 2px dashed #ffd1ad;
  background: #fffaf3;
}
.gradio-container .gr-file:hover {
  border-color: var(--accent);
  background: #fff4e5;
}

/* GALLERY SELECTION */
.gradio-container .gr-gallery .thumbnail.selected {
  box-shadow: 0 0 0 3px var(--accent) inset, 0 6px 18px rgba(255,122,0,0.18);
  border-radius: 8px;
}

/* MODALS / DIALOGS */
.gradio-container dialog[open] {
  border: 1px solid #ffe1c7;
  box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}
.gradio-container dialog::backdrop {
  background: rgba(255,122,0,0.14);
}

/* PROGRESS BARS */
.gradio-container progress,
.gradio-container .progress {
  accent-color: var(--accent);
}
.gradio-container .progress > div {
  background: linear-gradient(90deg, var(--accent), var(--accent-600));
}

/* CODE BLOCKS (Markdown) */
.gradio-container .prose pre, 
.gradio-container .prose code {
  border: 1px solid #f2d2b8;
  background: #fffaf3;
}
.gradio-container .prose pre code { background: transparent; }

/* LINKS (Markdown) */
.gradio-container .prose a { color: var(--accent-700); }
.gradio-container .prose a:hover { text-decoration: underline; }

/* DISABLED STATES */
.gradio-container *:disabled {
  opacity: 0.75;
  border-color: #ffd9bf !important;
}

/* SMALL BADGES FOR DROPDOWNS ON FOCUS */
.gradio-container select:active,
.gradio-container select:focus-visible {
  background-image: linear-gradient(0deg, #fff8ef, #ffffff);
  border-color: var(--accent);
}

/* HEADER TITLES (subtle) */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  color: var(--ink-900);
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
    import re
    if not isinstance(url, str):
        return None
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{6,})",
        r"youtu\\.be/([A-Za-z0-9_-]{6,})",
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
# Extract translated video
# =========================
def _pluck_translated_video(res: Any) -> Tuple[str, Optional[str]]:
    status = "Fail"
    translated = None
    try:
        if isinstance(res, dict):
            for k in (
                "output_video", "translated_video", "final_video",
                "out_video", "output", "dubbed_video",
            ):
                v = res.get(k)
                if isinstance(v, str) and (v.endswith(".mp4") or v.startswith("http")):
                    translated = v
                    break
            if translated is None and isinstance(res.get("videos"), (list, tuple)):
                for v in res["videos"]:
                    if isinstance(v, str) and (v.endswith(".mp4") or v.startswith("http")):
                        translated = v
                        break
        elif isinstance(res, (list, tuple)):
            for v in res:
                if isinstance(v, str) and (v.endswith(".mp4") or v.startswith("http")):
                    translated = v
        elif isinstance(res, str):
            if res.endswith(".mp4") or res.startswith("http"):
                translated = res
        if translated:
            status = "Success"
    except Exception:
        status = "Fail"
    return status, translated

# =========================
# Main callback (8 inputs)
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
):
    subtitle_lang_code = _normalize_lang(subtitle_language)
    tts_lang_code = _normalize_lang(tts_target_language)

    if subtitle_lang_code != tts_lang_code:
        print("Note: Subtitle language and TTS language differ; alignment may suffer.")

    from tools.do_everything import do_everything

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
    return status, translated, _build_youtube_embed(video_url)

# =========================
# Interface
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

CUSTOM_CSS = """
/* ---------- Palette (unchanged, additive) ---------- */
:root{
  --accent:#ff7a00; --accent-600:#e86d00; --accent-700:#cc6000;
  --success:#107832; --success-bg:#e7f7ea;
  --danger:#b32626; --danger-bg:#fde9e9;
  --warn-bg:#fff4e5; --ink-900:#2a2a2a; --ink-700:#3b3b3b; --ink-600:#4a4a4a;
  --soft-blue-50:#eef5ff; --soft-blue-100:#e6f0ff;
}

/* Keep your existing border/padding; add a super-subtle backdrop only on left */
.gradio-container .main > .container > .grid > :first-child{
  border:3px solid #ff7a00!important;
  border-radius:10px!important;
  padding:12px!important;
  box-shadow:0 0 0 2px rgba(255,122,0,0.08), 0 6px 22px rgba(0,0,0,0.06);
  background-image: radial-gradient(1200px 400px at 0% -10%, rgba(255,122,0,0.06), transparent 60%),
                    radial-gradient(800px 300px at 100% 110%, rgba(232,109,0,0.05), transparent 60%);
}

/* ---------- Status badge (unchanged) ---------- */
#status input{font-weight:700;text-transform:uppercase;letter-spacing:.06em;font-size:.9rem;padding:4px 10px!important;border-radius:10px!important;width:150px!important;min-height:32px!important;}
#status input[value="Success"]{color:#107832!important;background:#e7f7ea!important;border:1px solid #a9e0b7!important;}
#status input[value="Fail"]{color:#b32626!important;background:#fde9e9!important;border:1px solid #f4b2b2!important;}

/* ---------- Warning callouts (unchanged) ---------- */
#subtitle_lang_box::after{
  content:"Warning: Subtitle language and TTS target language should match for best alignment.";
  display:block;background:#fff4e5;border-left:4px solid #ff7a00;padding:6px 10px;border-radius:6px;margin-top:6px;color:#5b3600;font-weight:600;font-size:.9rem;
}
#emotion_box::after{
  content:"Warning: Emotion is still in development — output may sound choppy.";
  display:block;background:#fff4e5;border-left:4px solid #ff7a00;padding:6px 10px;border-radius:6px;margin-top:6px;color:#5b3600;font-weight:600;font-size:.9rem;
}

/* =========================
   LEFT PANEL–ONLY ENHANCEMENTS
   ========================= */
.left-panel-scope,
.gradio-container .main > .container > .grid > :first-child{
  --lp-border:#ffd1ad;
  --lp-fill:#fffaf3;
  --lp-fill-strong:#fff4e5;
  --lp-ring:rgba(255,122,0,.18);
}

/* Inputs/selects/textarea — accent focus within left panel only */
.gradio-container .main > .container > .grid > :first-child input:focus {
},
.gradio-container .main > .container > .grid > :first-child textarea:focus,
.gradio-container .main > .container > .grid > :first-child select:focus{
  outline:2px solid var(--accent)!important; outline-offset:2px!important; box-shadow:0 0 0 3px var(--lp-ring)!important;
}

/* Subtle field wrapping to group controls */
.gradio-container .main > .container > .grid > :first-child .gr-block{
  background:var(--lp-fill);
  border:1px solid var(--lp-border);
  border-radius:10px;
  padding:10px;
}
.gradio-container .main > .container > .grid > :first-child .gr-block + .gr-block{
  margin-top:10px;
}

/* Labels inside left column: stronger contrast + tiny accent underline */
.gradio-container .main > .container > .grid > :first-child label,
.gradio-container .main > .container > .grid > :first-child .label-wrap label{
  color:var(--ink-700);
  position:relative;
}
.gradio-container .main > .container > .grid > :first-child .label-wrap label::after{
  content:"";
  position:absolute; left:0; bottom:-4px; width:22px; height:2px;
  background:linear-gradient(90deg, var(--accent), var(--accent-600));
  border-radius:2px;
}

/* Buttons in left panel: accent ring + gentle hover */
.gradio-container .main > .container > .grid > :first-child button:focus-visible{
  outline:2px solid var(--accent)!important; outline-offset:2px!important; box-shadow:0 0 0 3px var(--lp-ring)!important;
}
.gradio-container .main > .container > .grid > :first-child button:hover{
  box-shadow:0 6px 18px rgba(255,122,0,0.12); border-color:var(--accent)!important;
}

/* Sliders in left panel */
.gradio-container .main > .container > .grid > :first-child input[type="range"]{ accent-color:var(--accent); }
.gradio-container .main > .container > .grid > :first-child input[type="range"]::-webkit-slider-thumb{ background:var(--accent); border:2px solid #fff; }
.gradio-container .main > .container > .grid > :first-child input[type="range"]::-moz-range-thumb{ background:var(--accent); border:2px solid #fff; }

/* Dropdowns in left panel */
.gradio-container .main > .container > .grid > :first-child select:hover{ background:#fffaf3; border-color:var(--accent); }
.gradio-container .main > .container > .grid > :first-child select:focus-visible{
  background-image:linear-gradient(0deg,#fff8ef,#ffffff); border-color:var(--accent);
}

/* Tabs in left panel */
.gradio-container .main > .container > .grid > :first-child .gr-tabs{ border-bottom:2px solid #f3f3f3; }
.gradio-container .main > .container > .grid > :first-child .gr-tabs .gr-tab{ color:var(--ink-600); }
.gradio-container .main > .container > .grid > :first-child .gr-tabs .gr-tab:hover{ background:#fff8ef; }
.gradio-container .main > .container > .grid > :first-child .gr-tabs .gr-tab[aria-selected="true"]{
  color:var(--ink-900); font-weight:600; position:relative;
}
.gradio-container .main > .container > .grid > :first-child .gr-tabs .gr-tab[aria-selected="true"]::after{
  content:""; position:absolute; left:8px; right:8px; bottom:-2px; height:3px;
  background:linear-gradient(90deg, var(--accent), var(--accent-600)); border-radius:3px;
}

/* Accordions in left panel */
.gradio-container .main > .container > .grid > :first-child .gr-accordion > .label{
  border-left:4px solid var(--accent); background:var(--lp-fill);
}
.gradio-container .main > .container > .grid > :first-child .gr-accordion > .label:hover{ background:var(--lp-fill-strong); }
.gradio-container .main > .container > .grid > :first-child .gr-accordion[open] > .label{
  background:var(--lp-fill-strong); box-shadow:inset 0 0 0 1px #ffe1c7;
}

/* File uploads in left panel */
.gradio-container .main > .container > .grid > :first-child .gr-file{
  border:2px dashed #ffd1ad; background:var(--lp-fill);
}
.gradio-container .main > .container > .grid > :first-child .gr-file:hover{
  border-color:var(--accent); background:var(--lp-fill-strong);
}

/* Checkbox/Radio focus in left panel */
.gradio-container .main > .container > .grid > :first-child input[type="checkbox"]:focus-visible,
.gradio-container .main > .container > .grid > :first-child input[type="radio"]:focus-visible{
  outline:2px solid var(--accent)!important; outline-offset:2px!important;
}

/* Tiny section dividers between groups in left panel */
.gradio-container .main > .container > .grid > :first-child .gr-group + .gr-group{
  border-top:1px dashed #ffd9bf; margin-top:10px; padding-top:12px;
}

/* Keep right panel styles from your prior version (no changes) */
#status input[value="Success"]{color:#107832!important;background:#e7f7ea!important;border:1px solid #a9e0b7!important;}
#status input[value="Fail"]{color:#b32626!important;background:#fde9e9!important;border:1px solid #f4b2b2!important;}
/* Output frames on right remain softly blue as before */
.gradio-container .gr-video,
.gradio-container .gr-html{ border:2px solid var(--soft-blue-100); border-radius:10px; box-shadow:0 2px 14px rgba(20,60,140,0.06); }
.gradio-container .gr-html > div{ background:var(--soft-blue-50); border-radius:8px; padding:6px; }
"""


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=6006, share=True, inbrowser=True)
