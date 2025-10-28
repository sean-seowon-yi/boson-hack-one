# -*- coding: utf-8 -*-
"""
tools/step050_synthesize_video.py
Stable web-playback + dynamic subtitles:
- Atomic ffmpeg writes (temp -> os.replace)
- Output path is ALWAYS the last ffmpeg arg
- Subtitle font size & margins scale by resolution (mode: height/width/auto)
- Width-aware line wrapping so subtitles always fit on screen
- Bottom-centered alignment with safe margins
- -shortest and +faststart for web playback
- Unique publish filename per render to avoid stale caches
- Returns an absolute URL if PUBLIC_BASE_URL is set, else /media/... URL
"""

import json
import os
import shutil
import string
import subprocess
import time
import random
import traceback
import uuid
import urllib.parse
from typing import List, Tuple

from loguru import logger

# ---------------------------------------------------------------------
# Atomic I/O helpers
# ---------------------------------------------------------------------
def _atomic_replace(src_path: str, dst_path: str):
    """Atomically replace dst with src (after fsync)."""
    try:
        with open(src_path, "rb") as f:
            os.fsync(f.fileno())
    except Exception as e:
        logger.debug(f"fsync failed (continuing): {e}")
    os.replace(src_path, dst_path)

def _mktemp_like(path: str, suffix: str = ".part") -> str:
    base, ext = os.path.splitext(path)
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{base}.{rand}{suffix}"

def _file_nontrivial(path: str, min_bytes: int = 1024) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= min_bytes
    except Exception:
        return False

def _run_ffmpeg_atomic(cmd: list, final_out: str, log_tag: str) -> bool:
    """
    Run ffmpeg writing to a temp file, then atomically replace final_out on success.
    Assumes the LAST argument of cmd is the output path.
    """
    tmp_out = _mktemp_like(final_out, ".part.mp4")

    if not cmd or cmd[-1].startswith('-'):
        logger.error(f"{log_tag}: expected output path last, got {cmd[-1] if cmd else 'EMPTY'}")
        return False

    cmd = list(cmd)
    cmd[-1] = tmp_out

    logger.info(f"{log_tag}: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_output = (result.stderr or b'').decode('utf-8', errors='ignore')

    if result.returncode != 0:
        logger.error(f"{log_tag} failed (code {result.returncode}). FFmpeg stderr:\n{stderr_output}")
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        return False

    if not _file_nontrivial(tmp_out):
        logger.error(f"{log_tag} produced empty or tiny file: {tmp_out}")
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        return False

    try:
        _atomic_replace(tmp_out, final_out)
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

    return True

# ---------------------------------------------------------------------
# Subtitle sizing / wrapping helpers
# ---------------------------------------------------------------------
def _compute_subtitle_layout(width: int, height: int, mode: str = "height") -> Tuple[int, int, int, int, int]:
    """
    Returns (font_size, outline, margin_v, margin_lr, max_line_char).
    mode: "height" | "width" | "auto"
    """
    m = (mode or "height").lower()
    # Base size proportional to frame; conservative defaults
    if m == "width":
        base_size = width / 100.0         # ~19px @1920 (will be scaled)
    elif m == "auto":
        base_size = min(width, height) / 45.0
    else:  # "height"
        base_size = height / 45.0         # ~24px @1080 before scaling

    # Optional multiplier via env (e.g., SUB_FONT_SCALE=0.9)
    scale = float(os.getenv("SUB_FONT_SCALE", "0.95"))
    font_size = int(round(max(16.0, base_size * scale)))
    # Hard cap ~9% of height to avoid huge subs
    font_size = min(font_size, int(height * 0.09))

    # Outline & margins
    outline   = max(1, int(round(font_size * 0.12)))   # ~12% of size
    margin_lr = max(12, int(round(width  * 0.05)))     # 5% side margins
    margin_v  = max(12, int(round(height * 0.07)))     # 7% bottom margin

    # Compute wrap width by pixels -> translate to average char count
    usable_w    = max(1, width - 2 * margin_lr)
    # Approx avg glyph width in pixels (works well for Latin & CJK)
    avg_char_w  = max(1.0, 0.53 * font_size)
    max_line_char = max(8, int(usable_w // avg_char_w))
    return font_size, outline, margin_v, margin_lr, max_line_char

def _wrap_text_by_width(text: str, max_chars: int) -> List[str]:
    """
    Greedy line wrap constrained by character count (proxy for pixel width).
    For CJK/no-space languages, slices by run length.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []

    # If there are spaces, wrap by words
    if " " in text:
        current: List[str] = []
        cur_len = 0
        for word in text.split():
            add = len(word) + (1 if current else 0)
            if cur_len + add > max_chars and current:
                chunks.append(" ".join(current))
                current, cur_len = [word], len(word)
            else:
                if current: current.append(word)
                else: current = [word]
                cur_len += add
        if current:
            chunks.append(" ".join(current))
    else:
        # No spaces (CJK, hashtags, etc.): slice hard
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])

    # Keep to a reasonable number of lines; if >3, rebalance to 3
    if len(chunks) > 3:
        merged = " ".join(chunks)
        per = max(8, int(round(len(merged) / 3.0)))
        chunks = [merged[i:i+per].strip() for i in range(0, len(merged), per)][:3]

    return chunks

def split_text(input_data,
               punctuations=['，', '；', '：', '。', '？', '！', '\n', '”']):
    """
    Keep your original semantic splitting, then we will width-wrap each piece later.
    """
    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item["text"]
        sentence_start = 0
        if not text:
            continue
        total_chars = max(1, len(text))
        duration_per_char = (item["end"] - item["start"]) / total_chars
        for i, _char in enumerate(text):
            if i != len(text) - 1 and _char not in punctuations:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and text[i + 1] in punctuations:
                continue
            sentence = text[sentence_start:i + 1]
            sentence_end = start + duration_per_char * len(sentence)
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence.strip(),
                "speaker": speaker
            })
            start = sentence_end
            sentence_start = i + 1
    return output_data

def format_timestamp(seconds):
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"

def generate_srt(translation, srt_path, speed_up=1, max_line_char=30):
    """
    Generates SRT with width-aware wrapping and bottom-aligned rendering (via ASS style later).
    """
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(translation):
            start = format_timestamp(line['start'] / max(1e-6, speed_up))
            end   = format_timestamp(line['end']   / max(1e-6, speed_up))
            text  = (line.get('translation') or '').strip()
            if not text:
                continue

            wrapped_lines = _wrap_text_by_width(text, max_line_char)
            wrapped = "\n".join(wrapped_lines)

            f.write(f'{i + 1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{wrapped}\n\n')

# ---------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------
def get_aspect_ratio(video_path):
    command = [
        'ffprobe', '-hide_banner', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        info = json.loads(result.stdout)
        dims = info['streams'][0]
        w, h = int(dims['width']), int(dims['height'])
        return w / h if h else 16 / 9.0
    except Exception:
        return 16 / 9.0

def convert_resolution(aspect_ratio, resolution='1080p'):
    try:
        base = int(''.join([c for c in resolution if c.isdigit()]))
    except Exception:
        base = 1080
    if aspect_ratio < 1:
        width = base
        height = int(width / max(1e-6, aspect_ratio))
    else:
        height = base
        width = int(height * aspect_ratio)
    width -= width % 2
    height -= height % 2
    return width, height

def _chain_atempo(speed):
    speed = max(0.25, float(speed))
    filters, remaining = [], speed
    while remaining < 0.5:
        filters.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        filters.append(2.0)
        remaining /= 2.0
    filters.append(remaining)
    return [f"atempo={v:.6g}" for v in filters if abs(v - 1.0) > 1e-9] or ["atempo=1.0"]

def _web_safe_video_flags(crf='18', preset='veryfast'):
    return [
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
        '-level', '4.1', '-preset', str(preset), '-crf', str(crf),
        '-vsync', 'cfr', '-movflags', '+faststart',
    ]

def _web_safe_audio_flags():
    return ['-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2']

def _escape_for_ass(path: str) -> str:
    p = os.path.abspath(path).replace('\\', '/')
    p = p.replace(':', r'\:').replace("'", r"\'")
    return p

# ---------------------------------------------------------------------
# Publish + URL helpers
# ---------------------------------------------------------------------
def _publish_for_web(final_path: str) -> str:
    """
    Copy the finished MP4 to a unique, cache-busting filename in the same folder.
    Returns the filesystem path of the published file.
    """
    folder, base = os.path.split(final_path)
    stem, ext = os.path.splitext(base)
    unique = f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    published = os.path.join(folder, unique)
    shutil.copy2(final_path, published)
    return published

def _to_media_url(abs_path: str) -> str:
    """
    Convert an absolute file path under ./videos to a URL.
    If PUBLIC_BASE_URL is set (e.g. http://127.0.0.1:6006), return absolute URL.
    Otherwise, return a relative /media/... path.
    Always add a cache-busting query.
    """
    videos_root = os.path.abspath("videos")
    ap = os.path.abspath(abs_path)
    rel = os.path.relpath(ap, videos_root).replace("\\", "/")
    path = "/media/" + urllib.parse.quote(rel)
    bust = f"?t={int(time.time())}"

    base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if base:
        return f"{base}{path}{bust}"
    return f"{path}{bust}"

# ---------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------
def synthesize_video(
    folder,
    subtitles=True,
    speed_up=1.00,
    fps=30,
    resolution='1080p',
    background_music=None,
    watermark_path=None,
    bgm_volume=0.5,
    video_volume=1.0
):
    translation_path = os.path.join(folder, 'translation.json')
    emotion_audio = os.path.join(folder, 'audio_combined_emotion.wav')
    default_audio = os.path.join(folder, 'audio_combined.wav')
    input_audio = emotion_audio if os.path.exists(emotion_audio) else default_audio
    input_video = os.path.join(folder, 'download.mp4')

    if not all(os.path.exists(p) for p in [translation_path, input_audio, input_video]):
        logger.error(f"Missing required files in {folder}")
        return None  # returning None propagates failure to caller

    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    # Determine output resolution
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    resolution_str = f'{width}x{height}'

    # Dynamic subtitle layout (mode via env: height|width|auto)
    sub_mode = os.getenv("SUB_SCALE_MODE", "height")
    font_size, outline, margin_v, margin_lr, max_line_char = _compute_subtitle_layout(
        width, height, mode=sub_mode
    )

    # Prepare SRT (with dynamic wrap width)
    srt_path = os.path.join(folder, 'subtitles.srt')
    final_video = os.path.join(folder, 'video.mp4')
    generate_srt(translation, srt_path, speed_up, max_line_char=max_line_char)
    srt_path_ass = _escape_for_ass(srt_path)

    # Filters
    video_speed_filter = f"setpts=PTS/{max(1e-6, speed_up)}"
    atempos = _chain_atempo(speed_up)
    audio_speed_filter = ",".join(atempos)

    # Fonts dir
    font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../font"))
    os.makedirs(font_dir, exist_ok=True)

    # libass style with dynamic sizes/margins:
    #  - WrapStyle=2 (smart wrapping)
    #  - Alignment=2 (bottom-center)
    #  - MarginV / MarginL / MarginR set by resolution
    style = (
        f"FontName=SimHei,"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"        # ARGB little-endian (full white)
        f"OutlineColour=&H00000000,"        # black outline
        f"Outline={outline},"
        f"Shadow=0,"
        f"BorderStyle=1,"                   # outline (not box)
        f"Alignment=2,"                     # bottom-center
        f"WrapStyle=2,"
        f"LineSpacing=1,"
        f"MarginV={margin_v},"
        f"MarginL={margin_lr},"
        f"MarginR={margin_lr}"
    )

    # Build filtergraph and inputs
    fc_parts = [f"[0:v]{video_speed_filter}[v0]", f"[1:a]{audio_speed_filter},volume={video_volume}[va]"]
    maps_after_overlay = "[v0]"
    input_list = ['-i', input_video, '-i', input_audio]

    if watermark_path and os.path.exists(watermark_path):
        input_list += ['-i', watermark_path]
        fc_parts += ["[2:v]scale=iw*0.15:ih*0.15[wm]", "[v0][wm]overlay=W-w-10:H-h-10[v1]"]
        maps_after_overlay = "[v1]"

    filter_complex = ";".join(fc_parts)

    # PASS 1: main render (no subtitles yet — we burn them in a second pass for clarity)
    ffmpeg_command = [
        'ffmpeg',
        *input_list,
        '-filter_complex', filter_complex,
        '-map', maps_after_overlay,
        '-map', '[va]',
        '-r', str(fps),
        '-s', resolution_str,
        *_web_safe_video_flags(),
        *_web_safe_audio_flags(),
        '-shortest',
        '-y',
        '-threads', '2',
        final_video,  # output last
    ]
    if not _run_ffmpeg_atomic(ffmpeg_command, final_video, "FFmpeg PASS1"):
        return None

    # PASS 2: background music mix (optional)
    if background_music and os.path.exists(background_music):
        final_video_with_bgm = final_video.replace('.mp4', '_bgm.mp4')
        ffmpeg_command_bgm = [
            'ffmpeg',
            '-i', final_video,
            '-i', background_music,
            '-filter_complex', f'[0:a]volume={video_volume}[v0];[1:a]volume={bgm_volume}[v1];[v0][v1]amix=inputs=2:duration=first[a]',
            '-map', '0:v',
            '-map', '[a]',
            '-r', str(fps),
            *_web_safe_video_flags(),
            *_web_safe_audio_flags(),
            '-shortest',
            '-y',
            '-threads', '2',
            final_video_with_bgm,  # output last
        ]
        if _run_ffmpeg_atomic(ffmpeg_command_bgm, final_video_with_bgm, "FFmpeg BGM"):
            _atomic_replace(final_video_with_bgm, final_video)

    # PASS 3: subtitle burn-in with dynamic, bottom-aligned style
    try:
        if subtitles:
            final_video_with_subtitles = final_video.replace('.mp4', '_subtitles.mp4')
            subtitle_filter = (
                f"subtitles='{srt_path_ass}':charenc=UTF-8:"
                f"fontsdir='{_escape_for_ass(font_dir)}':"
                f"force_style='{style}'"
            )
            if add_subtitles(final_video, srt_path_ass, final_video_with_subtitles, subtitle_filter):
                _atomic_replace(final_video_with_subtitles, final_video)
    except Exception as e:
        logger.warning(f"Subtitle burn-in failed: {e}")
        traceback.print_exc()

    # Publish with a fresh name (bust caches), then convert to URL
    published_fs_path = _publish_for_web(final_video)
    return _to_media_url(published_fs_path)

# ---------------------------------------------------------------------
# Subtitle rendering
# ---------------------------------------------------------------------
def add_subtitles(video_path, srt_path, output_path, subtitle_filter=None, method='ffmpeg'):
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, f"temp_video_{random.randint(1000,9999)}.mp4")
        temp_srt_path = os.path.join(temp_dir, f"temp_srt_{random.randint(1000,9999)}.srt")
        temp_output_path = os.path.join(temp_dir, f"temp_out_{random.randint(1000,9999)}.mp4")

        shutil.copyfile(video_path, temp_video_path)
        shutil.copyfile(srt_path.replace(r"\:", ":"), temp_srt_path)

        font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../font"))
        os.makedirs(font_dir, exist_ok=True)

        # Default style (only used if subtitle_filter not provided)
        default_style = (
            "FontName=SimHei,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "Outline=2,Shadow=0,BorderStyle=1,Alignment=2,WrapStyle=2,LineSpacing=1,MarginV=60,MarginL=60,MarginR=60"
        )
        filter_option = subtitle_filter or (
            f"subtitles='{_escape_for_ass(temp_srt_path)}':charenc=UTF-8:"
            f"fontsdir='{_escape_for_ass(font_dir)}':"
            f"force_style='{default_style}'"
        )

        command = [
            'ffmpeg',
            '-i', temp_video_path,
            '-vf', filter_option,
            *_web_safe_video_flags(),
            *_web_safe_audio_flags(),
            '-shortest',
            '-y',
            '-threads', '2',
            temp_output_path,  # output last
        ]
        return _run_ffmpeg_atomic(command, output_path, "FFmpeg SUBS")
    except Exception as e:
        logger.error(f"Subtitle add failed: {e}")
        return False
    finally:
        for p in [locals().get('temp_video_path'), locals().get('temp_srt_path'), locals().get('temp_output_path')]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# ---------------------------------------------------------------------
# Batch entrypoint
# ---------------------------------------------------------------------
def synthesize_all_video_under_folder(
    folder,
    subtitles=True,
    speed_up=1.00,
    fps=30,
    background_music=None,
    bgm_volume=0.5,
    video_volume=1.0,
    resolution='1080p',
    watermark_path="f_logo.png"
):
    watermark_path = None if not os.path.exists(watermark_path) else watermark_path
    output_url = None
    for root, dirs, files in os.walk(folder):
        if 'download.mp4' in files:
            output_url = synthesize_video(
                root,
                subtitles=subtitles,
                speed_up=speed_up,
                fps=fps,
                resolution=resolution,
                background_music=background_music,
                watermark_path=watermark_path,
                bgm_volume=bgm_volume,
                video_volume=video_volume
            )
    return f'Synthesized all videos under {folder}', output_url


if __name__ == '__main__':
    pass
