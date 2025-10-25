# -*- coding: utf-8 -*-
import os
import re
import json
from datetime import datetime
from typing import Iterable, Dict, Any, List, Tuple, Generator, Optional

from loguru import logger
import yt_dlp


# -------------------------
# Helpers
# -------------------------

def sanitize_title(title: str) -> str:
    """
    Keep letters, digits, underscore, hyphen, Chinese/Japanese/Korean chars, and spaces.
    Collapse consecutive spaces and trim ends.
    """
    if not isinstance(title, str):
        title = str(title) if title is not None else "Unknown"
    # allow: word chars, CJK (CN/JP/KR), spaces, hyphen, underscore, digits
    title = re.sub(
        r"[^\w\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af \-]",
        "",
        title,
    )
    title = re.sub(r"\s+", " ", title).strip()
    return title or "Untitled"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _derive_upload_date(info: Dict[str, Any]) -> str:
    """
    Prefer 'upload_date' (YYYYMMDD). If missing, try 'timestamp'/'release_timestamp'.
    Fallback to '00000000' to keep pipeline moving.
    """
    ud = (info.get("upload_date") or "").strip()
    if re.fullmatch(r"\d{8}", ud):
        return ud

    for key in ("timestamp", "release_timestamp", "epoch"):
        ts = info.get(key)
        if isinstance(ts, (int, float)) and ts > 0:
            try:
                return datetime.utcfromtimestamp(int(ts)).strftime("%Y%m%d")
            except Exception:
                pass

    # last resort
    return "00000000"


def get_target_folder(info: Dict[str, Any], folder_path: str) -> str:
    """
    Build the output folder as <folder_path>/<uploader>/<upload_date> <title>.
    Never returns None (falls back to '00000000' date).
    """
    sanitized_title = sanitize_title(info.get("title", ""))
    sanitized_uploader = sanitize_title(info.get("uploader", "Unknown"))
    upload_date = _derive_upload_date(info)
    output_folder = os.path.join(folder_path, sanitized_uploader, f"{upload_date} {sanitized_title}")
    return output_folder


# -------------------------
# Download
# -------------------------

def download_single_video(info: Dict[str, Any], folder_path: str, resolution: str = "1080p") -> Optional[str]:
    """
    Download one video using yt-dlp.
    Returns the output folder path (where 'download.mp4' will be saved),
    or None if no webpage_url is available.
    """
    webpage_url = info.get("webpage_url")
    if not webpage_url:
        logger.error("Missing 'webpage_url' in info; skipping.")
        return None

    output_folder = get_target_folder(info, folder_path)
    _ensure_dir(output_folder)

    # If already downloaded, short-circuit
    target_mp4 = os.path.join(output_folder, "download.mp4")
    if os.path.exists(target_mp4):
        logger.info(f"Video already downloaded in {output_folder}")
        return output_folder

    # Parse "1080p" -> "1080"
    res_num = (resolution or "").lower().replace("p", "")
    try:
        height_limit = int(res_num)
    except ValueError:
        height_limit = 1080

    # Cookie handling: use cookies.txt if present; otherwise omit the key
    cookiefile = "cookies.txt"
    use_cookiefile = os.path.exists(cookiefile)

    ydl_opts = {
        # Prefer mp4 final container, fallback chain keeps things robust
        "format": f"bestvideo[ext=mp4][height<={height_limit}]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "writethumbnail": True,
        # ensure the final filename becomes download.mp4
        "outtmpl": os.path.join(output_folder, "download.%(ext)s"),
        "ignoreerrors": True,
    }
    if use_cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    # If you prefer using browser cookies instead, uncomment one:
    # ydl_opts["cookiesfrombrowser"] = ("chrome",)              # Chrome
    # ydl_opts["cookiesfrombrowser"] = ("firefox", "default")   # Firefox default profile

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([webpage_url])

    # Some sites may deliver non-mp4 despite merge preference; normalize if needed
    produced = None
    for fn in os.listdir(output_folder):
        if fn.startswith("download.") and fn.split(".")[-1].lower() in ("mp4", "m4v", "mov", "webm", "mkv"):
            produced = os.path.join(output_folder, fn)
            break

    if produced and produced != target_mp4:
        # try to remux to mp4 via ffmpeg only when needed
        if produced.lower().endswith(".mp4"):
            os.replace(produced, target_mp4)
        else:
            os.system(
                f'ffmpeg -loglevel error -y -i "{produced}" -c copy "{target_mp4}"'
            )
            try:
                if os.path.exists(target_mp4):
                    os.remove(produced)
            except Exception:
                pass

    if os.path.exists(target_mp4):
        logger.info(f"Video downloaded in {output_folder}")
        return output_folder

    logger.warning(f"Download finished but {target_mp4} not found in {output_folder}")
    return output_folder  # still return the folder so later steps can decide


def download_videos(info_list: Iterable[Dict[str, Any]], folder_path: str, resolution: str = "1080p") -> Optional[str]:
    """
    Download multiple videos; returns the last video's output folder (unchanged behavior).
    """
    last_folder = None
    for info in info_list:
        if not info:
            continue
        out = download_single_video(info, folder_path, resolution)
        if out:
            last_folder = out
    return last_folder


# -------------------------
# Info fetching
# -------------------------

def get_info_list_from_url(url: Iterable[str] | str, num_videos: int) -> Generator[Dict[str, Any], None, None]:
    """
    Yield video info dicts from one or more URLs (playlist or single video).
    """
    urls = [url] if isinstance(url, str) else list(url)

    ydl_opts = {
        "playlistend": num_videos,
        "ignoreerrors": True,
        "extract_flat": False,   # ensure rich entries (with webpage_url, formats, etc.)
        "retries": 3,
        "socket_timeout": 20,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in urls:
            try:
                result = ydl.extract_info(u, download=False)
            except Exception as e:
                logger.error(f"Failed to extract info for {u}: {e}")
                continue
            if result is None:
                continue
            entries = result.get("entries")
            if entries is not None:
                # Playlist: entries may contain None for removed/private videos
                for video_info in entries:
                    if video_info:
                        yield video_info
            else:
                # Single video
                yield result


# -------------------------
# High-level convenience
# -------------------------

def download_from_url(url: Iterable[str] | str,
                      folder_path: str,
                      resolution: str = "1080p",
                      num_videos: int = 5) -> Tuple[str, str | None, dict | None]:
    """
    Download up to num_videos from given URL(s). Returns:
      (message, example_mp4_path, example_info_json)
    The example_* values come from the last downloaded video (kept for compatibility).
    """
    # Gather metadata first
    video_info_list: List[Dict[str, Any]] = list(get_info_list_from_url(url, num_videos))

    if not video_info_list:
        return "No videos found for the provided URL(s).", None, None

    # Download videos
    example_output_folder = download_videos(video_info_list, folder_path, resolution)

    # Try to read the last video's info JSON (if present)
    download_info_json = None
    example_info_path = os.path.join(example_output_folder, "download.info.json") if example_output_folder else None
    if example_info_path and os.path.exists(example_info_path):
        try:
            with open(example_info_path, "r", encoding="utf-8") as f:
                download_info_json = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read info JSON: {e}")

    example_mp4 = os.path.join(example_output_folder, "download.mp4") if example_output_folder else None
    return f"All videos have been downloaded under the '{folder_path}' folder", example_mp4, download_info_json


# -------------------------
# Manual test
# -------------------------

if __name__ == "__main__":
    # Example usage
    url = 'https://www.youtube.com/watch?v=5aYwU4nj5QA'
    folder_path = "videos"
    os.makedirs(folder_path, exist_ok=True)
    msg, mp4_path, info_json = download_from_url(url, folder_path)
    logger.info(msg)
    if mp4_path:
        logger.info(f"Example MP4: {mp4_path}")
