import os
import re
import json
from typing import Iterable, Dict, Any, List, Tuple, Generator, Optional

from loguru import logger
import yt_dlp


# -------------------------
# Helpers
# -------------------------

def sanitize_title(title: str) -> str:
    """
    Keep letters, digits, underscore, hyphen, Chinese chars, and spaces.
    Collapse consecutive spaces and trim ends.
    """
    if not isinstance(title, str):
        title = str(title) if title is not None else "Unknown"
    title = re.sub(r"[^\w\u4e00-\u9fff \d_-]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    # Avoid empty names
    return title or "Untitled"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_target_folder(info: Dict[str, Any], folder_path: str) -> Optional[str]:
    """
    Build the output folder as <folder_path>/<uploader>/<upload_date> <title>.
    Returns None if upload_date is unavailable.
    """
    sanitized_title = sanitize_title(info.get("title", ""))
    sanitized_uploader = sanitize_title(info.get("uploader", "Unknown"))
    upload_date = info.get("upload_date", "Unknown")
    if upload_date == "Unknown":
        return None
    output_folder = os.path.join(folder_path, sanitized_uploader, f"{upload_date} {sanitized_title}")
    return output_folder


# -------------------------
# Download
# -------------------------

def download_single_video(info: Dict[str, Any], folder_path: str, resolution: str = "1080p") -> Optional[str]:
    """
    Download one video using yt-dlp.
    Returns the output folder path (where 'download.mp4' will be saved),
    or None if we can't determine upload_date.
    """
    sanitized_title = sanitize_title(info.get("title", ""))
    sanitized_uploader = sanitize_title(info.get("uploader", "Unknown"))
    upload_date = info.get("upload_date", "Unknown")
    if upload_date == "Unknown":
        return None

    output_folder = os.path.join(folder_path, sanitized_uploader, f"{upload_date} {sanitized_title}")
    _ensure_dir(output_folder)

    # If already downloaded, short-circuit
    if os.path.exists(os.path.join(output_folder, "download.mp4")):
        logger.info(f"Video already downloaded in {output_folder}")
        return output_folder

    # Parse "1080p" -> "1080"
    res_num = resolution.lower().replace("p", "")
    try:
        # Fall back gracefully if user passes weird values
        height_limit = int(res_num)
    except ValueError:
        height_limit = 1080

    # Cookie handling: use cookies.txt if present; otherwise omit
    cookiefile = "cookies.txt" if os.path.exists("cookies.txt") else None

    ydl_opts = {
        "format": f"bestvideo[ext=mp4][height<={height_limit}]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "writeinfojson": True,
        "writethumbnail": True,
        # 'download' is the base name; yt-dlp will append proper extension(s)
        "outtmpl": os.path.join(output_folder, "download"),
        "ignoreerrors": True,
        "cookiefile": cookiefile,
        # If you prefer using browser cookies instead, uncomment one:
        # "cookiesfrombrowser": ("chrome",),                # Chrome
        # "cookiesfrombrowser": ("firefox", "default"),     # Firefox default profile
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([info["webpage_url"]])

    logger.info(f"Video downloaded in {output_folder}")
    return output_folder


def download_videos(info_list: Iterable[Dict[str, Any]], folder_path: str, resolution: str = "1080p") -> Optional[str]:
    """
    Download multiple videos; returns the last video's output folder (unchanged behavior).
    """
    last_folder = None
    for info in info_list:
        if info is None:
            continue
        last_folder = download_single_video(info, folder_path, resolution)
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
        # We don't need to set 'format' for metadata-only extraction.
        "dumpjson": True,              # not strictly required for extract_info(..., download=False)
        "playlistend": num_videos,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in urls:
            result = ydl.extract_info(u, download=False)
            if result is None:
                continue
            entries = result.get("entries")
            if entries is not None:
                # Playlist: entries may contain None for removed/private videos
                for video_info in entries:
                    if video_info is not None:
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
        return f"No videos found for the provided URL(s).", None, None

    # Download videos
    example_output_folder = download_videos(video_info_list, folder_path, resolution)

    # Try to read the last video's info JSON (if present)
    download_info_json = None
    example_info_path = os.path.join(example_output_folder, "download.info.json") if example_output_folder else None
    if example_info_path and os.path.exists(example_info_path):
        with open(example_info_path, "r", encoding="utf-8") as f:
            download_info_json = json.load(f)

    example_mp4 = os.path.join(example_output_folder, "download.mp4") if example_output_folder else None
    return f"All videos have been downloaded under the '{folder_path}' folder", example_mp4, download_info_json


# -------------------------
# Manual test
# -------------------------

if __name__ == "__main__":
    # Example usage
    # YouTube
    # url = 'https://www.youtube.com/watch?v=5aYwU4nj5QA'
    # Bilibili
    # url = 'https://www.bilibili.com/video/BV1KZ4y1h7ke/'
    # url = 'https://www.bilibili.com/video/BV1Tt411P72Q/'
    url = 'https://www.bilibili.com/video/BV1kr421M7vz/'
    folder_path = "videos"
    os.makedirs(folder_path, exist_ok=True)
    msg, mp4_path, info_json = download_from_url(url, folder_path)
    logger.info(msg)
    if mp4_path:
        logger.info(f"Example MP4: {mp4_path}")
