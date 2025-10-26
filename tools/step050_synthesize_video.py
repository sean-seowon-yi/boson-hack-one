# -*- coding: utf-8 -*-
import json
import os
import shutil
import string
import subprocess
import time
import random
import traceback

from loguru import logger


def split_text(input_data,
               punctuations=['，', '；', '：', '。', '？', '！', '\n', '”']):
    # Chinese punctuation marks for sentence ending

    def is_punctuation(char):
        return char in punctuations

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
                "translation": sentence,
                "speaker": speaker
            })

            start = sentence_end
            sentence_start = i + 1

    return output_data


def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"


def generate_srt(translation, srt_path, speed_up=1, max_line_char=30):
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(translation):
            start = format_timestamp(line['start'] / max(1e-6, speed_up))
            end = format_timestamp(line['end'] / max(1e-6, speed_up))
            text = (line.get('translation') or '').strip()
            if not text:
                continue
            lines = max(1, len(text) // (max_line_char + 1) + 1)
            avg = min(max(1, round(len(text) / lines)), max_line_char)
            wrapped = '\n'.join([text[i * avg:(i + 1) * avg] for i in range(lines)])

            f.write(f'{i + 1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{wrapped}\n\n')


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
        if h == 0:
            raise ValueError("Height is zero")
        return w / h
    except Exception as e:
        logger.warning(f"ffprobe failed to get aspect ratio ({e}); defaulting to 16:9")
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
    filters = []
    remaining = speed
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
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'high',
        '-level', '4.1',
        '-preset', str(preset),
        '-crf', str(crf),
        '-vsync', 'cfr',
        '-movflags', '+faststart',
    ]


def _web_safe_audio_flags():
    return [
        '-c:a', 'aac',
        '-b:a', '192k',
        '-ar', '48000',
        '-ac', '2',
    ]


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
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')

    if not os.path.exists(translation_path) or not os.path.exists(input_audio) or not os.path.exists(input_video):
        logger.error(f"Missing required files in {folder}")
        return None

    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    srt_path = os.path.join(folder, 'subtitles.srt')
    final_video = os.path.join(folder, 'video.mp4')
    generate_srt(translation, srt_path, speed_up)
    srt_path = os.path.abspath(srt_path).replace('\\', '/')

    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    resolution_str = f'{width}x{height}'

    video_speed_filter = f"setpts=PTS/{max(1e-6, speed_up)}"
    atempos = _chain_atempo(speed_up)
    audio_speed_filter = ",".join(atempos)

    font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../font"))
    os.makedirs(font_dir, exist_ok=True)

    font_size = max(12, int(width / 100))
    outline = max(1, int(round(font_size / 8)))
    style = (
        f"FontName=SimHei,FontSize={font_size},"
        f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline={outline},WrapStyle=2"
    )

    fc_parts = [f"[0:v]{video_speed_filter}[v0]", f"[1:a]{audio_speed_filter},volume={video_volume}[va]"]
    maps_after_overlay = "[v0]"
    input_list = ['-i', input_video, '-i', input_audio]

    if watermark_path:
        watermark_path = os.path.abspath(watermark_path)
        input_list += ['-i', watermark_path]
        fc_parts += ["[2:v]scale=iw*0.15:ih*0.15[wm]", "[v0][wm]overlay=W-w-10:H-h-10[v1]"]
        maps_after_overlay = "[v1]"

    filter_complex = ";".join(fc_parts)

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
        final_video,
        '-y',
        '-threads', '2',
    ]

    logger.info(f"FFmpeg PASS1: {' '.join(ffmpeg_command)}")
    subprocess.run(ffmpeg_command, check=False)

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
            final_video_with_bgm,
            '-y',
            '-threads', '2'
        ]
        logger.info(f"FFmpeg BGM: {' '.join(ffmpeg_command_bgm)}")
        subprocess.run(ffmpeg_command_bgm, check=False)
        try:
            if os.path.exists(final_video):
                os.remove(final_video)
            os.rename(final_video_with_bgm, final_video)
        except Exception as e:
            logger.warning(f"Replacing video after BGM mix failed: {e}")
        time.sleep(0.5)

    try:
        if subtitles:
            final_video_with_subtitles = final_video.replace('.mp4', '_subtitles.mp4')
            subtitle_filter = (
                f"subtitles='{srt_path}':charenc=UTF-8:fontsdir='{font_dir}':"
                f"force_style='{style}'"
            )
            add_subtitles(final_video, srt_path, final_video_with_subtitles, subtitle_filter, 'ffmpeg')

            if os.path.exists(final_video):
                os.remove(final_video)
            os.rename(final_video_with_subtitles, final_video)
            time.sleep(0.5)
    except Exception as e:
        logger.info(f"An error occurred during subtitle burn-in: {e}")
        traceback.format_exc()

    return final_video


def add_subtitles(video_path, srt_path, output_path, subtitle_filter=None, method='ffmpeg'):
    """
    Add subtitles to a video file.
    """
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        def _tmp(name):
            rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            return os.path.join(temp_dir, f"{name}_{rand}.mp4")

        temp_video_path = os.path.abspath(_tmp("temp_video"))
        temp_srt_path = os.path.abspath(os.path.join(temp_dir, f"temp_srt_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}.srt"))
        temp_output_path = os.path.abspath(_tmp("temp_output"))

        if not os.path.exists(video_path):
            logger.error(f"Input video file does not exist: {video_path}")
            return False
        if not os.path.exists(srt_path):
            logger.error(f"Subtitle file does not exist: {srt_path}")
            return False

        shutil.copyfile(video_path, temp_video_path)
        shutil.copyfile(srt_path, temp_srt_path)

        if not os.path.exists(temp_srt_path):
            logger.error(f"Subtitle file does not exist: {temp_srt_path}")
            return False
        if not os.path.exists(temp_video_path):
            logger.error(f"Input video file does not exist: {temp_video_path}")
            return False

        if method == 'moviepy':
            from moviepy import VideoFileClip, TextClip
            from moviepy.video.tools.subtitles import SubtitlesClip

            video = VideoFileClip(temp_video_path)
            generator = lambda txt: TextClip(txt, font='font/SimHei.ttf', fontsize=24, color='white')
            subtitles = SubtitlesClip(temp_srt_path, generator)
            final_video = video.set_subtitles(subtitles)
            final_video.write_videofile(temp_output_path, fps=video.fps)

            if os.path.exists(temp_output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copyfile(temp_output_path, output_path)
                logger.info(f"Subtitle added successfully, output to: {output_path}")
                return True
            else:
                logger.error(f"Output file was not generated: {temp_output_path}")
                return False

        elif method == 'ffmpeg':
            font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../font"))
            os.makedirs(font_dir, exist_ok=True)

            default_style = "FontName=SimHei,FontSize=15,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,WrapStyle=2"

            filter_option = subtitle_filter or (
                f"subtitles='{temp_srt_path}':charenc=UTF-8:fontsdir='{font_dir}':"
                f"force_style='{default_style}'"
            )

            command = [
                'ffmpeg',
                '-i', f"{temp_video_path}",
                '-vf', f"{filter_option}",
                *_web_safe_video_flags(),
                *_web_safe_audio_flags(),
                f"{temp_output_path}",
                '-y',
                '-threads', '2',
            ]

            logger.info(f"Executing FFmpeg command: {' '.join(command)}")

            result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stderr_output = (result.stderr or b'').decode('utf-8', errors='ignore')
            logger.debug(f"FFmpeg output: {stderr_output}")

            if os.path.exists(temp_output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copyfile(temp_output_path, output_path)
                logger.info(f"Subtitle added successfully, output to: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg executed successfully but output file not found: {temp_output_path}")
                return False

        else:
            logger.error(f"Unsupported method: {method}. Please use 'moviepy' or 'ffmpeg'")
            return False

    except Exception as e:
        logger.error(f"An error occurred while adding subtitles: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        return False
    finally:
        try:
            for p in [locals().get('temp_video_path'), locals().get('temp_srt_path'), locals().get('temp_output_path')]:
                if p and os.path.exists(p):
                    os.remove(p)
        except Exception as e:
            logger.debug(f"Unable to delete temporary file: {e}")


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
    output_video = None
    for root, dirs, files in os.walk(folder):
        if 'download.mp4' in files:
            output_video = synthesize_video(
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
    return f'Synthesized all videos under {folder}', output_video


if __name__ == '__main__':
    pass
