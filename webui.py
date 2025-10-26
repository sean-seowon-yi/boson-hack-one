import os
import gradio as gr

from tools.do_everything import do_everything
from tools.utils import SUPPORT_VOICE

# --- Wrapper: forwards new emotion controls to your pipeline safely ---
def run_with_emotion(
    output_folder,
    video_url,
    num_videos,
    download_resolution,
    demucs_model,
    device,
    num_shifts,
    asr_backend,
    whisperx_size,
    batch_size,
    enable_diar,
    min_speakers,
    max_speakers,
    translation_method,
    subtitle_language,
    tts_method,
    tts_target_language,
    edgetts_voice,
    subtitles,
    playback_speed,
    fps,
    bgm_path,
    bgm_vol,
    video_vol,
    output_resolution,
    max_workers,
    max_retries,
    emotion,              # <--- NEW (UI dropdown)
    emotion_strength,     # <--- NEW (UI slider)
):
    try:
        return do_everything(
            output_folder,
            video_url,
            num_videos,
            download_resolution,
            demucs_model,
            device,
            num_shifts,
            asr_backend,
            whisperx_size,
            batch_size,
            enable_diar,
            min_speakers,
            max_speakers,
            translation_method,
            subtitle_language,
            tts_method,
            tts_target_language,
            edgetts_voice,
            subtitles,
            playback_speed,
            fps,
            bgm_path,
            bgm_vol,
            video_vol,
            output_resolution,
            max_workers,
            max_retries,
            emotion=emotion,                    # preferred kwarg path
            emotion_strength=float(emotion_strength),
        )
    except TypeError:
        # Backward-compat: ENV bridge if do_everything doesn't yet accept these kwargs
        os.environ["EMOTION_PRESET"] = str(emotion)
        os.environ["EMOTION_STRENGTH"] = str(emotion_strength)
        return do_everything(
            output_folder,
            video_url,
            num_videos,
            download_resolution,
            demucs_model,
            device,
            num_shifts,
            asr_backend,
            whisperx_size,
            batch_size,
            enable_diar,
            min_speakers,
            max_speakers,
            translation_method,
            subtitle_language,
            tts_method,
            tts_target_language,
            edgetts_voice,
            subtitles,
            playback_speed,
            fps,
            bgm_path,
            bgm_vol,
            video_vol,
            output_resolution,
            max_workers,
            max_retries,
        )

my_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="green")

# One-click pipeline
full_auto_interface = gr.Interface(
    theme=my_theme,
    title="Smart Multilingual Video Dubbing/Translation",
    fn=run_with_emotion,                         # <--- use wrapper
    inputs=[
        gr.Textbox(label="Output folder", value="videos"),
        gr.Textbox(
            label="Video URL",
            placeholder="Enter a YouTube/Bilibili video, playlist, or channel URL",
            value="https://www.youtube.com/watch?v=VowXFWlAXIU",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Number of videos to download", value=5, visible=False),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Download resolution",
            value="1080p",
            visible=False,
        ),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Demucs model",
            value="htdemucs_ft",
            visible=False
        ),
        gr.Radio(["auto", "cuda", "cpu"], label="Device", value="auto", visible=False),
        gr.Slider(minimum=0, maximum=10, step=1, label="Number of shifts", value=5, visible=False),

        # ASR
        gr.Dropdown(["Higgs"], label="ASR backend", value="Higgs"),
        gr.Radio(["large", "medium", "small", "base", "tiny"], label="WhisperX size", value="large", visible=False),
        gr.Slider(minimum=1, maximum=128, step=1, label="Batch size", value=32, visible=False),
        gr.Checkbox(label="Enable speaker diarization", value=True, visible=False),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Min speakers", value=None, visible=False),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Max speakers", value=None, visible=False),

        # Translation
        gr.Dropdown(["LLM"], label="Translation method (LLM uses Boson/Qwen)", value="LLM"),
        gr.Dropdown(
            ["Simplified Chinese (简体中文)", "Traditional Chinese (繁体中文)", "English", "Korean", "Spanish"],
            label="Subtitle language",
            value="Simplified Chinese (简体中文)",
        ),

        # TTS
        gr.Dropdown(["Higgs", "xtts", "cosyvoice"], label="TTS method", value="Higgs"),
        gr.Dropdown(
            ["Chinese (中文)", "English", "Korean", "Spanish", "French"],
            label="TTS target language",
            value="Chinese (中文)",
        ),
        gr.Dropdown(SUPPORT_VOICE, value="zh-CN-XiaoxiaoNeural", label="EdgeTTS voice", visible=False),

        gr.Checkbox(label="Subtitles", value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label="Playback speed", value=1.00, visible=False),
        gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=30, visible=False),
        gr.Audio(label="Background music", sources=["upload"], type="filepath", visible=False),
        gr.Slider(minimum=0, maximum=1, step=0.05, label="BGM volume", value=0.5, visible=False),
        gr.Slider(minimum=0, maximum=1, step=0.05, label="Video volume", value=1.0, visible=False),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Output resolution",
            value="1080p",
            visible=False
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Max workers", value=1, visible=False),
        gr.Slider(minimum=1, maximum=10, step=1, label="Max retries", value=3, visible=False),

        # --- NEW: Emotion controls (auto-tuned via Higgs-understanding in pipeline) ---
        gr.Dropdown(
            ["natural", "happy", "sad", "angry"],
            label="Emotion",
            value="natural",
            info="Auto-tuned after TTS via Higgs understanding. 'natural' skips shaping.",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.6,
            label="Emotion strength",
            info="0=no change, 1=max intensity. Used by the auto-tuner.",
        ),
    ],
    outputs=[gr.Text(label="Status"), gr.Video(label="Sample output")],
    allow_flagging="never",
)

demo = full_auto_interface

demo = demo.queue(concurrency_count=1, max_size=8)

if __name__ == "__main__":
    demo.launch()  # no host/port/share/inbrowser args
