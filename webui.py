import gradio as gr
from tools.step000_video_downloader import download_from_url
from tools.step010_demucs_vr import separate_all_audio_under_folder
from tools.step020_asr import transcribe_all_audio_under_folder
from tools.step030_translation import translate_all_transcript_under_folder
from tools.step040_tts import generate_all_wavs_under_folder
from tools.step050_synthesize_video import synthesize_all_video_under_folder
from tools.do_everything import do_everything
from tools.utils import SUPPORT_VOICE

# One-click pipeline
full_auto_interface = gr.Interface(
    fn=do_everything,
    inputs=[
        gr.Textbox(label="Output folder", value="videos"),
        gr.Textbox(
            label="Video URL",
            placeholder="Enter a YouTube/Bilibili video, playlist, or channel URL",
            value="https://www.bilibili.com/video/BV1kr421M7vz/",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Number of videos to download", value=5),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Download resolution",
            value="1080p",
        ),

        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Demucs model",
            value="htdemucs_ft",
        ),
        gr.Radio(["auto", "cuda", "cpu"], label="Device", value="auto"),
        gr.Slider(minimum=0, maximum=10, step=1, label="Number of shifts", value=5),

        # ASR — add Higgs path
        gr.Dropdown(["WhisperX", "FunASR", "Higgs"], label="ASR backend", value="WhisperX"),
        gr.Radio(["large", "medium", "small", "base", "tiny"], label="WhisperX size", value="large"),
        gr.Slider(minimum=1, maximum=128, step=1, label="Batch size", value=32),
        gr.Checkbox(label="Enable speaker diarization", value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Min speakers", value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Max speakers", value=None),

        # Translation — route “LLM” to Boson/Qwen in your code; remove OpenAI/Ernie
        gr.Dropdown(
            ["LLM", "Google Translate", "Bing Translate"],
            label="Translation method (LLM uses Boson/Qwen)",
            value="LLM",
        ),
        # UPDATED target-language choices for translation (kept default)
        gr.Dropdown(
            ["简体中文", "繁体中文", "English", "Japanese", "Korean", "Spanish", "Tamil", "Cantonese"],
            label="Target language",
            value="简体中文",
        ),

        # TTS — keep the 3 supported methods in your router
        gr.Dropdown(["xtts", "cosyvoice", "EdgeTTS", "Higgs"], label="TTS method", value="Higgs"),
        # UPDATED TTS target-language choices (deduped; Tamil shown but works best with EdgeTTS)
        gr.Dropdown(
            ["中文", "English", "Japanese", "Korean", "Spanish", "French", "粤语", "Tamil"],
            label="TTS target language",
            value="中文",
        ),
        gr.Dropdown(SUPPORT_VOICE, value="zh-CN-XiaoxiaoNeural", label="EdgeTTS voice"),

        gr.Checkbox(label="Burn subtitles", value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label="Playback speed", value=1.00),
        gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=30),
        gr.Audio(label="Background music", sources=["upload"]),
        gr.Slider(minimum=0, maximum=1, step=0.05, label="BGM volume", value=0.5),
        gr.Slider(minimum=0, maximum=1, step=0.05, label="Video volume", value=1.0),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Output resolution",
            value="1080p",
        ),

        gr.Slider(minimum=1, maximum=100, step=1, label="Max workers", value=1),
        gr.Slider(minimum=1, maximum=10, step=1, label="Max retries", value=3),
    ],
    outputs=[gr.Text(label="Status"), gr.Video(label="Sample output")],
    allow_flagging="never",
)

# Download interface
download_interface = gr.Interface(
    fn=download_from_url,
    inputs=[
        gr.Textbox(
            label="Video URL",
            placeholder="Enter a YouTube/Bilibili video, playlist, or channel URL",
            value="https://www.bilibili.com/video/BV1kr421M7vz/",
        ),
        gr.Textbox(label="Output folder", value="videos"),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Download resolution",
            value="1080p",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Number of videos to download", value=5),
    ],
    outputs=[gr.Textbox(label="Download status"), gr.Video(label="Example video"), gr.Json(label="Download info")],
    allow_flagging="never",
)

# Demucs (vocal separation)
demucs_interface = gr.Interface(
    fn=separate_all_audio_under_folder,
    inputs=[
        gr.Textbox(label="Video folder", value="videos"),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Demucs model",
            value="htdemucs_ft",
        ),
        gr.Radio(["auto", "cuda", "cpu"], label="Device", value="auto"),
        gr.Checkbox(label="Show progress", value=True),
        gr.Slider(minimum=0, maximum=10, step=1, label="Number of shifts", value=5),
    ],
    outputs=[gr.Text(label="Separation status"), gr.Audio(label="Vocals"), gr.Audio(label="Instruments")],
    allow_flagging="never",
)

# ASR
asr_inference = gr.Interface(
    fn=transcribe_all_audio_under_folder,
    inputs=[
        gr.Textbox(label="Video folder", value="videos"),
        gr.Dropdown(["WhisperX", "FunASR", "Higgs"], label="ASR backend", value="WhisperX"),
        gr.Radio(["large", "medium", "small", "base", "tiny"], label="WhisperX size", value="large"),
        gr.Radio(["auto", "cuda", "cpu"], label="Device", value="auto"),
        gr.Slider(minimum=1, maximum=128, step=1, label="Batch size", value=32),
        gr.Checkbox(label="Enable speaker diarization", value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Min speakers", value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Max speakers", value=None),
    ],
    outputs=[gr.Text(label="ASR status"), gr.Json(label="ASR details")],
    allow_flagging="never",
)

# Translation
translation_interface = gr.Interface(
    fn=translate_all_transcript_under_folder,
    inputs=[
        gr.Textbox(label="Video folder", value="videos"),
        gr.Dropdown(
            ["LLM", "Google Translate", "Bing Translate"],
            label="Translation method (LLM uses Boson/Qwen)",
            value="LLM",
        ),
        # UPDATED target-language choices for translation (kept default)
        gr.Dropdown(
            ["简体中文", "繁体中文", "English", "Japanese", "Korean", "Spanish", "Tamil", "Cantonese"],
            label="Target language",
            value="简体中文",
        ),
    ],
    outputs=[gr.Text(label="Translation status"), gr.Json(label="Summary"), gr.Json(label="Translations")],
    allow_flagging="never",
)

# TTS
tts_interface = gr.Interface(
    fn=generate_all_wavs_under_folder,
    inputs=[
        gr.Textbox(label="Video folder", value="videos"),
        gr.Dropdown(["xtts", "cosyvoice", "EdgeTTS"], label="TTS method", value="xtts"),
        # UPDATED TTS target-language choices (same as One-Click)
        gr.Dropdown(
            ["中文", "English", "Japanese", "Korean", "Spanish", "French", "粤语", "Tamil"],
            label="TTS target language",
            value="中文",
        ),
        gr.Dropdown(SUPPORT_VOICE, value="zh-CN-XiaoxiaoNeural", label="EdgeTTS voice"),
    ],
    outputs=[gr.Text(label="TTS status"), gr.Audio(label="Synthesized voice"), gr.Audio(label="Original audio")],
    allow_flagging="never",
)

# Synthesize video
synthesize_video_interface = gr.Interface(
    fn=synthesize_all_video_under_folder,
    inputs=[
        gr.Textbox(label="Video folder", value="videos"),
        gr.Checkbox(label="Burn subtitles", value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label="Playback speed", value=1.00),
        gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=30),
        gr.Audio(label="Background music", sources=["upload"], type="filepath"),
        gr.Slider(minimum=0, maximum=1, step=0.05, label="BGM volume", value=0.5),
        gr.Slider(minimum=0, maximum=1, step=0.05, label="Video volume", value=1.0),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Output resolution",
            value="1080p",
        ),
    ],
    outputs=[gr.Text(label="Synthesis status"), gr.Video(label="Final video")],
    allow_flagging="never",
)

# Lip-sync (placeholder)
linly_talker_interface = gr.Interface(
    fn=lambda: None,
    inputs=[
        gr.Textbox(label="Video folder", value="videos"),
        gr.Dropdown(["Wav2Lip", "Wav2Lipv2", "SadTalker"], label="Lip-sync method", value="Wav2Lip"),
    ],
    outputs=[
        gr.Markdown(value="Work in progress. See [https://github.com/Kedreamix/Linly-Talker](https://github.com/Kedreamix/Linly-Talker)"),
        gr.Text(label="Status"),
        gr.Video(label="Lip-sync result"),
    ],
)

my_theme = gr.themes.Soft()

app = gr.TabbedInterface(
    theme=my_theme,
    interface_list=[
        full_auto_interface,
        download_interface,
        demucs_interface,
        asr_inference,
        translation_interface,
        tts_interface,
        synthesize_video_interface,
        linly_talker_interface,
    ],
    tab_names=[
        "One-Click",
        "Download",
        "Vocal Separation",
        "ASR",
        "Translation",
        "TTS",
        "Video Synthesis",
        "Lip-Sync (WIP)",
    ],
    title="Multi-Language AI Dubbing / Translation Toolkit",
)

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=6006,
        share=True,
        inbrowser=True,
    )
