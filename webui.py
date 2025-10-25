import gradio as gr

from tools.do_everything import do_everything
from tools.utils import SUPPORT_VOICE

my_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="green")

# One-click pipeline
full_auto_interface = gr.Interface(
    theme=my_theme,
    title="Smart Multilingual Video Dubbing/Translation",
    fn=do_everything,
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

        # ASR — add Higgs path
        gr.Dropdown(["Higgs", "WhisperX", "FunASR"], label="ASR backend", value="Higgs"),
        gr.Radio(["large", "medium", "small", "base", "tiny"], label="WhisperX size", value="large", visible=False),
        gr.Slider(minimum=1, maximum=128, step=1, label="Batch size", value=32, visible=False),
        gr.Checkbox(label="Enable speaker diarization", value=True, visible=False),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Min speakers", value=None, visible=False),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Max speakers", value=None, visible=False),

        # Translation — route “LLM” to Boson/Qwen in your code; remove OpenAI/Ernie
        gr.Dropdown(
            ["LLM",
            #  "Google Translate", "Bing Translate"
             ],
            label="Translation method (LLM uses Boson/Qwen)",
            value="LLM",
        ),
        # UPDATED target-language choices for translation (kept default)
        gr.Dropdown(
        ["Simplified Chinese (简体中文)", "Traditional Chinese (繁体中文)", "English", "Korean", "Spanish"],
        label="Subtitle language",
        value="Simplified Chinese (简体中文)",
        ),

        # TTS — keep the 3 supported methods in your router
        gr.Dropdown(["Higgs", "xtts", "cosyvoice", "EdgeTTS"], label="TTS method", value="Higgs"),
        # UPDATED TTS target-language choices (deduped; Tamil shown but works best with EdgeTTS)
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
    ],
    outputs=[gr.Text(label="Status"), gr.Video(label="Sample output")],
    allow_flagging="never",
)

app = full_auto_interface

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=6006,
        share=True,
        inbrowser=True,
    )
