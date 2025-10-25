import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtCore import Qt

# Ensure required modules are importable
try:
    # UI components (side-effects may register widgets/styles)
    from ui_components import (
        CustomSlider,   # noqa: F401
        FloatSlider,    # noqa: F401
        RadioButtonGroup,  # noqa: F401
        AudioSelector,  # noqa: F401
        VideoPlayer,    # noqa: F401
    )

    # Feature tabs
    from tabs.full_auto_tab import FullAutoTab
    from tabs.settings_tab import SettingsTab
    from tabs.download_tab import DownloadTab
    from tabs.demucs_tab import DemucsTab
    from tabs.asr_tab import ASRTab
    from tabs.translation_tab import TranslationTab
    from tabs.tts_tab import TTSTab
    from tabs.video_tab import SynthesizeVideoTab
    from tabs.linly_talker_tab import LinlyTalkerTab

    # Optional heavy tools (app still runs without them)
    try:
        from tools.step000_video_downloader import download_from_url  # noqa: F401
        from tools.step010_demucs_vr import separate_all_audio_under_folder  # noqa: F401
        from tools.step020_asr import transcribe_all_audio_under_folder  # noqa: F401
        from tools.step030_translation import translate_all_transcript_under_folder  # noqa: F401
        from tools.step040_tts import generate_all_wavs_under_folder  # noqa: F401
        from tools.step050_synthesize_video import synthesize_all_video_under_folder  # noqa: F401
        from tools.do_everything import do_everything  # noqa: F401
        from tools.utils import SUPPORT_VOICE  # noqa: F401
    except ImportError as e:
        print(f"Warning: some tool modules could not be imported: {e}")
        SUPPORT_VOICE = [
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunxiNeural",
            "en-US-JennyNeural",
            "ja-JP-NanamiNeural",
        ]

except ImportError as e:
    print(f"Error: failed to initialize application: {e}")
    sys.exit(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linly-Dubbing — Smart Multilingual Video Dubbing/Translation")
        self.resize(1024, 768)

        tabs = QTabWidget()

        # Create tabs
        self.full_auto_tab = FullAutoTab()
        self.settings_tab = SettingsTab()

        # Propagate settings changes to the One-Click tab
        self.settings_tab.config_changed.connect(self.full_auto_tab.update_config)

        # English-only tab labels
        tabs.addTab(self.full_auto_tab, "One-Click")
        tabs.addTab(self.settings_tab, "Settings")
        tabs.addTab(DownloadTab(), "Auto Download")
        tabs.addTab(DemucsTab(), "Vocal Separation")
        tabs.addTab(ASRTab(), "ASR Speech Recognition")
        tabs.addTab(TranslationTab(), "Subtitle Translation")
        tabs.addTab(TTSTab(), "TTS Synthesis")
        tabs.addTab(SynthesizeVideoTab(), "Video Composition")
        tabs.addTab(LinlyTalkerTab(), "Linly-Talker Lip-Sync (WIP)")

        self.setCentralWidget(tabs)


def main():
    # High-DPI: enable crisp UI on modern displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent cross-platform look

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
