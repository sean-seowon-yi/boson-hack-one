import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QSlider, QRadioButton, QLineEdit, QPushButton,
                               QFileDialog, QGroupBox)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


class CustomSlider(QWidget):


    def __init__(self, minimum, maximum, step, label, value, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.label = QLabel(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setSingleStep(step)
        self.slider.setValue(value)

        self.value_label = QLabel(str(value))
        self.slider.valueChanged.connect(self.update_value)

        self.layout.addWidget(self.label)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.value_label)

        self.layout.addLayout(slider_layout)
        self.setLayout(self.layout)

    def update_value(self, value):
        self.value_label.setText(str(value))

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.setValue(value)
        self.value_label.setText(str(value))


class FloatSlider(QWidget):


    def __init__(self, minimum, maximum, step, label, value, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.step = step

        self.label = QLabel(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(minimum / step))
        self.slider.setMaximum(int(maximum / step))
        self.slider.setSingleStep(1)
        self.slider.setValue(int(value / step))

        self.value_label = QLabel(f"{value:.2f}")
        self.slider.valueChanged.connect(self.update_value)

        self.layout.addWidget(self.label)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.value_label)

        self.layout.addLayout(slider_layout)
        self.setLayout(self.layout)

    def update_value(self, value):
        float_value = value * self.step
        self.value_label.setText(f"{float_value:.2f}")

    def value(self):
        return self.slider.value() * self.step

    def setValue(self, value):
        self.slider.setValue(int(value / self.step))
        self.value_label.setText(f"{value:.2f}")


class RadioButtonGroup(QWidget):


    def __init__(self, options, label, default_value, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.group_box = QGroupBox(label)
        self.button_layout = QVBoxLayout()

        self.buttons = []
        for option in options:
            option_str = str(option) if option is not None else "None"
            radio = QRadioButton(option_str)
            self.buttons.append((option, radio))
            if option == default_value:
                radio.setChecked(True)
            self.button_layout.addWidget(radio)

        self.group_box.setLayout(self.button_layout)
        self.layout.addWidget(self.group_box)
        self.setLayout(self.layout)

    def value(self):
        for option, button in self.buttons:
            if button.isChecked():
                return option
        return None


class AudioSelector(QWidget):

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.label = QLabel(label)
        self.layout.addWidget(self.label)

        self.file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.browse_button = QPushButton("browse...")
        self.browse_button.clicked.connect(self.browse_file)

        self.file_layout.addWidget(self.file_path)
        self.file_layout.addWidget(self.browse_button)

        self.layout.addLayout(self.file_layout)
        self.setLayout(self.layout)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "choose audio file", "", "audio (*.mp3 *.wav *.ogg)")
        if file_path:
            self.file_path.setText(file_path)

    def value(self):
        return self.file_path.text() if self.file_path.text() else None


class VideoPlayer(QWidget):

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.label = QLabel(label)
        self.layout.addWidget(self.label)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(200)  

        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)

        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)  
        self.audio_output.setVolume(0.7)  

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setToolTip("volume")
        self.volume_slider.valueChanged.connect(self.set_volume)

        self.media_player.errorOccurred.connect(self.handle_error)

        self.controls_layout = QHBoxLayout()
        self.play_button = QPushButton("play")
        self.play_button.clicked.connect(self.play_pause)

        self.stop_button = QPushButton("stop")
        self.stop_button.clicked.connect(self.stop_video)

        self.status_label = QLabel("ready")

        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.stop_button)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("volume:"))
        volume_layout.addWidget(self.volume_slider)

        self.controls_layout.addLayout(volume_layout)
        self.controls_layout.addWidget(self.status_label)

        self.layout.addWidget(self.video_widget)
        self.layout.addLayout(self.controls_layout)
        self.setLayout(self.layout)

        self.video_path = None

    def set_volume(self, volume):
        self.audio_output.setVolume(volume / 100.0)
        self.status_label.setText(f"volume: {volume}%")

    def set_video(self, path):
        if not os.path.exists(path):
            self.status_label.setText(f"Error: No such file")
            return

        self.video_path = path
        try:
            url = QUrl.fromLocalFile(os.path.abspath(path))
            self.media_player.setSource(url)
            self.status_label.setText(f"loaded: {os.path.basename(path)}")
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"error: {str(e)}")

    def play_pause(self):
        if not self.video_path:
            self.status_label.setText("error: no video loaded")
            return

        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("play")
            self.status_label.setText("paused")
        else:
            self.media_player.play()
            self.play_button.setText("pause")
            self.status_label.setText("play")

    def stop_video(self):
        self.media_player.stop()
        self.play_button.setText("play")
        self.status_label.setText("stopped")

    def handle_error(self, error, error_string):
        self.status_label.setText(f"play error: {error_string}")
        print(f"Video player error ({error}): {error_string}")
