---
title: pentalingual
app_file: webui.py
sdk: gradio
sdk_version: 5.49.1
---
# Intelligent Multi-language AI Dubbing/Translation Tool



## Introduction


Key features include:

- **Multi-language Support**: Offers dubbing and subtitle translation in English, Chinese, Korean, French, and Spanish.
- **AI Speech Recognition**: Employs advanced AI for speech-to-text conversion and speaker recognition.
- **Large Language Model Translation**: Uses leading language models like Qwen for fast and accurate translations.
- **AI Voice Cloning**: Utilizes voice cloning to generate speech closely matching the original video's tone and emotion.
- **Digital Sync Technology**: Synchronizes dubbing with video visuals, enhancing realism.
- **Flexible Upload and Translation**: Users can upload videos, choose translation languages, and standards.

---

## Installation and Usage Guide

### Test Environment

This guide applies to the following test environments:

- Python 3.10, PyTorch 2.3.1, CUDA 12.1
- Python 3.10, PyTorch 2.3.1, CUDA 11.8

Follow the steps below to install. 

### 1. Clone the Repository

First, clone the `pentalingual` repository to your local machine and initialize submodules.

```bash
# Clone the project to your local machine
git clone https://github.com/sean-seowon-yi/pentalingual

# Navigate to the project directory
cd pentalingual
```

### 2. Install Dependencies

Before proceeding, please create a new Python environment and install the required dependencies.

```bash
# Create a conda environment named 'pentalingual' and specify Python version 3.10
conda create -n pentalingual python=3.10 -y

# Activate the newly created environment
conda activate pentalingual

# Install the ffmpeg tool
# Install ffmpeg using conda
conda install -y -c conda-forge ffmpeg==7.0.2
# Install ffmpeg using a mirror
# conda install -y -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ ffmpeg==7.0.2

# Pynini (needed by WeTextProcessing) — conda is the hassle-free route
conda install -y -c conda-forge pynini==2.1.5

# Fast pip resolver + prebuilt wheels
python -m pip install -U pip

# Install requirements
pip install -r requirements.txt

# Submodule-specific requirements
pip install -r requirements_module.txt
```

### 3. Configure Environment Variables

Before running the program, you need to configure the necessary environment variables. In the root directory of the project, create a `.env` file by renaming `env.example` and filling in the following variables:

- `BOSON_API_KEY`: Your Boson API key.



### 4. Run the Application

Before launching the application, run the following commands to automatically download the required models (including Qwen, XTTSv2, and faster-whisper-large-v3):

```bash
# For Linux
bash scripts/download_models.sh

Once the download is complete, launch the WebUI interface using the following command:

```bash
python webui.py
```


## Detailed Features and Technical Details

### Automatic Video Download

**yt-dlp** is a powerful open-source command-line tool designed for downloading video and audio from YouTube and other websites. This tool offers a wide range of parameter options, allowing users to customize download behavior to their needs. Whether choosing specific formats, resolutions, or extracting audio, yt-dlp provides flexible solutions. It also supports extensive post-processing features, such as automatically adding metadata and renaming files. For more details on parameters and usage, refer to the [yt-dlp official repository](https://github.com/yt-dlp/yt-dlp).

### Vocal Separation

#### Demucs 

**Demucs** is an advanced sound separation model developed by the Facebook research team, designed to separate different sound sources from mixed audio. Although its architecture is simple, Demucs is powerful enough to isolate instruments, voices, and background noise, making it easier for users to perform post-processing and editing. Its user-friendly design has made it a preferred tool for many audio processing applications, including music production and post-production in films. More information can be found on the [Demucs project page](https://github.com/facebookresearch/demucs).


### AI Speech Recognition

#### Higgs

**Higgs** an open-source text-to-audio foundation model pretrained on over ten million hours of diverse audio paired with large-scale text data, built to deliver high-fidelity, expressive speech (including multi-speaker, multilingual, and voice-cloned output) via a unified architecture. It employs a custom audio tokenizer capturing both semantic and acoustic tokens, and a “Dual-FFN” transformer-based backbone to efficiently model long acoustic sequences while maintaining strong language understanding. The model supports zero-shot voice cloning (i.e., generating audio in a reference speaker’s voice without speaker-specific fine-tuning) and multi-speaker dialogue generation, achieving state-of-the-art results

### Large Language Model Translation

#### Qwen

**Qwen** is a localized large language model that supports multi-language translation. Although its performance may not match OpenAI's top models, its open-source nature and local execution make it a cost-effective option. Qwen is capable of handling text translations across various languages and serves as a powerful open-source alternative. 

### Text to Speech

#### XTTS

**XTTS** (Cross-lingual Text-to-Speech) is an advanced multilingual, multi-speaker TTS model designed to generate natural, expressive, and high-quality speech across a wide range of languages and accents. Developed as part of the Coqui TTS ecosystem, XTTS builds upon the Tacotron and VITS architectures, integrating a cross-lingual transfer mechanism that allows it to reproduce speaker identity and emotional tone even in languages the speaker never explicitly recorded. The model supports zero-shot voice cloning — users can synthesize speech in any supported language using only a short reference audio clip of the target speaker. 

#### Higgs

**Higgs** an open-source text-to-audio foundation model pretrained on over ten million hours of diverse audio paired with large-scale text data, built to deliver high-fidelity, expressive speech (including multi-speaker, multilingual, and voice-cloned output) via a unified architecture. It employs a custom audio tokenizer capturing both semantic and acoustic tokens, and a “Dual-FFN” transformer-based backbone to efficiently model long acoustic sequences while maintaining strong language understanding. The model supports zero-shot voice cloning (i.e., generating audio in a reference speaker’s voice without speaker-specific fine-tuning) and multi-speaker dialogue generation, achieving state-of-the-art results

### Key Features
1. **Multi-language support**: Handles speech synthesis tasks in various languages.
2. **Multi-style speech synthesis**: Controls the emotion and tone of speech through commands.
3. **Streaming inference support**: Future plans include real-time streaming inference support.
4. **Cross-lingual support:** Inference across languages different from the training dataset, currently supporting English, Korean, and Chinese.


## Note
In the UI, the caption language and the TTS language should be the same for the funtionality.

---

## License

> [!Caution]
>
> When using this tool, please comply with relevant laws, including copyright, data protection, and privacy laws. Do not use this tool without permission from the original author and/or rights holder.

Follows the Apache License 2.0. When using this tool, please comply with relevant laws, including copyright, data protection, and privacy laws. Do not use this tool without permission from the original author and/or rights holder.

---

## Credit

Our project was inspired by Linly-Dubbing and further optimized for the purpose of intelligent multi-language AI dubbing and translation.

---

## References

In developing this project, I referenced and drew inspiration from several outstanding open-source projects and related resources. Special thanks to the developers and contributors of these projects and the open-source community. Below are the main projects we referenced:

- [YouDub-webui](https://github.com/liuzhao1225/): Provides a feature-rich web interface for downloading and processing YouTube videos, from which we drew much inspiration and technical implementation details.

- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Qwen](https://github.com/QwenLM/Qwen)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Linly-Talker](https://github.com/Kedreamix/Linly-Talker)

---

