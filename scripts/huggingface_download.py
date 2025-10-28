# pip install huggingface_hub
from huggingface_hub import snapshot_download

# https://huggingface.co/coqui/XTTS-v2
snapshot_download('coqui/XTTS-v2', local_dir='models/TTS/XTTS-v2', resume_download=True, local_dir_use_symlinks=False)

# https://huggingface.co/pyannote/speaker-diarization-3.1
# snapshot_download('pyannote/speaker-diarization-3.1', local_dir='models/ASR/whisper/speaker-diarization-3.1', resume_download=True, local_dir_use_symlinks=False)
