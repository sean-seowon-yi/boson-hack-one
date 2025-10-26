
import os
import torch
import numpy as np
from dotenv import load_dotenv
# from .step021_asr_whisperx import whisperx_transcribe_audio
from .step023_asr_higgs import higgs_transcribe_audio
from .utils import save_wav
import json
import librosa
from loguru import logger
load_dotenv()

def merge_segments(
    segments,
    max_gap: float = 0.40,          # seconds: merge if next.start - prev.end <= max_gap
    max_chars: int = 120,           # don't let merged text grow too long
    joiner: str = " "
):
    """
    Merge short/adjacent ASR segments into larger sentences.
    Safely handles empty text to avoid IndexError.
    Input:  List[{"start": float, "end": float, "text": str, "speaker": str}]
    Output: Same shape, merged.
    """
    if not segments:
        return []

    # sort just in case
    segs = sorted(segments, key=lambda x: float(x.get("start", 0.0)))

    # sentence-ending characters across languages
    ending = set(list(".!?。！？…」』”’】》") + ["]", "）", ")"])

    merged = []
    buffer = None

    def _clean_text(s):
        # normalize text to avoid None and trailing/leading spaces
        return (s or "").strip()

    for seg in segs:
        text = _clean_text(seg.get("text", ""))
        # Skip segments with no text at all
        if not text:
            continue

        start = float(seg.get("start", 0.0))
        end   = float(seg.get("end", start))
        spk   = seg.get("speaker", "SPEAKER_00")

        if buffer is None:
            buffer = {
                "start": start,
                "end": end,
                "text": text,
                "speaker": spk,
            }
            continue

        # Only merge if:
        #   1) temporal gap is small
        #   2) same speaker (optional but typical for diarized streams)
        #   3) previous buffer doesn't already end with sentence punctuation
        #   4) max length constraint respected
        gap = max(0.0, start - float(buffer["end"]))
        prev_text = _clean_text(buffer["text"])
        prev_last = prev_text[-1] if prev_text else ""
        prev_ends_sentence = prev_last in ending

        can_merge = (
            gap <= max_gap
            and spk == buffer["speaker"]
            and not prev_ends_sentence
            and (len(prev_text) + 1 + len(text) <= max_chars)
        )

        if can_merge:
            buffer["text"] = (prev_text + joiner + text).strip()
            buffer["end"] = max(float(buffer["end"]), end)
        else:
            merged.append(buffer)
            buffer = {
                "start": start,
                "end": end,
                "text": text,
                "speaker": spk,
            }

    if buffer is not None:
        merged.append(buffer)

    return merged


def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.05
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end']+delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end]
        speaker_dict[segment['speaker']] = np.concatenate((speaker_dict.get(
            segment['speaker'], np.zeros((0, ))), speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)
    
    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(
            speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path)


def transcribe_audio(method, folder, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True,min_speakers=None, max_speakers=None):
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        return False
    
    logger.info(f'Transcribing {wav_path}')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # if method == 'WhisperX':
    #     transcript = whisperx_transcribe_audio(wav_path, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers)
    if method == 'FunASR':
        transcript = funasr_transcribe_audio(wav_path, device, batch_size, diarization)
    elif method == 'Higgs':
        transcript = higgs_transcribe_audio(wav_path, device, batch_size, diarization)
    else:
        logger.error('Invalid ASR method')
        raise ValueError('Invalid ASR method')

    transcript = merge_segments(transcript)
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    generate_speaker_audio(folder, transcript)
    return transcript

def transcribe_all_audio_under_folder(folder, asr_method, whisper_model_name: str = 'large', device='auto', batch_size=32, diarization=False, min_speakers=None, max_speakers=None):
    transcribe_json = None
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            transcribe_json = transcribe_audio(asr_method, root, whisper_model_name, 'models/ASR/whisper', device, batch_size, diarization, min_speakers, max_speakers)
        elif 'transcript.json' in files:
            transcribe_json = json.load(open(os.path.join(root, 'transcript.json'), 'r', encoding='utf-8'))

            # logger.info(f'Transcript already exists in {root}')
    return f'Transcribed all audio under {folder}', transcribe_json

if __name__ == '__main__':
    _, transcribe_json = transcribe_all_audio_under_folder('videos', 'WhisperX')
    print(transcribe_json)
    # _, transcribe_json = transcribe_all_audio_under_folder('videos', 'FunASR')    
    # print(transcribe_json)