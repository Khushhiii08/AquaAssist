"""
AquaAssist: Multimodal Audio Transcription & Normalization
Resamples browser audio recordings to 16 kHz mono WAV via FFmpeg
and transcribes vernacular speech directly to normalized English text.
"""

import os
import subprocess
from typing import Optional
import torch
import whisper

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

_WHISPER_MODEL = None


def get_whisper_model():
    """Lazy loader for Whisper base model to avoid startup memory spikes."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print(f"[*] Loading Whisper base model on {DEVICE}...")
        _WHISPER_MODEL = whisper.load_model("base", device=DEVICE)
    return _WHISPER_MODEL


def standardize_audio_for_whisper(raw_audio_path: str) -> str:
    """
    Standardizes input audio (WebM, OGG, MP3) to 16 kHz mono 16-bit PCM WAV.
    Gracefully returns original path if FFmpeg is absent.
    """
    if not raw_audio_path or not os.path.exists(raw_audio_path):
        return raw_audio_path

    processed_path = "temp_16khz_mono.wav"
    command = [
        "ffmpeg", "-y", "-i", raw_audio_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", processed_path
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(processed_path):
            return processed_path
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return raw_audio_path


def transcribe_audio(audio_path: Optional[str]) -> str:
    """
    Transcribes farmer voice recordings.
    Automatically translates Telugu speech to English text query.
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""

    try:
        clean_path = standardize_audio_for_whisper(audio_path)
        model = get_whisper_model()

        # Whisper task='translate' translates Telugu/vernacular directly into English text
        result = model.transcribe(
            clean_path,
            task="translate",
            fp16=False
        )
        transcribed_text = result.get("text", "").strip()

        # Clean up temporary standardized audio file
        if clean_path != audio_path and os.path.exists(clean_path):
            try:
                os.remove(clean_path)
            except OSError:
                pass

        return transcribed_text
    except Exception as e:
        print(f"[!] Whisper Transcription Error: {e}")
        return ""