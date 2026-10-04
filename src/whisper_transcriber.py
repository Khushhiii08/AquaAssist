import whisper
import torch
import subprocess
import os

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Loading Whisper 'base' model on {DEVICE}...")
model = whisper.load_model("base", device=DEVICE)

def standardize_audio_for_whisper(raw_audio_path):
    """
    Leverages system-level FFmpeg to resample browser-recorded audio into 16kHz mono WAV.
    Safely bypasses this step if run locally without FFmpeg installed.
    """
    processed_path = "temp_16kHz_mono.wav"
    command = [
        "ffmpeg", "-y", "-i", raw_audio_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", processed_path
    ]
    
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return processed_path if os.path.exists(processed_path) else raw_audio_path
    except FileNotFoundError:
        # Prevents crashing if the user runs Streamlit locally on Windows before building the Docker image
        print("[!] WARNING: FFmpeg not found on host machine! Skipping resampling. Ensure you use Docker for production.")
        return raw_audio_path
    except Exception as e:
        print(f"[!] FFmpeg processing failed: {e}")
        return raw_audio_path

def transcribe_audio(audio_path):
    """
    Task 2.1: Multimodal Normalization.
    Translates Telugu farmer audio into an English text query.
    """
    try:
        clean_audio_path = standardize_audio_for_whisper(audio_path)
        
        result = model.transcribe(
            clean_audio_path,
            language="te",
            task="translate",
            fp16=False
        )
        return result["text"].strip()
    except Exception as e:
        print(f"[!] Whisper Transcription Error: {e}")
        return ""