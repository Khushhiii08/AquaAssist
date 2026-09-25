import whisper
import torch

# Hardware acceleration for edge/local testing
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Loading Whisper 'base' model on {DEVICE}...")
model = whisper.load_model("base", device=DEVICE)

def transcribe_audio(audio_path):
    """
    Task 2.1: Multimodal Normalization.
    Directly translates Telugu farmer audio into an English text query.
    """
    try:
        result = model.transcribe(
            audio_path,
            language="te",        # Optimize for Telugu acoustics
            task="translate",     # Instantly output English text
            fp16=False            # Safe fallback for edge hardware
        )
        return result["text"].strip()
    except Exception as e:
        print(f"[!] Whisper Transcription Error: {e}")
        return ""