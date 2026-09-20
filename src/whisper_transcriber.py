from faster_whisper import WhisperModel


MODEL_SIZE = "tiny"


_model = None


def load_whisper():
    global _model

    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

    return _model


def transcribe_audio(audio_path):
    model = load_whisper()

    segments, info = model.transcribe(
        audio_path,
        language="te",
        beam_size=5
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return text