"""WhisperX transcription + alignment + pyannote diarization.

Heavy models are loaded once and cached for the worker process lifetime.
WhisperX wraps pyannote for diarization, so this single module covers the
whole speech->speaker-labeled-segments stage.
"""
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _whisper_model():
    import whisperx

    return whisperx.load_model(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


@lru_cache(maxsize=8)
def _align_model(language_code: str):
    import whisperx

    return whisperx.load_align_model(
        language_code=language_code, device=settings.whisper_device
    )


@lru_cache(maxsize=1)
def _diarize_pipeline():
    # DiarizationPipeline moved to whisperx.diarize in recent releases;
    # fall back to the older top-level location.
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError:  # pragma: no cover - version dependent
        from whisperx import DiarizationPipeline  # type: ignore

    import inspect

    # The HF auth kwarg was renamed use_auth_token -> token across whisperx /
    # pyannote versions; pick whichever this build actually accepts.
    params = inspect.signature(DiarizationPipeline.__init__).parameters
    kwargs = {"device": settings.whisper_device}
    token = settings.hf_token or None
    if "use_auth_token" in params:
        kwargs["use_auth_token"] = token
    elif "token" in params:
        kwargs["token"] = token
    return DiarizationPipeline(**kwargs)


def load_audio(path: str):
    import whisperx

    return whisperx.load_audio(path)


def transcribe(audio) -> dict:
    """Returns {'segments': [...], 'language': 'xx'}."""
    model = _whisper_model()
    return model.transcribe(audio, batch_size=settings.whisper_batch_size)


def align(segments, language_code: str, audio) -> dict:
    import whisperx

    model_a, metadata = _align_model(language_code)
    return whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        settings.whisper_device,
        return_char_alignments=False,
    )


def diarize(audio, aligned_result: dict, num_speakers: int | None = None) -> dict:
    import whisperx

    pipeline = _diarize_pipeline()
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    diarize_segments = pipeline(audio, **kwargs)
    return whisperx.assign_word_speakers(diarize_segments, aligned_result)
