"""Whisper speech->English translation + independent pyannote diarization.

Two passes over the same audio, joined later by timestamp overlap:
  * Whisper (task=translate) -> English segments with timestamps
  * pyannote                 -> speaker spans with timestamps (acoustic, no text)

Heavy models are loaded once and cached for the worker process lifetime.
"""
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _whisper_translate_model():
    import whisperx

    # task="translate": Whisper emits ENGLISH directly for any source language,
    # which also makes bilingual (e.g. Korean+English) audio come out uniformly
    # in English instead of being mis-decoded as one forced language.
    return whisperx.load_model(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        task="translate",
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

    # Newer whisperx defaults to the gated `speaker-diarization-community-1`.
    # Pin to a model we have access to (override with DIARIZATION_MODEL).
    import os

    model_name = os.environ.get(
        "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
    )
    if "model_name" in params:
        kwargs["model_name"] = model_name

    return DiarizationPipeline(**kwargs)


def load_audio(path: str):
    import whisperx

    return whisperx.load_audio(path)


def translate_to_english(audio) -> dict:
    """Whisper translate pass. Returns {'segments': [...], 'language': 'xx'}.

    Each segment has 'start', 'end', 'text' (English).
    """
    model = _whisper_translate_model()
    return model.transcribe(audio, batch_size=settings.whisper_batch_size)


def diarize_spans(audio, num_speakers: int | None = None) -> list[dict]:
    """Run pyannote on the audio. Returns [{'start', 'end', 'speaker'}, ...]."""
    pipeline = _diarize_pipeline()
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    df = pipeline(audio, **kwargs)  # pandas DataFrame: start, end, speaker
    spans = [
        {"start": float(row.start), "end": float(row.end), "speaker": row.speaker}
        for row in df.itertuples()
    ]
    spans.sort(key=lambda s: s["start"])
    return spans
