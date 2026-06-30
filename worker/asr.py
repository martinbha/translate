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
    # whisperx's wrapper around pyannote. This whisperx build (paired with
    # pyannote.audio 4.x) returns (DataFrame, {speaker: embedding}) when called
    # with return_embeddings=True — so we get voiceprints for free.
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError:  # pragma: no cover - version dependent
        from whisperx import DiarizationPipeline  # type: ignore

    import inspect
    import os

    params = inspect.signature(DiarizationPipeline.__init__).parameters
    kwargs = {"device": settings.whisper_device}
    token = settings.hf_token or None
    if "use_auth_token" in params:
        kwargs["use_auth_token"] = token
    elif "token" in params:
        kwargs["token"] = token

    # Must be the pyannote-4.x-native model. The legacy 3.1 pipeline returns a
    # generator under pyannote 4.0 and breaks whisperx's `.speaker_diarization`.
    model_name = os.environ.get(
        "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
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


def diarize(audio, num_speakers: int | None = None) -> tuple[list[dict], dict]:
    """Diarize the audio.

    Returns:
      spans: [{'start', 'end', 'speaker'}, ...]  (speaker = raw label)
      embeddings: {label: np.ndarray}            (one voiceprint per speaker)
    """
    import numpy as np

    pipeline = _diarize_pipeline()
    kwargs = {"return_embeddings": True}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    result = pipeline(audio, **kwargs)
    # return_embeddings=True yields (df, {speaker: [floats]}); be defensive in
    # case a build returns just the df.
    if isinstance(result, tuple):
        df, spk_emb = result
    else:
        df, spk_emb = result, None

    spans = [
        {"start": float(r.start), "end": float(r.end), "speaker": r.speaker}
        for r in df.itertuples()
    ]
    spans.sort(key=lambda s: s["start"])

    embeddings: dict = {}
    if spk_emb:
        for label, vec in spk_emb.items():
            arr = np.asarray(vec, dtype="float32")
            if arr.size and np.all(np.isfinite(arr)):
                embeddings[label] = arr
    return spans, embeddings
