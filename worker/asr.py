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
    # Call pyannote.audio 4.x directly (NOT via whisperx's wrapper, which on
    # this install passes min/max_speakers=None and trips a generator return
    # path -> 'generator' has no attribute 'speaker_diarization').
    import inspect
    import os

    import torch
    from pyannote.audio import Pipeline

    model_name = os.environ.get(
        "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
    )
    params = inspect.signature(Pipeline.from_pretrained).parameters
    kw = {}
    token = settings.hf_token or None
    if "token" in params:
        kw["token"] = token
    elif "use_auth_token" in params:
        kw["use_auth_token"] = token

    pipe = Pipeline.from_pretrained(model_name, **kw)
    if pipe is None:
        raise RuntimeError(
            f"Could not load diarization pipeline '{model_name}'. "
            "Check HF token + that you've accepted the model's terms."
        )
    pipe.to(torch.device(settings.whisper_device))
    return pipe


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
    """Diarize the audio via the official pyannote 4.x API.

    Returns:
      spans: [{'start', 'end', 'speaker'}, ...]  (speaker = raw label)
      embeddings: {label: np.ndarray}            (one voiceprint per speaker)
    """
    import types

    import numpy as np
    import torch

    pipe = _diarize_pipeline()
    inp = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": 16000}

    # Call the documented way: minimal kwargs. Only pass num_speakers if known.
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    output = pipe(inp, **kwargs)

    # Some 4.0.x builds return a generator; the final yielded item is the result.
    if isinstance(output, types.GeneratorType):
        last = None
        for last in output:
            pass
        output = last

    # pyannote 4.x: output.speaker_diarization (an Annotation). Be tolerant of
    # an older build that returns the Annotation directly.
    diar = getattr(output, "speaker_diarization", output)

    spans: list[dict] = []
    if hasattr(diar, "itertracks"):
        for turn, _, speaker in diar.itertracks(yield_label=True):
            spans.append(
                {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
            )
    else:  # 4.0 docs iterate (turn, speaker) pairs directly
        for turn, speaker in diar:
            spans.append(
                {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
            )
    spans.sort(key=lambda s: s["start"])

    # Voiceprints, if this build exposes them (best-effort).
    embeddings: dict = {}
    emb = getattr(output, "speaker_embeddings", None)
    if emb is not None and hasattr(diar, "labels"):
        for i, label in enumerate(sorted(diar.labels())):
            try:
                arr = np.asarray(emb[i], dtype="float32")
                if arr.size and np.all(np.isfinite(arr)):
                    embeddings[label] = arr
            except (IndexError, TypeError, ValueError):
                pass
    return spans, embeddings
