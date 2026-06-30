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
    # Call pyannote directly (not via whisperx's wrapper) so we can request
    # per-speaker embeddings — the voiceprints used for identification.
    import inspect
    import os

    import torch
    from pyannote.audio import Pipeline

    model_name = os.environ.get(
        "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
    )
    # token vs use_auth_token kwarg renamed across versions.
    params = inspect.signature(Pipeline.from_pretrained).parameters
    kw = {}
    token = settings.hf_token or None
    if "token" in params:
        kw["token"] = token
    elif "use_auth_token" in params:
        kw["use_auth_token"] = token

    pipeline = Pipeline.from_pretrained(model_name, **kw)
    if pipeline is None:
        raise RuntimeError(
            f"Failed to load diarization pipeline '{model_name}'. "
            "Check HF token + model access."
        )
    pipeline.to(torch.device(settings.whisper_device))
    return pipeline


def load_audio(path: str):
    import whisperx

    return whisperx.load_audio(path)


def translate_to_english(audio) -> dict:
    """Whisper translate pass. Returns {'segments': [...], 'language': 'xx'}.

    Each segment has 'start', 'end', 'text' (English).
    """
    model = _whisper_translate_model()
    return model.transcribe(audio, batch_size=settings.whisper_batch_size)


SAMPLE_RATE = 16000


@lru_cache(maxsize=1)
def _embedding_model():
    # Dedicated speaker-embedding model — decoupled from the diarization
    # pipeline's internal embeddings (whose API varies across versions).
    # ECAPA-VoxCeleb is ungated and ships with pyannote's speechbrain dep.
    import os

    import torch
    from pyannote.audio.pipelines.speaker_verification import (
        PretrainedSpeakerEmbedding,
    )

    name = os.environ.get("EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
    return PretrainedSpeakerEmbedding(name, device=torch.device(settings.whisper_device))


def _speaker_embeddings(audio, diarization) -> dict:
    """One voiceprint per speaker: embed up to ~30s of each speaker's audio."""
    import numpy as np
    import torch

    model = _embedding_model()
    by_speaker: dict = {}
    for turn, _, label in diarization.itertracks(yield_label=True):
        by_speaker.setdefault(label, []).append((turn.start, turn.end))

    out: dict = {}
    for label, segs in by_speaker.items():
        chunks, total = [], 0
        # Longest segments first — they give the cleanest voiceprint.
        for s, e in sorted(segs, key=lambda x: x[1] - x[0], reverse=True):
            a, b = int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)
            chunks.append(audio[a:b])
            total += b - a
            if total >= 30 * SAMPLE_RATE:
                break
        if not chunks:
            continue
        wav = np.concatenate(chunks).astype("float32")
        if wav.shape[0] < SAMPLE_RATE // 2:  # < 0.5s: too little to embed
            continue
        tensor = torch.from_numpy(wav).reshape(1, 1, -1)  # (batch, channel, samples)
        vec = np.asarray(model(tensor)).reshape(-1)
        out[label] = vec
    return out


def diarize(audio, num_speakers: int | None = None) -> tuple[list[dict], dict]:
    """Diarize the audio.

    Returns:
      spans: [{'start', 'end', 'speaker'}, ...]  (speaker = raw label)
      embeddings: {label: np.ndarray}            (one voiceprint per speaker)

    Embedding is best-effort: if it fails, spans still come back so the
    transcript completes (speakers just stay anonymous).
    """
    import torch

    pipeline = _diarize_pipeline()
    # pyannote wants a file path or an in-memory waveform dict. whisperx gives
    # us a float32 mono numpy array at 16 kHz.
    waveform = torch.from_numpy(audio).unsqueeze(0)
    inp = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(inp, **kwargs)

    spans = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    spans.sort(key=lambda s: s["start"])

    embeddings: dict = {}
    try:
        embeddings = _speaker_embeddings(audio, diarization)
    except Exception:  # noqa: BLE001 - identification is optional
        import traceback

        print("Speaker embedding failed; transcript will be anonymous:")
        traceback.print_exc()
    return spans, embeddings
