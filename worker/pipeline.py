"""End-to-end: audio file -> diarized, English-translated Markdown.

Importable as `run_pipeline(...)` (used by the arq worker) and runnable as a
CLI for local testing:  python -m worker.pipeline path/to/audio.m4a
"""
from pathlib import Path
from typing import Callable, Optional

from app.config import settings
from worker import asr
from worker.render import render_markdown

# A progress callback gets (stage_label, fraction_0_to_1).
ProgressCb = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    print(f"[{frac*100:5.1f}%] {stage}")


def _normalize_speaker(label: str | None) -> str:
    # pyannote emits "SPEAKER_00"; prettify to "Speaker 1".
    if not label:
        return "Unknown speaker"
    if label.upper().startswith("SPEAKER_"):
        try:
            return f"Speaker {int(label.split('_')[-1]) + 1}"
        except ValueError:
            return label
    return label


def _speaker_for(start, end, spans: list[dict]) -> str:
    """Pick the diarization speaker whose span overlaps [start, end] most.

    Falls back to the nearest span by midpoint when nothing overlaps (e.g. a
    Whisper segment landing in a gap between diarized turns).
    """
    if start is None or end is None or not spans:
        return "Unknown speaker"
    best, best_overlap = None, 0.0
    for sp in spans:
        overlap = min(end, sp["end"]) - max(start, sp["start"])
        if overlap > best_overlap:
            best_overlap, best = overlap, sp["speaker"]
    if best is None:
        mid = (start + end) / 2
        best = min(spans, key=lambda s: abs((s["start"] + s["end"]) / 2 - mid))["speaker"]
    return _normalize_speaker(best)


def run_pipeline(
    audio_path: str,
    output_path: str,
    original_filename: str,
    num_speakers: Optional[int] = None,
    progress: ProgressCb = _noop,
) -> dict:
    """Run the full pipeline and write Markdown to `output_path`.

    Returns metadata: {source_language, num_speakers, output_path}.
    """
    progress("Loading audio", 0.02)
    audio = asr.load_audio(audio_path)

    # Diarization is acoustic and independent of the text — run it first.
    progress("Identifying speakers", 0.10)
    spans = asr.diarize_spans(audio, num_speakers=num_speakers)

    # Whisper translate pass: source speech -> English segments w/ timestamps.
    progress("Transcribing & translating", 0.45)
    result = asr.translate_to_english(audio)
    language = result.get("language")
    segments = result.get("segments", [])

    # Join the two passes: assign each English segment the speaker whose
    # diarized span overlaps it most in time.
    progress("Assigning speakers", 0.90)
    clean: list[dict] = []
    speakers: set[str] = set()
    for seg in segments:
        start, end = seg.get("start"), seg.get("end")
        spk = _speaker_for(start, end, spans)
        speakers.add(spk)
        clean.append(
            {
                "speaker": spk,
                "start": start,
                "end": end,
                "text": (seg.get("text") or "").strip(),
            }
        )

    progress("Rendering document", 0.97)
    # Speaker count comes from diarization, not from how many distinct labels
    # happened to win a segment.
    num_detected = len({s["speaker"] for s in spans}) if spans else None
    markdown = render_markdown(
        clean,
        original_filename=original_filename,
        source_language=language,
        num_speakers=num_detected,
    )
    Path(output_path).write_text(markdown, encoding="utf-8")

    progress("Done", 1.0)
    return {
        "source_language": language,
        "num_speakers": num_detected,
        "output_path": output_path,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m worker.pipeline <audio_file> [out.md]")
        raise SystemExit(1)

    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else str(settings.output_dir / "out.md")
    meta = run_pipeline(src, out, original_filename=Path(src).name)
    print(f"\nWrote {out}\n{meta}")
