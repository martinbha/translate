"""End-to-end: audio file -> diarized, English-translated Markdown.

Importable as `run_pipeline(...)` (used by the arq worker) and runnable as a
CLI for local testing:  python -m worker.pipeline path/to/audio.m4a
"""
from pathlib import Path
from typing import Callable, Optional

from app.config import settings
from worker import asr, translate
from worker.lang import to_nllb
from worker.render import render_markdown

# A progress callback gets (stage_label, fraction_0_to_1).
ProgressCb = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    print(f"[{frac*100:5.1f}%] {stage}")


def _normalize_speaker(label: str | None) -> str:
    # whisperx emits "SPEAKER_00"; prettify to "Speaker 1".
    if not label:
        return "Unknown speaker"
    if label.upper().startswith("SPEAKER_"):
        try:
            return f"Speaker {int(label.split('_')[-1]) + 1}"
        except ValueError:
            return label
    return label


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

    progress("Transcribing", 0.10)
    result = asr.transcribe(audio)
    language = result.get("language")

    progress("Aligning words", 0.45)
    aligned = asr.align(result["segments"], language, audio)

    progress("Identifying speakers", 0.60)
    diarized = asr.diarize(audio, aligned, num_speakers=num_speakers)
    segments = diarized.get("segments", [])

    # Collapse word/segment data into clean {speaker, start, end, text} dicts.
    clean: list[dict] = []
    speakers: set[str] = set()
    for seg in segments:
        spk = _normalize_speaker(seg.get("speaker"))
        speakers.add(spk)
        clean.append(
            {
                "speaker": spk,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": (seg.get("text") or "").strip(),
            }
        )

    # Translate to English unless the source already is English.
    src_nllb = to_nllb(language)
    if language == "en" or src_nllb is None:
        if language != "en":
            progress("Source language not translatable; keeping original", 0.80)
    else:
        progress("Translating to English", 0.80)
        texts = [c["text"] for c in clean]

        def tcb(frac: float) -> None:
            progress("Translating to English", 0.80 + 0.15 * frac)

        translated = translate.translate_segments(texts, src_nllb, progress_cb=tcb)
        for c, t in zip(clean, translated):
            c["text"] = t

    progress("Rendering document", 0.97)
    num_detected = len(speakers) if speakers else None
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
