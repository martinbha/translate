"""End-to-end: audio file -> diarized, identified, English-translated Markdown.

Importable as `run_pipeline(...)` (used by the arq worker) and runnable as a
CLI for local testing:  python -m worker.pipeline path/to/audio.m4a
"""
from pathlib import Path
from typing import Callable, Optional

from app.config import settings
from worker import asr
from worker import embeddings as emb
from worker.render import render_markdown

# A progress callback gets (stage_label, fraction_0_to_1).
ProgressCb = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    print(f"[{frac*100:5.1f}%] {stage}")


def _label_for(start, end, spans: list[dict]) -> Optional[str]:
    """Raw diarization label whose span overlaps [start, end] most."""
    if start is None or end is None or not spans:
        return None
    best, best_overlap = None, 0.0
    for sp in spans:
        overlap = min(end, sp["end"]) - max(start, sp["start"])
        if overlap > best_overlap:
            best_overlap, best = overlap, sp["speaker"]
    if best is None:
        mid = (start + end) / 2
        best = min(spans, key=lambda s: abs((s["start"] + s["end"]) / 2 - mid))["speaker"]
    return best


def _identify(embeddings: dict, all_labels: list[str]) -> dict:
    """Match each diarized label to a known person (or leave anonymous).

    Returns {label: {display_name, person_id, auto_matched, match_score,
    embedding(bytes|None)}}.
    """
    from app.people import people_for_matching

    people = people_for_matching()
    info: dict = {}
    unmatched = 0
    for label in all_labels:
        vec = embeddings.get(label)
        emb_bytes = emb.to_bytes(vec) if (vec is not None and emb.is_valid(vec)) else None
        person_id = name = None
        score = None
        if emb_bytes is not None and people:
            person_id, name, score = emb.best_match(
                vec, people, settings.speaker_match_threshold
            )
        if person_id is not None:
            display = name
        else:
            unmatched += 1
            display = f"Speaker {unmatched}"
        info[label] = {
            "display_name": display,
            "person_id": person_id,
            "auto_matched": person_id is not None,
            "match_score": score,
            "embedding": emb_bytes,
        }
    return info


def extract_voiceprint(audio_path: str) -> Optional[bytes]:
    """Embed a single clean enrollment sample, in the SAME space pyannote uses
    for diarization (force one speaker, take that centroid). Returns float32
    bytes, or None if no usable voice was found.
    """
    audio = asr.load_audio(audio_path)
    _, embeddings = asr.diarize(audio, num_speakers=1)
    for vec in embeddings.values():
        if emb.is_valid(vec):
            return emb.to_bytes(vec)
    return None


def run_pipeline(
    audio_path: str,
    output_path: str,
    original_filename: str,
    job_id: Optional[str] = None,
    num_speakers: Optional[int] = None,
    progress: ProgressCb = _noop,
) -> dict:
    """Run the full pipeline. When `job_id` is given, persist segments +
    speaker identities so names can be corrected later; otherwise (CLI) just
    render straight to `output_path`.
    """
    progress("Loading audio", 0.02)
    audio = asr.load_audio(audio_path)

    # Diarization (acoustic) yields speaker spans + a voiceprint per speaker.
    progress("Identifying speakers", 0.10)
    spans, speaker_embeddings = asr.diarize(audio, num_speakers=num_speakers)
    all_labels = sorted({s["speaker"] for s in spans})

    # Whisper translate pass: source speech -> English segments w/ timestamps.
    progress("Transcribing & translating", 0.45)
    result = asr.translate_to_english(audio)
    language = result.get("language")
    segments = result.get("segments", [])

    # Identify speakers by voice against the known-people registry.
    progress("Matching voices", 0.88)
    label_info = _identify(speaker_embeddings, all_labels)
    label_to_name = {lbl: info["display_name"] for lbl, info in label_info.items()}

    # Assign each English segment to a raw speaker label (kept for re-render).
    progress("Assigning speakers", 0.92)
    label_segments: list[dict] = []
    for seg in segments:
        start, end = seg.get("start"), seg.get("end")
        label_segments.append(
            {
                "label": _label_for(start, end, spans),
                "start": start,
                "end": end,
                "text": (seg.get("text") or "").strip(),
            }
        )

    num_detected = len(all_labels) if all_labels else None

    progress("Rendering document", 0.97)
    if job_id:
        from app.jobs import update_job
        from app.people import (
            refresh_job_markdown,
            save_job_speakers,
            write_segments,
        )

        write_segments(job_id, label_segments)
        save_job_speakers(
            job_id,
            [
                {
                    "label": lbl,
                    "embedding": info["embedding"],
                    "person_id": info["person_id"],
                    "display_name": info["display_name"],
                    "auto_matched": info["auto_matched"],
                    "match_score": info["match_score"],
                }
                for lbl, info in label_info.items()
            ],
        )
        # Set header fields before rendering so the .md reflects them.
        update_job(job_id, source_language=language, num_speakers=num_detected)
        refresh_job_markdown(job_id)  # writes output_dir/{job_id}.md
    else:
        resolved = [
            {
                "speaker": label_to_name.get(s["label"], "Unknown speaker"),
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
            }
            for s in label_segments
        ]
        md = render_markdown(
            resolved,
            original_filename=original_filename,
            source_language=language,
            num_speakers=num_detected,
        )
        Path(output_path).write_text(md, encoding="utf-8")

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
