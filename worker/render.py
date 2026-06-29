"""Render the diarized + translated segments into a Markdown document."""
from datetime import datetime, timedelta


def _ts(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"
    return str(timedelta(seconds=int(seconds)))


def render_markdown(
    segments: list[dict],
    *,
    original_filename: str,
    source_language: str | None,
    num_speakers: int | None,
) -> str:
    """`segments` items: {speaker, start, end, text}.

    Consecutive segments from the same speaker are merged into one turn.
    """
    lines: list[str] = []
    lines.append(f"# Transcript — {original_filename}")
    lines.append("")
    meta = [
        f"- **Source language:** {source_language or 'unknown'}",
        "- **Translated to:** English",
        f"- **Speakers detected:** {num_speakers if num_speakers else 'n/a'}",
        f"- **Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    lines.extend(meta)
    lines.append("")
    lines.append("---")
    lines.append("")

    cur_speaker = object()  # sentinel that never equals a real speaker
    buf: list[str] = []
    turn_start: float | None = None

    def flush():
        if buf:
            label = cur_speaker if isinstance(cur_speaker, str) else "Speaker"
            lines.append(f"### {label} · {_ts(turn_start)}")
            lines.append(" ".join(s.strip() for s in buf if s.strip()))
            lines.append("")

    for seg in segments:
        speaker = seg.get("speaker") or "Unknown speaker"
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if speaker != cur_speaker:
            flush()
            cur_speaker = speaker
            buf = []
            turn_start = seg.get("start")
        buf.append(text)
    flush()

    return "\n".join(lines).rstrip() + "\n"
