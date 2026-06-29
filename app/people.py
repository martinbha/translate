"""People, voiceprints, and per-job speaker identities.

Shared by the worker (auto-matching at transcription time) and the API
(enrollment + naming corrections). The transcript is rendered from stored
segments + the *current* speaker names, so renaming a speaker updates the
document immediately — no re-transcription.
"""
import json
from pathlib import Path
from typing import Optional

from sqlmodel import Session, delete, select

from app.config import settings
from app.db import engine
from app.models import Job, JobSpeaker, Person, Voiceprint
from worker import embeddings as emb
from worker.render import render_markdown


# --- People / voiceprints --------------------------------------------------
def list_people() -> list[Person]:
    with Session(engine) as db:
        return list(db.exec(select(Person).order_by(Person.name)).all())


def create_person(name: str) -> Person:
    person = Person(name=name.strip())
    with Session(engine) as db:
        db.add(person)
        db.commit()
        db.refresh(person)
    return person


def delete_person(person_id: int) -> None:
    with Session(engine) as db:
        db.exec(delete(Voiceprint).where(Voiceprint.person_id == person_id))
        db.exec(delete(Person).where(Person.id == person_id))
        db.commit()


def add_voiceprint(person_id: int, embedding: bytes, source: str = "enroll") -> None:
    with Session(engine) as db:
        db.add(Voiceprint(person_id=person_id, embedding=embedding, source=source))
        db.commit()


def voiceprint_count(person_id: int) -> int:
    with Session(engine) as db:
        rows = db.exec(
            select(Voiceprint).where(Voiceprint.person_id == person_id)
        ).all()
    return len(rows)


def people_for_matching() -> list[dict]:
    """[{id, name, voiceprints: [np.ndarray, ...]}] for everyone with samples."""
    with Session(engine) as db:
        people = db.exec(select(Person)).all()
        out = []
        for p in people:
            vps = db.exec(
                select(Voiceprint).where(Voiceprint.person_id == p.id)
            ).all()
            if not vps:
                continue
            out.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "voiceprints": [emb.from_bytes(v.embedding) for v in vps],
                }
            )
    return out


# --- Per-job speakers ------------------------------------------------------
def save_job_speakers(job_id: str, entries: list[dict]) -> None:
    with Session(engine) as db:
        db.exec(delete(JobSpeaker).where(JobSpeaker.job_id == job_id))
        for e in entries:
            db.add(JobSpeaker(job_id=job_id, **e))
        db.commit()


def get_job_speaker(speaker_id: int) -> Optional[JobSpeaker]:
    with Session(engine) as db:
        return db.get(JobSpeaker, speaker_id)


def list_job_speakers(job_id: str) -> list[JobSpeaker]:
    with Session(engine) as db:
        return list(
            db.exec(
                select(JobSpeaker)
                .where(JobSpeaker.job_id == job_id)
                .order_by(JobSpeaker.display_name)
            ).all()
        )


def assign_speaker(
    job_speaker_id: int,
    *,
    person_id: Optional[int] = None,
    new_name: Optional[str] = None,
) -> Optional[str]:
    """Manually name a job speaker. Either link to an existing `person_id` or
    create one from `new_name`. The speaker's stored voiceprint is added to
    that person (learn-from-correction). Returns the job_id for re-render.
    """
    with Session(engine) as db:
        js = db.get(JobSpeaker, job_speaker_id)
        if js is None:
            return None

        if person_id is None and new_name:
            person = Person(name=new_name.strip())
            db.add(person)
            db.commit()
            db.refresh(person)
            person_id = person.id
        person = db.get(Person, person_id) if person_id else None
        if person is None:
            return None

        js.person_id = person.id
        js.display_name = person.name
        js.auto_matched = False
        db.add(js)

        # Learn: mint a voiceprint for this person from the cluster embedding.
        if js.embedding:
            db.add(
                Voiceprint(
                    person_id=person.id, embedding=js.embedding, source="correction"
                )
            )
        db.commit()
        job_id = js.job_id

    refresh_job_markdown(job_id)
    return job_id


# --- Segment storage + rendering ------------------------------------------
def segments_path(job_id: str) -> Path:
    return settings.output_dir / f"{job_id}.segments.json"


def write_segments(job_id: str, segments: list[dict]) -> None:
    segments_path(job_id).write_text(json.dumps(segments), encoding="utf-8")


def read_segments(job_id: str) -> list[dict]:
    p = segments_path(job_id)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def build_markdown(job_id: str) -> str:
    """Render Markdown from stored label-segments + current speaker names."""
    with Session(engine) as db:
        job = db.get(Job, job_id)
    segments = read_segments(job_id)
    label_to_name = {js.label: js.display_name for js in list_job_speakers(job_id)}

    resolved = [
        {
            "speaker": label_to_name.get(s.get("label"), "Unknown speaker"),
            "start": s.get("start"),
            "end": s.get("end"),
            "text": s.get("text", ""),
        }
        for s in segments
    ]
    return render_markdown(
        resolved,
        original_filename=job.original_filename if job else job_id,
        source_language=job.source_language if job else None,
        num_speakers=job.num_speakers if job else None,
    )


def refresh_job_markdown(job_id: str) -> None:
    """Re-write the downloadable .md to reflect current speaker names."""
    md = build_markdown(job_id)
    (settings.output_dir / f"{job_id}.md").write_text(md, encoding="utf-8")
