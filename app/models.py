"""Database models (SQLModel)."""
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.utcnow()


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    totp_secret: str
    created_at: datetime = Field(default_factory=_now)


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)  # uuid hex
    user_id: int = Field(index=True, foreign_key="user.id")

    original_filename: str
    stored_path: str
    output_path: Optional[str] = None

    status: JobStatus = Field(default=JobStatus.queued)
    stage: str = "queued"          # human-readable current step
    progress: float = 0.0          # 0..1
    error: Optional[str] = None

    source_language: Optional[str] = None
    num_speakers: Optional[int] = None  # detected speaker count

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Person(SQLModel, table=True):
    """A known speaker the system can identify by voice."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    created_at: datetime = Field(default_factory=_now)


class Voiceprint(SQLModel, table=True):
    """One voice-embedding sample for a Person. People accumulate several:
    from explicit enrollment and from naming corrections (learn-as-you-go)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    person_id: int = Field(index=True, foreign_key="person.id")
    embedding: bytes  # float32 vector, np.tobytes()
    source: str = "enroll"  # "enroll" | "correction"
    created_at: datetime = Field(default_factory=_now)


class JobSpeaker(SQLModel, table=True):
    """A diarized speaker cluster within one job: its centroid voiceprint,
    the (auto or manual) identity, and the label shown in the transcript.
    Stored so a later naming correction can mint a new Voiceprint."""
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, foreign_key="job.id")
    label: str  # raw diarization label, e.g. "SPEAKER_00"
    embedding: Optional[bytes] = None  # None if cluster had too little speech
    person_id: Optional[int] = Field(default=None, foreign_key="person.id")
    display_name: str  # "Martin" if identified, else "Speaker 1"
    auto_matched: bool = False  # True if assigned by voice match (not manual)
    match_score: Optional[float] = None  # cosine similarity of the auto match
    created_at: datetime = Field(default_factory=_now)
