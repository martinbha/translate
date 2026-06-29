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
