"""Job persistence helpers + arq enqueue, shared by web app and worker."""
import uuid
from datetime import datetime
from typing import Optional

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import Job, JobStatus


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def get_arq_pool() -> ArqRedis:
    return await create_pool(redis_settings())


# --- CRUD ------------------------------------------------------------------
def create_job(user_id: int, original_filename: str, stored_path: str) -> Job:
    job = Job(
        id=uuid.uuid4().hex,
        user_id=user_id,
        original_filename=original_filename,
        stored_path=stored_path,
    )
    with Session(engine) as db:
        db.add(job)
        db.commit()
        db.refresh(job)
    return job


def get_job(job_id: str, user_id: Optional[int] = None) -> Optional[Job]:
    with Session(engine) as db:
        job = db.get(Job, job_id)
    if job is None:
        return None
    if user_id is not None and job.user_id != user_id:
        return None
    return job


def list_jobs(user_id: int, limit: int = 100) -> list[Job]:
    with Session(engine) as db:
        rows = db.exec(
            select(Job)
            .where(Job.user_id == user_id)
            .order_by(Job.created_at.desc())
            .limit(limit)
        ).all()
    return list(rows)


def update_job(job_id: str, **fields) -> None:
    """Partial update; always bumps updated_at."""
    with Session(engine) as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = datetime.utcnow()
        db.add(job)
        db.commit()


def set_progress(job_id: str, stage: str, progress: float) -> None:
    update_job(
        job_id,
        status=JobStatus.processing,
        stage=stage,
        progress=max(0.0, min(1.0, progress)),
    )


def mark_done(job_id: str, output_path: str, **extra) -> None:
    update_job(
        job_id,
        status=JobStatus.done,
        stage="done",
        progress=1.0,
        output_path=output_path,
        **extra,
    )


def mark_error(job_id: str, message: str) -> None:
    update_job(job_id, status=JobStatus.error, stage="error", error=message[:2000])
