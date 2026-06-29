"""arq worker definition. Run with:  arq worker.settings.WorkerSettings"""
from pathlib import Path

from app.config import settings
from app.db import init_db
from app.jobs import (
    get_job,
    mark_done,
    mark_error,
    redis_settings,
    set_progress,
    update_job,
)
from app.models import JobStatus
from worker.pipeline import run_pipeline


async def startup(ctx) -> None:
    init_db()
    # Warm the GPU models so the first real job isn't paying load time.
    try:
        from worker import asr

        asr._whisper_model()
        print("Whisper model warm.")
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort
        print(f"Model warmup skipped: {exc}")


async def transcribe_job(ctx, job_id: str) -> None:
    """The single GPU task. arq runs these serially (max_jobs=1)."""
    job = get_job(job_id)
    if job is None:
        return

    def progress(stage: str, frac: float) -> None:
        set_progress(job_id, stage, frac)

    update_job(job_id, status=JobStatus.processing, stage="starting", progress=0.0)
    out_path = str(settings.output_dir / f"{job_id}.md")
    try:
        meta = run_pipeline(
            audio_path=job.stored_path,
            output_path=out_path,
            original_filename=job.original_filename,
            progress=progress,
        )
        mark_done(
            job_id,
            output_path=meta["output_path"],
            source_language=meta.get("source_language"),
            num_speakers=meta.get("num_speakers"),
        )
    except Exception as exc:  # noqa: BLE001 - surface failure to the user
        import traceback

        traceback.print_exc()
        mark_error(job_id, f"{type(exc).__name__}: {exc}")


class WorkerSettings:
    functions = [transcribe_job]
    on_startup = startup
    redis_settings = redis_settings()
    # One GPU -> one job at a time. No timeout cap: long files are expected.
    max_jobs = 1
    job_timeout = 60 * 60 * 6  # 6h safety ceiling
    keep_result = 0
