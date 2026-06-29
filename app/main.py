"""FastAPI app: auth, upload, job status (SSE), download. Serves on port 8000."""
import asyncio
import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

from app.auth import (
    get_current_user,
    login_locked,
    record_login_failure,
    reset_login_failures,
)
from app.config import settings
from app.db import engine, init_db
from app.jobs import create_job, get_arq_pool, get_job, list_jobs
from app.models import Job, JobStatus, User
from app.security import (
    make_session_token,
    verify_password,
    verify_totp,
)

app = FastAPI(title="Transcribe & Translate")


@app.on_event("startup")
async def _startup() -> None:
    init_db()
    app.state.arq = await get_arq_pool()


# --- Schemas ---------------------------------------------------------------
class LoginIn(BaseModel):
    username: str
    password: str
    totp: str


class JobOut(BaseModel):
    id: str
    original_filename: str
    status: JobStatus
    stage: str
    progress: float
    error: str | None
    source_language: str | None
    num_speakers: int | None
    created_at: str

    @classmethod
    def from_job(cls, j: Job) -> "JobOut":
        return cls(
            id=j.id,
            original_filename=j.original_filename,
            status=j.status,
            stage=j.stage,
            progress=j.progress,
            error=j.error,
            source_language=j.source_language,
            num_speakers=j.num_speakers,
            created_at=j.created_at.isoformat(),
        )


# --- Auth ------------------------------------------------------------------
@app.post("/api/login")
def login(body: LoginIn, request: Request, response: Response):
    ip = request.client.host if request.client else "unknown"
    if login_locked(ip):
        raise HTTPException(429, "Too many attempts. Try again later.")

    with Session(engine) as db:
        user = db.exec(select(User).where(User.username == body.username)).first()

    ok = (
        user is not None
        and verify_password(body.password, user.password_hash)
        and verify_totp(user.totp_secret, body.totp)
    )
    if not ok:
        record_login_failure(ip)
        raise HTTPException(401, "Invalid credentials")

    reset_login_failures(ip)
    token = make_session_token(user.id)
    response.set_cookie(
        settings.cookie_name,
        token,
        max_age=settings.session_max_age,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
    return {"username": user.username}


@app.post("/api/logout")
def logout(response: Response, user: User = Depends(get_current_user)):
    response.delete_cookie(settings.cookie_name)
    return {"ok": True}


@app.get("/api/me")
def me(user: User = Depends(get_current_user)):
    return {"username": user.username}


# --- Jobs ------------------------------------------------------------------
def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {suffix or 'none'}")
    return suffix


@app.post("/api/jobs", response_model=JobOut)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    suffix = _safe_suffix(file.filename or "")

    job_id_path = settings.upload_dir
    job_id_path.mkdir(parents=True, exist_ok=True)

    # Stream to disk with a hard size cap.
    import uuid

    stored = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with open(stored, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                fh.close()
                os.unlink(stored)
                raise HTTPException(413, "File too large")
            fh.write(chunk)

    job = create_job(
        user_id=user.id,
        original_filename=file.filename or stored.name,
        stored_path=str(stored),
    )
    await app.state.arq.enqueue_job("transcribe_job", job.id)
    return JobOut.from_job(job)


@app.get("/api/jobs", response_model=list[JobOut])
def jobs(user: User = Depends(get_current_user)):
    return [JobOut.from_job(j) for j in list_jobs(user.id)]


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def job_detail(job_id: str, user: User = Depends(get_current_user)):
    job = get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobOut.from_job(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, user: User = Depends(get_current_user)):
    # Confirm ownership before opening the stream.
    if get_job(job_id, user_id=user.id) is None:
        raise HTTPException(404, "Job not found")

    async def stream():
        last = None
        while True:
            job = get_job(job_id, user_id=user.id)
            if job is None:
                break
            payload = JobOut.from_job(job).model_dump()
            payload["created_at"] = str(payload["created_at"])
            blob = json.dumps(payload, default=str)
            if blob != last:
                yield f"data: {blob}\n\n"
                last = blob
            if job.status in (JobStatus.done, JobStatus.error):
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/markdown")
def job_markdown(job_id: str, user: User = Depends(get_current_user)):
    job = get_job(job_id, user_id=user.id)
    if job is None or not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "Not ready")
    return {"markdown": Path(job.output_path).read_text(encoding="utf-8")}


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str, user: User = Depends(get_current_user)):
    job = get_job(job_id, user_id=user.id)
    if job is None or not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(404, "Not ready")
    stem = Path(job.original_filename).stem
    return FileResponse(
        job.output_path,
        media_type="text/markdown",
        filename=f"{stem}.md",
    )


# --- Static frontend (built React) -----------------------------------------
# Mounted last so /api/* takes precedence. `html=True` serves index.html for
# client-side routes.
_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
