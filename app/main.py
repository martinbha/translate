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
from app import people
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


async def _stream_to_disk(file: UploadFile, prefix: str = "") -> Path:
    """Stream an upload to disk with a hard size cap. Returns the stored path."""
    import uuid

    suffix = _safe_suffix(file.filename or "")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stored = settings.upload_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
    size = 0
    with open(stored, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                fh.close()
                os.unlink(stored)
                raise HTTPException(413, "File too large")
            fh.write(chunk)
    return stored


@app.post("/api/jobs", response_model=JobOut)
async def upload(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    stored = await _stream_to_disk(file)
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


# --- People (known speakers) -----------------------------------------------
class NameIn(BaseModel):
    name: str


class AssignIn(BaseModel):
    person_id: int | None = None
    new_name: str | None = None


def _person_out(p) -> dict:
    return {"id": p.id, "name": p.name, "voiceprints": people.voiceprint_count(p.id)}


@app.get("/api/people")
def get_people(user: User = Depends(get_current_user)):
    return [_person_out(p) for p in people.list_people()]


@app.post("/api/people")
def post_person(body: NameIn, user: User = Depends(get_current_user)):
    if not body.name.strip():
        raise HTTPException(400, "Name required")
    return _person_out(people.create_person(body.name))


@app.delete("/api/people/{person_id}")
def remove_person(person_id: int, user: User = Depends(get_current_user)):
    people.delete_person(person_id)
    return {"ok": True}


@app.post("/api/people/{person_id}/enroll")
async def enroll_voice(
    person_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Upload a clean voice sample; the worker extracts a voiceprint (GPU)."""
    stored = await _stream_to_disk(file, prefix="enroll_")
    await app.state.arq.enqueue_job("enroll_job", person_id, str(stored))
    return {"ok": True}


# --- Per-job speakers (identify / rename) ----------------------------------
def _speaker_out(js) -> dict:
    return {
        "id": js.id,
        "label": js.label,
        "display_name": js.display_name,
        "person_id": js.person_id,
        "auto_matched": js.auto_matched,
        "match_score": js.match_score,
    }


@app.get("/api/jobs/{job_id}/speakers")
def job_speakers(job_id: str, user: User = Depends(get_current_user)):
    if get_job(job_id, user_id=user.id) is None:
        raise HTTPException(404, "Job not found")
    return [_speaker_out(js) for js in people.list_job_speakers(job_id)]


@app.post("/api/jobs/{job_id}/speakers/{speaker_id}/assign")
def assign_job_speaker(
    job_id: str,
    speaker_id: int,
    body: AssignIn,
    user: User = Depends(get_current_user),
):
    if get_job(job_id, user_id=user.id) is None:
        raise HTTPException(404, "Job not found")
    js = people.get_job_speaker(speaker_id)
    if js is None or js.job_id != job_id:
        raise HTTPException(404, "Speaker not found")
    if body.person_id is None and not (body.new_name or "").strip():
        raise HTTPException(400, "person_id or new_name required")
    if people.assign_speaker(
        speaker_id, person_id=body.person_id, new_name=body.new_name
    ) is None:
        raise HTTPException(400, "Could not assign")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/speakers/{speaker_id}/sample")
def speaker_sample(
    job_id: str, speaker_id: int, user: User = Depends(get_current_user)
):
    """A short representative audio clip of this speaker, for the naming UI."""
    job = get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(404, "Job not found")
    js = people.get_job_speaker(speaker_id)
    if js is None or js.job_id != job_id:
        raise HTTPException(404, "Speaker not found")

    segs = [
        s for s in people.read_segments(job_id)
        if s.get("label") == js.label and s.get("start") is not None
    ]
    if not segs:
        raise HTTPException(404, "No audio for this speaker")
    longest = max(segs, key=lambda s: (s.get("end") or 0) - (s.get("start") or 0))
    start = float(longest["start"])
    dur = min(8.0, max(1.0, float(longest["end"]) - start))

    import subprocess

    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-ss", str(start), "-t", str(dur),
            "-i", job.stored_path, "-ac", "1", "-ar", "16000",
            "-f", "mp3", "-loglevel", "error", "pipe:1",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise HTTPException(500, "Could not extract audio sample")
    return Response(content=proc.stdout, media_type="audio/mpeg")


# --- Static frontend (built React) -----------------------------------------
# Mounted last so /api/* takes precedence. `html=True` serves index.html for
# client-side routes.
_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
