# Transcribe & Translate

Self-hosted web app: log in, drop an audio (or video) file, and get back a
nicely formatted Markdown transcript — **diarized**, **speaker-identified by
voice**, and **translated into English**.

- **Whisper** (large-v3, via WhisperX) in `translate` mode for direct X→English
- **pyannote** for speaker diarization **and** voiceprint embeddings
- **Voice identification**: enroll known people + learn from naming corrections
- **FastAPI** + a single-GPU **arq** worker, **SQLite**, single-user login with **TOTP 2FA**
- Lightweight **React** frontend, served on **port 8000** (your nginx subdomain proxies to it)

## Pipeline

```
                ┌─ pyannote diarize ──▶ speaker spans + a voiceprint per speaker
audio ──────────┤                          │
                └─ Whisper translate ─▶ English segments w/ timestamps
                                           │
              join by timestamp overlap ───┘ + match voiceprints to known people
                                           │
              render Markdown  (### Martin · 00:01:23   /   ### Speaker 2 · …)
```

Diarization and translation are **two independent passes** over the audio,
joined by timestamp. Whisper's `translate` task emits English directly (so
bilingual audio is normalized to English), while pyannote handles *who spoke
when* acoustically — and hands us a centroid **voiceprint** per speaker, which
we cosine-match against enrolled people to put real names on the transcript.

## Speaker identification

- **Enroll**: upload a clean voice sample per person → a voiceprint is stored.
- **Auto-label**: future meetings match each diarized voice to known people
  (above `SPEAKER_MATCH_THRESHOLD`); unmatched voices stay `Speaker N`.
- **Learn from corrections**: naming a speaker in a transcript saves *that*
  voiceprint to the person, so identification improves with use.

The transcript renders from stored segments + current names, so renaming a
speaker updates the document instantly — no re-transcription.

## Architecture

```
browser ─https─▶ nginx (your subdomain) ─▶ :8000 FastAPI
                                              │ enqueue
                                          Redis (arq queue)
                                              │
                                          GPU worker  (serial, 1 job at a time)
                                              │
                                   SQLite (jobs/users) + ./data/outputs/*.md
```

---

## Setup (server with the RTX 5080)

The 5080 is **Blackwell (sm_120)** — it needs a **CUDA 12.8+** PyTorch build.
Default PyPI torch wheels won't have Blackwell kernels.

### Option A — Docker (recommended; isolates the CUDA mess)

Requires the NVIDIA Container Toolkit.

```bash
cp .env.example .env        # fill in SECRET_KEY + HF_TOKEN
cd frontend && npm install && npm run build && cd ..   # build the SPA
docker compose up -d --build
docker compose exec api python -m app.create_user      # create your login
```

### Option B — bare venv

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# PyTorch from the CUDA 12.8 index FIRST:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt

cp .env.example .env        # fill in SECRET_KEY + HF_TOKEN
python -m app.create_user   # create your login (prints a QR for your authenticator)

# build the frontend
cd frontend && npm install && npm run build && cd ..

# run (use a process manager / systemd in production):
redis-server &                                   # or your existing redis
arq worker.settings.WorkerSettings &             # the GPU worker
uvicorn app.main:app --host 0.0.0.0 --port 8000  # the API + frontend
```

### Generate a SECRET_KEY

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### HuggingFace token (required for diarization)

1. Create a token at <https://hf.co/settings/tokens>.
2. Accept the model terms for
   [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0).
3. Put it in `.env` as `HF_TOKEN`.

---

## nginx

Point your subdomain at the app. nginx terminates TLS; keep `COOKIE_SECURE=true`.

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Large uploads + long SSE progress streams:
    client_max_body_size 2048m;
    proxy_read_timeout 3600s;
    proxy_buffering off;            # so SSE progress isn't buffered
}
```

---

## Local development (no GPU)

The web layer runs anywhere; only the worker needs the GPU. On your laptop you
can run the API + frontend to work on the UI:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# in another shell:
cd frontend && npm install && npm run dev    # Vite dev server, proxies /api → :8000
```

Test the GPU pipeline directly on the server without the web stack:

```bash
python -m worker.pipeline path/to/audio.m4a out.md
```

---

## Notes & knobs

- **VRAM**: `large-v3` (translate) + diarization fit comfortably in 16 GB.
  Lower `WHISPER_COMPUTE_TYPE` to `int8_float16` if you ever run tight.
- **One job at a time**: the worker is `max_jobs=1` — the GPU does one anyway.
- **Files are kept**: uploads live in `data/uploads/`, outputs in `data/outputs/`
  (`{job}.md` + `{job}.segments.json`). Nothing is auto-deleted.
- **Identification threshold**: `SPEAKER_MATCH_THRESHOLD` (default 0.5). Raise it
  if guests get mislabeled as known people; lower it if known people are missed.
- **Translation is always → English** (Whisper `translate`). Other targets would
  need a separate MT/LLM step.
- **Security**: single user, argon2 password, TOTP 2FA, login lockout after 5
  failed attempts / 15 min, signed httpOnly session cookie, upload size + type
  validation. Note: voiceprints are biometric data, kept only in your SQLite DB.
