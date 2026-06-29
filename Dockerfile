# syntax=docker/dockerfile:1.7
# GPU image for the worker (and reused for the API). Built for Blackwell / 5080.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# apt + pip both use BuildKit cache mounts: even if these layers re-run, the
# downloaded .deb/.whl files are reused instead of re-fetched over the network.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 ffmpeg git

WORKDIR /srv

# PyTorch first, from the CUDA 12.8 index (has Blackwell kernels). ~2GB — the
# pip cache mount means a rebuild of this layer won't re-download it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Dependencies are copied + installed BEFORE app code, so editing Python source
# never invalidates this (expensive) layer.
COPY requirements.txt requirements-gpu.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -r requirements-gpu.txt

COPY app ./app
COPY worker ./worker
COPY frontend/dist ./frontend/dist

EXPOSE 8000

# Default command runs the API; the worker service overrides it (see compose).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
