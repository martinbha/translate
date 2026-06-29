# GPU image for the worker (and reused for the API). Built for Blackwell / 5080.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# PyTorch first, from the CUDA 12.8 index (has Blackwell kernels).
RUN pip3 install --upgrade pip && \
    pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt requirements-gpu.txt ./
RUN pip3 install -r requirements.txt && \
    pip3 install -r requirements-gpu.txt

COPY app ./app
COPY worker ./worker
COPY frontend/dist ./frontend/dist

EXPOSE 8000

# Default command runs the API; the worker service overrides it (see compose).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
