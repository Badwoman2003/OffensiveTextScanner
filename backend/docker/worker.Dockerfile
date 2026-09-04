FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS gpu-base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ots_core ./ots_core
COPY backend ./backend

RUN pip install --upgrade pip && pip install -e ".[backend,triton]"

CMD ["celery", "-A", "backend.worker.celery_app:app", "worker", "-Q", "gpu_inference", "-P", "solo", "--loglevel=info"]
