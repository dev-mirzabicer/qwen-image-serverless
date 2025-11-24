# Qwen-Image serverless worker (A100 80GB recommended)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf-cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf-cache \
    DIFFUSERS_CACHE=/runpod-volume/hf-cache \
    SAFETENSORS_FAST=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/src

WORKDIR /app

# System deps for PIL, image codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/

CMD ["python", "-u", "/app/src/handler.py"]
