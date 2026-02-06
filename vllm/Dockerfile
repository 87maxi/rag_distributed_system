FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/cache

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv para gestión rápida de paquetes
RUN pip3 install --no-cache-dir uv

WORKDIR /app

# Instalar PyTorch con CUDA
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar vLLM y dependencias
RUN uv pip install --system vllm
RUN uv pip install --system diffusers accelerate transformers
RUN uv pip install --system fastapi uvicorn httpx

# Crear directorios
RUN mkdir -p /models /app/outputs

# Script de verificación
COPY verify.py /app/verify.py

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--trust-remote-code"]
