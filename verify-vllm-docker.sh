#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Verificación de Integridad vLLM + Docker              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Verificar Docker
log_info "Verificando instalación de Docker..."
if ! command -v docker &> /dev/null; then
    log_error "Docker no está instalado"
    exit 1
fi
log_success "Docker instalado: $(docker --version)"

# 2. Verificar Docker Compose
log_info "Verificando Docker Compose..."
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose no está instalado"
    exit 1
fi
log_success "Docker Compose disponible"

# 3. Verificar NVIDIA Container Toolkit
log_info "Verificando NVIDIA Container Toolkit..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log_success "NVIDIA Container Toolkit funcionando correctamente"
else
    log_error "NVIDIA Container Toolkit no está configurado correctamente"
    log_info "Instala con: sudo apt-get install -y nvidia-container-toolkit"
    exit 1
fi

# 4. Verificar estructura de directorios
log_info "Verificando estructura de directorios..."
mkdir -p "$PROJECT_DIR/vllm"
mkdir -p "$PROJECT_DIR/data/vllm-models"
mkdir -p "$PROJECT_DIR/data/vllm-cache"
log_success "Directorios creados"

# 5. Crear Dockerfile si no existe
if [ ! -f "$PROJECT_DIR/vllm/Dockerfile" ]; then
    log_info "Creando Dockerfile para vLLM..."
    cat > "$PROJECT_DIR/vllm/Dockerfile" <<'EOF'
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
EOF
    log_success "Dockerfile creado"
fi

# 6. Crear script de verificación Python
log_info "Creando script de verificación..."
cat > "$PROJECT_DIR/vllm/verify.py" <<'EOF'
#!/usr/bin/env python3
"""Script de verificación para vLLM en Docker"""

import sys
import torch

def main():
    print("🔍 Verificando instalación de vLLM...")
    
    # Verificar vLLM
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
    except ImportError as e:
        print(f"❌ Error importando vLLM: {e}")
        sys.exit(1)
    
    # Verificar PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA no disponible - ejecutando en CPU")
    
    # Verificar otras dependencias
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("⚠️  Transformers no instalado")
    
    try:
        import diffusers
        print(f"✅ Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("⚠️  Diffusers no instalado")
    
    print("\n✅ Verificación completada exitosamente")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
chmod +x "$PROJECT_DIR/vllm/verify.py"
log_success "Script de verificación creado"

# 7. Construir imagen Docker
log_info "Construyendo imagen Docker de vLLM..."
cd "$PROJECT_DIR"
if docker build -t vllm-gpu:latest -f vllm/Dockerfile vllm/; then
    log_success "Imagen Docker construida exitosamente"
else
    log_error "Error al construir imagen Docker"
    exit 1
fi

# 8. Probar contenedor
log_info "Probando contenedor vLLM..."
if docker run --rm --gpus all vllm-gpu:latest python3 /app/verify.py; then
    log_success "Contenedor funciona correctamente"
else
    log_error "Error al ejecutar contenedor"
    exit 1
fi

# 9. Verificar docker-compose.yml
log_info "Verificando configuración de docker-compose.yml..."
if [ -f "$PROJECT_DIR/docker-compose.yml" ]; then
    if grep -q "vllm" "$PROJECT_DIR/docker-compose.yml"; then
        log_success "Servicio vLLM ya existe en docker-compose.yml"
    else
        log_warning "Servicio vLLM no encontrado en docker-compose.yml"
        log_info "Agrega el siguiente servicio a tu docker-compose.yml:"
        cat <<'COMPOSE'

  vllm-server:
    image: vllm-gpu:latest
    container_name: vllm_server
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    environment:
      - MODEL_NAME=${VLLM_MODEL:-facebook/opt-125m}
      - HF_TOKEN=${HF_TOKEN}
      - GPU_MEMORY_UTILIZATION=0.8
    ports:
      - "8000:8000"
    volumes:
      - ./data/vllm-models:/models
      - ./data/vllm-cache:/models/cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    networks:
      - rag_network
COMPOSE
    fi
else
    log_warning "docker-compose.yml no encontrado"
fi

# 10. Resumen final
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Verificación Completada                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
log_success "Imagen Docker: vllm-gpu:latest"
log_success "Directorios: $PROJECT_DIR/data/vllm-models"
echo ""
log_info "Para iniciar el servicio:"
echo "   docker-compose up vllm-server"
echo ""
log_info "Para probar la API:"
echo "   curl http://localhost:8000/v1/models"
echo ""
log_info "Para ejecutar un modelo específico:"
echo "   docker run --rm --gpus all -p 8000:8000 vllm-gpu:latest \\"
echo "     python3 -m vllm.entrypoints.openai.api_server \\"
echo "     --model facebook/opt-125m --host 0.0.0.0 --port 8000"
echo ""
