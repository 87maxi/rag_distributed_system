#!/bin/bash

# ==============================================================================
# Script de Instalación Resiliente para vLLM
# Soporta: Debian/Ubuntu, RHEL/Fedora, Arch Linux, Manjaro
# Requisitos: NVIDIA GPU (Recomendado) o CPU (Modo limitado)
# ==============================================================================

set -e  # Salir en caso de error
set -u  # Salir si hay variables no definidas

# Colores para la terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuración
VLLM_DIR="${HOME}/vllm-local"
PYTHON_VERSION="3.10"

# Funciones de Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 1. Detección de Distribución e Instalación de Dependencias
install_system_deps() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS="unknown"
    fi

    log_info "Detectando sistema operativo: $OS"

    case "$OS" in
        ubuntu|debian|pop|mint)
            sudo apt-get update
            sudo apt-get install -y python3-pip python3-venv git curl libgl1
            ;;
        fedora|rhel|centos)
            sudo dnf install -y python3-pip git curl mesa-libGL
            ;;
        arch|manjaro)
            sudo pacman -Sy --needed --noconfirm python-pip git curl mesa
            ;;
        *)
            log_warning "Distribución no reconocida. Asegúrate de tener Python 3.10+, pip y git."
            ;;
    esac
}

# 2. Verificación de GPU y CUDA
check_gpu_env() {
    HAS_GPU=false
    if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
        log_success "GPU NVIDIA detectada."
        HAS_GPU=true

        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
            log_info "CUDA Toolkit $CUDA_VERSION detectado."
        else
            log_warning "GPU detectada pero 'nvcc' no encontrado. Se usará el driver de sistema."
        fi
    else
        log_warning "No se detectó GPU NVIDIA. Instalando vLLM en modo CPU (limitado)."
    fi
}

# 3. Gestión de UV (Instalador rápido de Python)
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_info "Instalando 'uv' para una gestión de paquetes ultra-rápida..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Intentar cargar uv al path actual
        export PATH="$HOME/.cargo/bin:$PATH"

        if [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi

        if ! command -v uv &> /dev/null; then
            log_warning "No se pudo activar 'uv' automáticamente en esta sesión. Usando pip estándar."
            USE_UV=false
        else
            USE_UV=true
        fi
    else
        USE_UV=true
        log_success "'uv' ya está disponible."
    fi
}

# 4. Instalación de vLLM
install_vllm() {
    log_info "Preparando directorio de instalación: $VLLM_DIR"
    rm -rf "$VLLM_DIR"
    mkdir -p "$VLLM_DIR"
    cd "$VLLM_DIR"

    # Determinar índice de PyTorch basado en GPU
    if [ "$HAS_GPU" = true ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi

    if [ "$USE_UV" = true ]; then
        log_info "Instalando con uv..."
        uv venv --python "$PYTHON_VERSION" .venv
        source .venv/bin/activate
        uv pip install torch --index-url "$TORCH_INDEX"
        uv pip install vllm fastapi uvicorn httpx transformers accelerate
    else
        log_info "Instalando con venv + pip..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install torch --index-url "$TORCH_INDEX"
        pip install vllm fastapi uvicorn httpx transformers accelerate
    fi
}

# 5. Script de Verificación
create_verify_script() {
    cat > verify_vllm.py <<'EOF'
import torch
import sys
try:
    import vllm
    print(f"✅ vLLM version: {vllm.__version__}")
except ImportError:
    print("❌ vLLM no instalado")
    sys.exit(1)

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
EOF
}

# --- EJECUCIÓN ---
main() {
    log_info "Iniciando proceso de instalación de vLLM..."
    install_system_deps
    check_gpu_env
    check_uv
    install_vllm
    create_verify_script

    # Verificación final
    source .venv/bin/activate
    python verify_vllm.py || log_warning "La verificación automática falló, pero la instalación terminó."

    log_success "===================================================="
    log_success "INSTALACIÓN COMPLETADA"
    log_success "Entorno: $VLLM_DIR"
    log_success "Comando para activar: source $VLLM_DIR/.venv/bin/activate"
    log_success "Hardware detectado: $([ "$HAS_GPU" = true ] && echo 'NVIDIA GPU' || echo 'CPU Only')"
    log_success "===================================================="
}

main
