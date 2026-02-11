#!/bin/bash
# ==============================================================================
# Script de Instalación Avanzado y Resiliente para vLLM
#
# Características:
# - Detección multi-distro: Debian/Ubuntu, RHEL/Fedora, Arch, macOS.
# - Detección de hardware: NVIDIA (con mapeo de driver a CUDA), AMD (experimental), CPU.
# - Selección dinámica de PyTorch: Instala la versión de PyTorch (cu121, cu118, rocm, cpu)
#   adecuada para el hardware detectado.
# - Flexibilidad de Python: Busca una versión compatible (3.8+) en lugar de una fija.
# - Instalador rápido 'uv': Lo usa si está disponible para acelerar la creación
#   del entorno y la instalación de paquetes.
# - Seguridad: Pide confirmación antes de instalar paquetes a nivel de sistema
#   y maneja permisos con 'sudo' de forma inteligente.
# ==============================================================================

set -e
set -u

# --- Configuración y Colores ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VLLM_DIR="${HOME}/vllm-local"
SUDO_CMD=""
PKG_MANAGER=""
PYTHON_EXEC=""
TORCH_INDEX=""
HARDWARE_TYPE="CPU"
USE_UV=false

# --- Funciones de Logging ---
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 1. Detección de SO y gestor de paquetes
detect_os_and_pkg_manager() {
    log_info "Detectando sistema operativo y gestor de paquetes..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case "$ID" in
                ubuntu|debian|pop|mint) PKG_MANAGER="apt-get";;
                fedora|rhel|centos) PKG_MANAGER="dnf";;
                arch|manjaro) PKG_MANAGER="pacman";;
                *) log_warning "Distribución Linux no reconocida. Se intentará continuar.";;
            esac
        else
            log_warning "No se pudo detectar la distribución de Linux."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            PKG_MANAGER="brew"
        else
            log_error "Homebrew no está instalado. Por favor, instálalo desde https://brew.sh/"
        fi
    else
        log_error "Sistema operativo no soportado: $OSTYPE"
    fi
    log_success "Sistema detectado: ${OSTYPE}, Gestor de paquetes: ${PKG_MANAGER:-N/A}"

    # Determinar si se necesita sudo
    if [ "$(id -u)" -ne 0 ]; then
        if ! command -v sudo &> /dev/null; then
            log_error "'sudo' no encontrado. Por favor, instálalo o ejecuta el script como root."
        fi
        SUDO_CMD="sudo"
    fi
}

# 2. Instalación de dependencias del sistema
install_system_deps() {
    if [ -z "$PKG_MANAGER" ]; then
        log_warning "No se pudo determinar el gestor de paquetes. Saltando instalación de dependencias."
        log_warning "Asegúrate de tener Python 3.8+, pip, git y curl instalados manualmente."
        return
    fi

    log_info "Se pueden necesitar las siguientes dependencias: python3, pip, git, curl."
    read -p "¿Deseas que el script intente instalarlas usando '$PKG_MANAGER'? (s/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        log_warning "Instalación de dependencias omitida por el usuario."
        return
    fi

    log_info "Actualizando repositorios e instalando dependencias..."
    case "$PKG_MANAGER" in
        apt-get)
            $SUDO_CMD apt-get update
            $SUDO_CMD apt-get install -y python3 python3-pip python3-venv git curl libgl1
            ;;
        dnf)
            $SUDO_CMD dnf install -y python3 python3-pip git curl mesa-libGL
            ;;
        pacman)
            $SUDO_CMD pacman -Sy --needed --noconfirm python python-pip git curl mesa
            ;;
        brew)
            brew install python git
            ;;
    esac
    log_success "Dependencias del sistema instaladas."
}

# 3. Encontrar un ejecutable de Python compatible
find_python_executable() {
    log_info "Buscando un intérprete de Python compatible (3.8+)..."
    for cmd in python3.11 python3.10 python3.9 python3.8 python3; do
        if command -v "$cmd" &> /dev/null; then
            # Usar python para verificar su propia versión
            VERSION=$("$cmd" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            # Comparación de versiones sin 'bc' para mayor portabilidad
            MAJOR=$(echo "$VERSION" | cut -d. -f1)
            MINOR=$(echo "$VERSION" | cut -d. -f2)
            if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ]; then
                PYTHON_EXEC=$(command -v "$cmd")
                log_success "Python compatible encontrado: $PYTHON_EXEC (Versión $VERSION)"
                return
            fi
        fi
    done
    log_error "No se encontró una versión de Python compatible (3.8+). Por favor, instala Python 3.8 o superior."
}

# 4. Detección de Hardware y selección de índice de PyTorch
detect_hardware_and_torch_index() {
    log_info "Detectando hardware y configurando el índice de PyTorch..."

    # --- Detección de NVIDIA GPU ---
    if command -v nvidia-smi &> /dev/null; then
        log_success "GPU NVIDIA detectada."
        HARDWARE_TYPE="NVIDIA"
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)

        log_info "Versión del driver NVIDIA: $DRIVER_VERSION"

        if [ "$DRIVER_MAJOR" -ge 525 ]; then
            log_info "Driver compatible con CUDA 12.1+. Seleccionando PyTorch para 'cu121'."
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        elif [ "$DRIVER_MAJOR" -ge 470 ]; then
            log_info "Driver compatible con CUDA 11.8. Seleccionando PyTorch para 'cu118'."
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        else
            log_warning "La versión del driver NVIDIA ($DRIVER_VERSION) es muy antigua."
            log_warning "Se intentará usar la versión para CPU, pero podría ser inestable."
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            HARDWARE_TYPE="CPU (Driver NVIDIA antiguo)"
        fi
        return
    fi

    # --- Detección de AMD GPU (Experimental) ---
    if command -v rocminfo &> /dev/null || [ -d /dev/kfd ]; then
        log_success "GPU AMD (ROCm) detectada (Soporte experimental)."
        log_warning "La instalación para AMD puede requerir pasos adicionales y no está garantizada."
        HARDWARE_TYPE="AMD"
        TORCH_INDEX="https://download.pytorch.org/whl/rocm5.7" # Puede necesitar ajuste según la versión de ROCm
        return
    fi

    # --- Fallback a CPU ---
    log_warning "No se detectó GPU NVIDIA o AMD compatible. Instalando vLLM en modo CPU."
    HARDWARE_TYPE="CPU"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
}

# 5. Gestión de UV (Instalador rápido de Python)
check_and_install_uv() {
    if command -v uv &> /dev/null; then
        USE_UV=true
        log_success "'uv' ya está disponible."
        return
    fi

    log_info "El instalador rápido 'uv' no está presente."
    read -p "¿Deseas descargarlo e instalarlo para acelerar el proceso? (s/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        log_warning "'uv' no será instalado. Se usará 'pip' estándar (más lento)."
        USE_UV=false
        return
    fi

    log_info "Instalando 'uv'..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        export PATH="$HOME/.cargo/bin:$PATH"
        if [ -f "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi

        if command -v uv &> /dev/null; then
            log_success "'uv' instalado y activado para esta sesión."
            USE_UV=true
        else
            log_warning "No se pudo activar 'uv' automáticamente. Usando 'pip' estándar."
            USE_UV=false
        fi
    else
        log_warning "La instalación de 'uv' falló. Usando 'pip' estándar."
        USE_UV=false
    fi
}

# 6. Creación de entorno e instalación de vLLM
install_vllm() {
    log_info "Preparando directorio de instalación en: $VLLM_DIR"
    rm -rf "$VLLM_DIR"
    mkdir -p "$VLLM_DIR"
    cd "$VLLM_DIR"

    PACKAGES="vllm fastapi uvicorn httpx transformers accelerate"

    if [ "$USE_UV" = true ]; then
        log_info "Creando entorno virtual e instalando con 'uv'..."
        uv venv --python "$PYTHON_EXEC" .venv
        source .venv/bin/activate
        uv pip install --no-cache-dir torch --index-url "$TORCH_INDEX"
        uv pip install --no-cache-dir $PACKAGES
    else
        log_info "Creando entorno virtual e instalando con 'venv' y 'pip'..."
        "$PYTHON_EXEC" -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install --no-cache-dir torch --index-url "$TORCH_INDEX"
        pip install --no-cache-dir $PACKAGES
    fi

    if ! python -c "import vllm" &> /dev/null; then
       log_error "La instalación de vLLM parece haber fallado. Revisa los logs."
    fi
    log_success "vLLM y sus dependencias han sido instalados en el entorno virtual."
}

# 7. Script de Verificación
create_verify_script() {
    cat > verify_vllm.py <<'EOF'
import torch
import sys
import os

print("--- Verificación de vLLM ---")
try:
    import vllm
    print(f"✅ vLLM version: {vllm.__version__}")
except ImportError:
    print("❌ ERROR: vLLM no está instalado o no se puede importar.")
    sys.exit(1)

print(f"✅ PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"✅ CUDA disponible para PyTorch: {cuda_available}")

if cuda_available:
    try:
        device_count = torch.cuda.device_count()
        print(f"   - GPUs encontradas: {device_count}")
        for i in range(device_count):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"❌ ERROR al obtener información de la GPU: {e}")
elif "rocm" in torch.__version__:
    print("   - Plataforma: ROCm (AMD)")
else:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        print("   - Forzado a modo CPU a través de CUDA_VISIBLE_DEVICES.")
    else:
        print("   - Plataforma: CPU")

print("----------------------------")
EOF
log_success "Script de verificación 'verify_vllm.py' creado."
}


# --- Flujo Principal de Ejecución ---
main() {
    log_info "Iniciando proceso de instalación de vLLM..."

    detect_os_and_pkg_manager
    install_system_deps
    find_python_executable
    detect_hardware_and_torch_index
    check_and_install_uv
    install_vllm
    create_verify_script

    # Verificación final
    log_info "Ejecutando script de verificación final..."
    # 'cd' por si el script se ejecuta con 'bash install.sh' desde otro dir
    cd "$VLLM_DIR"
    source .venv/bin/activate
    python verify_vllm.py || log_warning "La verificación automática falló, pero la instalación terminó."

    # Resumen
    echo -e "${GREEN}====================================================${NC}"
    echo -e "${GREEN}      INSTALACIÓN DE VLLM COMPLETADA                ${NC}"
    echo -e "${GREEN}====================================================${NC}"
    echo -e " Entorno Virtual Creado en: ${YELLOW}$VLLM_DIR${NC}"
    echo -e " Para activarlo, ejecuta:   ${YELLOW}source $VLLM_DIR/.venv/bin/activate${NC}"
    echo -e " Hardware Detectado:          ${YELLOW}$HARDWARE_TYPE${NC}"
    echo -e " Índice de PyTorch Usado:   ${YELLOW}$TORCH_INDEX${NC}"
    echo -e " Python Usado:              ${YELLOW}$PYTHON_EXEC${NC}"
    echo -e "${GREEN}====================================================${NC}"
}

main
