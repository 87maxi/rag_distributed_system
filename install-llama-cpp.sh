#!/bin/bash

# ==============================================================================
# Script de Instalación Universal y Resiliente de llama.cpp
# Soporta: Debian/Ubuntu, RHEL/Fedora, Arch Linux, Alpine
# Hardware: NVIDIA GPU (CUDA), CPU-only, Notebooks (iGPU)
# ==============================================================================

set -e # Abortar en caso de error

# Colores para salida
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 1. Detección de Distribución y Gestión de Paquetes
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS="unknown"
    fi
}

install_dependencies() {
    log "Detectando sistema operativo: $OS"
    case "$OS" in
        ubuntu|debian|pop|mint)
            log "Instalando dependencias para base Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y build-essential cmake git pkg-config libcurl4-openssl-dev
            ;;
        fedora|rhel|centos)
            log "Instalando dependencias para base RHEL/Fedora..."
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y cmake git libcurl-devel
            ;;
        arch|manjaro)
            log "Instalando dependencias para Arch Linux..."
            sudo pacman -Sy --needed --noconfirm base-devel cmake git curl
            ;;
        *)
            warn "Distribución no reconocida explícitamente. Asegúrate de tener cmake, git y compiladores instalados."
            ;;
    esac
}

# 2. Búsqueda inteligente de Compiladores (GCC/Clang)
find_best_compiler() {
    log "Buscando el mejor compilador disponible..."
    for version in 14 13 12 11 ""; do
        CC_CMD="gcc-$version"
        CXX_CMD="g++-$version"
        if [ -z "$version" ]; then CC_CMD="gcc"; CXX_CMD="g++"; fi

        if command -v "$CC_CMD" &> /dev/null; then
            export CC=$(which "$CC_CMD")
            export CXX=$(which "$CXX_CMD")
            log "Usando compilador: $CC"
            return 0
        fi
    done

    # Fallback a clang si gcc no está
    if command -v clang &> /dev/null; then
        export CC=$(which clang)
        export CXX=$(which clang++)
        log "Usando compilador Clang: $CC"
    else
        error "No se encontró ningún compilador C++ compatible (GCC o Clang)."
    fi
}

# 3. Detección de Hardware (GPU vs CPU)
detect_hardware() {
    HAS_CUDA=false
    # 1. Verificar si existe nvidia-smi
    if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
        log "GPU NVIDIA detectada."
        # 2. Buscar NVCC (CUDA Toolkit)
        if command -v nvcc &> /dev/null; then
            CUDA_BIN=$(which nvcc)
            HAS_CUDA=true
        elif [ -d "/usr/local/cuda" ]; then
            CUDA_BIN="/usr/local/cuda/bin/nvcc"
            HAS_CUDA=true
        else
            # Búsqueda exhaustiva en rutas comunes de distros
            CUDA_BIN=$(find /usr/local/cuda* /opt/cuda* -name nvcc -print -quit 2>/dev/null || true)
            if [ -n "$CUDA_BIN" ]; then HAS_CUDA=true; fi
        fi
    fi

    if [ "$HAS_CUDA" = true ]; then
        log "CUDA Toolkit encontrado en: $CUDA_BIN"
    else
        warn "Instalando en modo CPU (No se detectó GPU NVIDIA o CUDA Toolkit)."
    fi
}

# --- EJECUCIÓN PRINCIPAL ---

detect_distro
install_dependencies
find_best_compiler
detect_hardware

# 4. Clonar y Preparar
BUILD_ROOT="/tmp/llama_build"
INSTALL_DIR="$HOME/bin/llama.cpp"

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

log "Clonando llama.cpp (versión más reciente)..."
git clone https://github.com/ggerganov/llama.cpp --depth 1
cd llama.cpp

# 5. Configuración de CMake
mkdir build && cd build
CMAKE_ARGS=("-DCMAKE_BUILD_TYPE=Release" "-DBUILD_SHARED_LIBS=ON")

if [ "$HAS_CUDA" = true ]; then
    log "Habilitando soporte CUDA..."
    CMAKE_ARGS+=("-DGGML_CUDA=ON")
    CMAKE_ARGS+=("-DCMAKE_CUDA_COMPILER=$CUDA_BIN")
    # Optimización para arquitecturas de GPU comunes (evita recompilaciones lentas)
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=all-major")
else
    # Soporte para AVX/AVX2/AVX512 se detecta automáticamente por llama.cpp
    log "Optimizando para ejecución en CPU..."
fi

cmake .. "${CMAKE_ARGS[@]}"

# 6. Compilación
log "Iniciando compilación con $(nproc) hilos..."
cmake --build . --config Release -j$(nproc)

# 7. Instalación y Enlaces
log "Instalando binarios en $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR/bin"
cp -r bin/* "$INSTALL_DIR/bin/"

# Manejo de librerías compartidas (.so / .dylib / .dll)
log "Configurando librerías..."
# Buscamos todas las librerías compartidas generadas en el proceso de build
SO_FILES=$(find . -name "*.so*" -o -name "*.dylib*" -o -name "*.dll*" 2>/dev/null || true)

if [ -n "$SO_FILES" ]; then
    log "Librerías encontradas: $(echo "$SO_FILES" | xargs -n1 basename | tr '\n' ' ')"
    for lib in $SO_FILES; do
        # Copiar a la carpeta de instalación local
        cp "$lib" "$INSTALL_DIR/bin/"

        # Intentar instalar en el sistema si tenemos sudo (opcional)
        if [ "$OS" != "unknown" ]; then
             sudo cp "$lib" /usr/local/lib/ 2>/dev/null || true
        fi
    done

    if [ "$OS" != "unknown" ]; then
        sudo ldconfig 2>/dev/null || warn "No se pudo ejecutar ldconfig. Es posible que necesites añadir $INSTALL_DIR/bin a tu LD_LIBRARY_PATH"
    fi
else
    warn "No se encontraron librerías compartidas. Esto es normal si la compilación fue estática."
fi

# 8. Crear Enlaces Simbólicos para compatibilidad
ln -sf "$INSTALL_DIR/bin/llama-server" "$INSTALL_DIR/bin/com.docker.llama-server"

# Crear enlace en el PATH del usuario si existe
mkdir -p "$HOME/.local/bin"
ln -sf "$INSTALL_DIR/bin/llama-server" "$HOME/.local/bin/llama-server"

log "===================================================="
log "INSTALACIÓN COMPLETADA EXITOSAMENTE"
log "Hardware: $([ "$HAS_CUDA" = true ] && echo 'GPU (CUDA)' || echo 'CPU')"
log "Sistema: $OS"
log "Binario principal: $INSTALL_DIR/bin/llama-server"
log "Tip: Asegúrate de que $HOME/.local/bin esté en tu PATH"
log "===================================================="
