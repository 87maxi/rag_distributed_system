#!/bin/bash

# ==============================================================================
# Script de Instalación Resiliente para Docker Model Runner (DMR) y MCP Gateway
# Soporta: Debian/Ubuntu, RHEL/Fedora, Arch Linux, Manjaro
# Requisitos: Go, Docker, Make, Git
# ==============================================================================

set -e # Salir en caso de error

# Colores para la terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 1. Detección de Distribución y Gestión de Paquetes
detect_and_install_deps() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS="unknown"
    fi

    log "Detectando dependencias para el sistema: $OS"

    case "$OS" in
        ubuntu|debian|pop|mint)
            sudo apt-get update
            sudo apt-get install -y git make golang-go docker.io curl
            ;;
        fedora|rhel|centos)
            sudo dnf install -y git make golang docker curl
            ;;
        arch|manjaro)
            sudo pacman -Sy --needed --noconfirm git make go docker curl
            ;;
        *)
            warn "Distribución no reconocida. Asegúrate de tener instalado: git, make, go (1.21+), docker."
            ;;
    esac
}

# 2. Verificación de versiones
check_requirements() {
    if ! command -v go &> /dev/null; then
        error "Go no está instalado. Por favor instálalo (versión 1.21 o superior recomendada)."
    fi

    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    log "Versión de Go detectada: $GO_VERSION"

    if ! command -v docker &> /dev/null; then
        warn "Docker no detectado. Es necesario para ejecutar los componentes una vez instalados."
    fi
}

# 3. Preparación de Entorno
WORKSPACE="/tmp/dmr_install_$(date +%s)"
PLUGIN_DIR="$HOME/.docker/cli-plugins"

log "Creando directorio de trabajo temporal: $WORKSPACE"
mkdir -p "$WORKSPACE"
mkdir -p "$PLUGIN_DIR"

# 4. Instalación de model-runner (Docker Model CLI)
install_model_runner() {
    log "Clonando docker/model-runner..."
    cd "$WORKSPACE"
    git clone https://github.com/docker/model-runner.git
    cd model-runner/cmd/cli

    log "Compilando model-cli..."
    make build

    if [ -f "model-cli" ]; then
        log "Instalando plugin 'docker-model' en $PLUGIN_DIR"
        cp model-cli "$PLUGIN_DIR/docker-model"
        chmod +x "$PLUGIN_DIR/docker-model"
    else
        error "No se pudo generar el binario model-cli."
    fi
}

# 5. Instalación de mcp-gateway
install_mcp_gateway() {
    log "Clonando docker/mcp-gateway..."
    cd "$WORKSPACE"
    git clone https://github.com/docker/mcp-gateway.git
    cd mcp-gateway

    log "Compilando mcp-gateway (make docker-mcp)..."
    # Nota: mcp-gateway a veces requiere que el daemon de docker esté corriendo
    # si el Makefile intenta construir una imagen.
    if make docker-mcp; then
        log "MCP Gateway compilado exitosamente."

        # Intentar instalar el plugin si el Makefile generó un binario local
        # Si el Makefile solo genera imágenes Docker, aquí ya terminamos.
        if [ -f "docker-mcp" ]; then
            log "Instalando plugin 'docker-mcp' en $PLUGIN_DIR"
            cp docker-mcp "$PLUGIN_DIR/docker-mcp"
            chmod +x "$PLUGIN_DIR/docker-mcp"
        fi
    else
        warn "Fallo la ejecución de 'make docker-mcp'. Verifica que Docker esté activo."
    fi
}

# --- EJECUCIÓN ---

detect_and_install_deps
check_requirements
install_model_runner
install_mcp_gateway

log "===================================================="
log "PROCESO FINALIZADO"
log "Plugins instalados en: $PLUGIN_DIR"
log "Puedes verificar la instalación ejecutando: docker model --help"
log "===================================================="

# Limpieza opcional
# rm -rf "$WORKSPACE"
