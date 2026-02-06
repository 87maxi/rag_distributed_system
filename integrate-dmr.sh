#!/bin/bash

# integrate-dmr.sh - Integra vLLM local con Docker Model Runner
# Realiza una instalación limpia en /opt para evitar problemas de permisos
# Requiere sudo

set -e

# Configuración
OPT_INSTALL_DIR="/opt/vllm-local"
DMR_VLLM_DIR="/opt/vllm-env"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Verificar sudo
if [ "$EUID" -ne 0 ]; then
  log_error "Este script debe ejecutarse con sudo para instalar en /opt"
  echo "Ejecuta: sudo $0"
  exit 1
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Integración vLLM y Docker Model Runner (System-Wide)     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "⚠️  Detectado problema de permisos en directorio home."
log_info "🔄 Cambiando estrategia: Instalación global en ${OPT_INSTALL_DIR}..."

# Instalar uv si no existe para root
if ! command -v uv &> /dev/null; then
    log_info "Instalando uv para root..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
    
    # Verificar de nuevo
    if ! command -v uv &> /dev/null; then
        log_error "No se pudo instalar uv. Por favor instálalo manualmente."
        exit 1
    fi
fi

# Limpiar instalación anterior en /opt si existe
if [ -d "$OPT_INSTALL_DIR" ]; then
    log_info "Limpiando instalación anterior en ${OPT_INSTALL_DIR}..."
    rm -rf "$OPT_INSTALL_DIR"
fi

# Crear entorno en /opt
log_info "Inicializando entorno en ${OPT_INSTALL_DIR}..."
mkdir -p "$OPT_INSTALL_DIR"
uv init "$OPT_INSTALL_DIR"
cd "$OPT_INSTALL_DIR"
uv venv --python 3.10

# Instalar dependencias
log_info "Instalando vLLM y dependencias (esto puede tardar unos minutos)..."
source .venv/bin/activate

# Instalar PyTorch con CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar paquetes
uv pip install vllm diffusers accelerate transformers

# Crear directorios DMR
log_info "Configurando estructura para Docker Model Runner..."
mkdir -p "${DMR_VLLM_DIR}/bin"

# Crear enlaces simbólicos apuntando a /opt
rm -f "${DMR_VLLM_DIR}/bin/vllm"
ln -s "${OPT_INSTALL_DIR}/.venv/bin/vllm" "${DMR_VLLM_DIR}/bin/vllm"
log_success "Enlace creado: ${DMR_VLLM_DIR}/bin/vllm -> ${OPT_INSTALL_DIR}/.venv/bin/vllm"

rm -f "${DMR_VLLM_DIR}/bin/python3"
ln -s "${OPT_INSTALL_DIR}/.venv/bin/python3" "${DMR_VLLM_DIR}/bin/python3"
log_success "Enlace creado: ${DMR_VLLM_DIR}/bin/python3 -> ${OPT_INSTALL_DIR}/.venv/bin/python3"

# Generar versión
log_info "Generando archivo de versión..."
VERSION=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
echo "${VERSION:-unknown}" > "${DMR_VLLM_DIR}/version"
log_success "Versión detectada: ${VERSION:-unknown}"

# Configurar backends adicionales
# Diffusers
log_info "Configurando diffusers..."
DIFFUSERS_DIR="/opt/diffusers-env"
mkdir -p "${DIFFUSERS_DIR}/bin"
rm -f "${DIFFUSERS_DIR}/bin/python3"
ln -s "${OPT_INSTALL_DIR}/.venv/bin/python3" "${DIFFUSERS_DIR}/bin/python3"
echo "0.26.0" > "${DIFFUSERS_DIR}/version"
log_success "Entorno diffusers configurado"

# SGLang
log_info "Configurando sglang..."
SGLANG_DIR="/opt/sglang-env"
mkdir -p "${SGLANG_DIR}/bin"
rm -f "${SGLANG_DIR}/bin/python3"
ln -s "${OPT_INSTALL_DIR}/.venv/bin/python3" "${SGLANG_DIR}/bin/python3"
log_success "Entorno sglang configurado"

log_info "Ajustando permisos de lectura global..."
chmod -R a+rx "${OPT_INSTALL_DIR}"
chmod -R a+rx "${DMR_VLLM_DIR}"
chmod -R a+rx "${DIFFUSERS_DIR}"
chmod -R a+rx "${SGLANG_DIR}"

log_success "Integración completada exitosamente"
