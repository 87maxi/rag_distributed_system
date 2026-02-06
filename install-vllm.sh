#!/bin/bash

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VLLM_DIR="${HOME}/vllm-local"
PYTHON_VERSION="3.10"

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_gpu() {
    log_info "Verificando disponibilidad de GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        log_success "GPU NVIDIA detectada"
        return 0
    else
        log_warning "No se detectó GPU NVIDIA. vLLM funcionará en modo CPU (muy lento)"
        return 1
    fi
}

check_cuda() {
    log_info "Verificando instalación de CUDA..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log_success "CUDA $CUDA_VERSION detectado"
        return 0
    else
        log_warning "CUDA no detectado. Instalando versión compatible..."
        return 1
    fi
}

check_uv() {
    log_info "Verificando instalación de uv..."
    
    if ! command -v uv &> /dev/null; then
        log_info "Instalando uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        log_success "uv instalado correctamente"
    else
        log_success "uv ya está instalado"
    fi
}

install_vllm() {
    log_info "Limpiando instalación anterior..."
    rm -rf "$VLLM_DIR"
    
    log_info "Creando nuevo entorno virtual en $VLLM_DIR..."
    uv init "$VLLM_DIR"
    cd "$VLLM_DIR"
    uv venv --python "$PYTHON_VERSION"
    
    log_info "Activando entorno virtual..."
    source .venv/bin/activate
    
    # Detectar versión de CUDA para PyTorch
    if check_gpu; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
        
        if [[ "$CUDA_VERSION" == "12."* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            log_info "Usando PyTorch para CUDA 12.1"
        elif [[ "$CUDA_VERSION" == "11."* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            log_info "Usando PyTorch para CUDA 11.8"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            log_warning "Versión CUDA no reconocida, usando CUDA 12.1 por defecto"
        fi
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        log_warning "Instalando PyTorch CPU-only"
    fi
    
    log_info "Instalando PyTorch con soporte GPU..."
    uv pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
    
    log_info "Instalando vLLM..."
    uv pip install vllm
    
    log_info "Instalando dependencias adicionales..."
    uv pip install diffusers accelerate transformers
    uv pip install fastapi uvicorn httpx
    
    log_success "Instalación completada"
}

verify_installation() {
    log_info "Verificando instalación..."
    
    cd "$VLLM_DIR"
    source .venv/bin/activate
    
    # Verificar importación de vLLM
    python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
        log_error "Error al importar vLLM"
        return 1
    }
    
    # Verificar PyTorch y CUDA
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || {
        log_error "Error al verificar PyTorch"
        return 1
    }
    
    log_success "Verificación completada exitosamente"
}

create_test_server() {
    log_info "Creando servidor de prueba..."
    
    cat > "$VLLM_DIR/test_server.py" <<'EOF'
#!/usr/bin/env python3
"""
Servidor de prueba para vLLM
Uso: python test_server.py --model <model_name>
"""
import argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m", 
                       help="Modelo a cargar (por defecto: facebook/opt-125m)")
    args = parser.parse_args()
    
    print(f"🚀 Cargando modelo: {args.model}")
    
    # Crear instancia de LLM
    llm = LLM(model=args.model, trust_remote_code=True)
    
    # Parámetros de muestreo
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
    
    # Prompt de prueba
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    
    print("\n📝 Generando respuestas...")
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    print("\n✅ Prueba completada exitosamente")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$VLLM_DIR/test_server.py"
    log_success "Servidor de prueba creado en $VLLM_DIR/test_server.py"
}

print_usage() {
    cat <<EOF

${GREEN}╔════════════════════════════════════════════════════════════╗
║           Instalación de vLLM Completada                   ║
╚════════════════════════════════════════════════════════════╝${NC}

${BLUE}📍 Ubicación:${NC} $VLLM_DIR

${BLUE}🚀 Para usar vLLM:${NC}
   cd $VLLM_DIR
   source .venv/bin/activate

${BLUE}🧪 Probar instalación:${NC}
   python test_server.py --model facebook/opt-125m

${BLUE}🐳 Para integración con Docker:${NC}
   Ejecuta: ./verify-vllm-docker.sh

${BLUE}📚 Modelos recomendados:${NC}
   - LLMs: meta-llama/Llama-2-7b-chat-hf
   - Pequeños: facebook/opt-125m (para pruebas)
   - Multimodal: llava-hf/llava-1.5-7b-hf

${YELLOW}⚠️  Nota:${NC} Para generación de imágenes, considera usar
   Stable Diffusion en lugar de vLLM.

EOF
}

# Main execution
main() {
    log_info "Iniciando instalación de vLLM..."
    
    check_uv
    check_gpu
    check_cuda
    install_vllm
    verify_installation
    create_test_server
    print_usage
    
    log_success "¡Instalación completada exitosamente!"
}

main "$@"
