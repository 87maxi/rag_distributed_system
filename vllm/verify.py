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
