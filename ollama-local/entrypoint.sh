#!/bin/sh
set -e

# Configuración de modelos
MODELS="nomic-embed-text:latest qwen2.5:0.5b gpt-oss:20b"

# Función para verificar si Ollama está listo
check_ollama() {
    curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1
    return $?
}

# Función para verificar si un modelo ya existe
model_exists() {
    ollama list | grep -q "$1"
    return $?
}

echo "🚀 Iniciando Ollama server en segundo plano..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Esperar a que el servidor responda
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if check_ollama; then
        echo "✅ Servidor Ollama detectado."

        # Procesar cada modelo
        for MODEL in $MODELS; do
            if model_exists "$MODEL"; then
                echo "    📦 El modelo '$MODEL' ya existe localmente. Omitiendo descarga."
            else
                echo "    📥 Descargando modelo '$MODEL'..."
                if ollama pull "$MODEL"; then
                    echo "    ✅ '$MODEL' descargado exitosamente."
                else
                    echo "    ❌ Error al descargar '$MODEL'."
                    exit 1
                fi
            fi
        done

        echo "📋 Resumen de modelos disponibles:"
        ollama list

        # Mantener el proceso principal activo
        echo "🔄 Ollama está listo y operando..."
        wait $OLLAMA_PID
        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "⏳ Esperando a Ollama... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

echo "❌ Timeout: El servidor Ollama no inició a tiempo."
exit 1
