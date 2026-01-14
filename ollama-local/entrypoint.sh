#!/bin/sh
set -e

# Función para verificar si Ollama está listo
check_ollama() {
    curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1
    return $?
}

# Iniciar servidor
echo "🚀 Iniciando Ollama server..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Esperar con timeout
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if check_ollama; then
        echo "✅ Servidor Ollama listo en puerto 11434"
        
        # Hacer pull del modelo
        echo "📥 Descargando modelo..."
        if ollama pull nomic-embed-text:latest; then
            echo "✅ Modelo descargado exitosamente"
            
            # Listar modelos disponibles
            echo "📋 Modelos disponibles:"
            ollama list
            
            # Mantener contenedor activo
            echo "🔄 Servidor Ollama en ejecución..."
            wait $OLLAMA_PID
            exit 0
        else
            echo "❌ Error al descargar el modelo"
            exit 1
        fi
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "⏳ Esperando servidor Ollama... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

echo "❌ Timeout: Ollama server no se inició en 60 segundos"
exit 1