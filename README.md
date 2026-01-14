# Rag Mcp llm

## Máquina Cliente (Local):

 **GPU 0 (Primaria):**  Compartida entre Ollama (para generar embeddings rápidos de 768 dimensiones) y Qdrant (para indexación HNSW acelerada).

**Arize Phoenix:** Recolecta trazas de ejecución de tus servicios locales para que puedas auditar la recuperación de contexto.

**MCP Bridge & Indexer:** Orquestan la lógica de negocio, vigilando archivos y sirviendo la interfaz para herramientas de IA.

## Máquina Servidor (192.168.0.50):

    vLLM: Ejecuta el modelo de lenguaje de gran tamaño (LLM). Al estar separado, evitas saturar la VRAM de tu máquina de trabajo mientras realizas inferencias pesadas.