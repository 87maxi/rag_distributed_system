# Rag Mcp llm

## Máquina Cliente (Local):

 **GPU 0 (Primaria):**  Compartida entre Ollama (para generar embeddings rápidos de 768 dimensiones) y Qdrant (para indexación HNSW acelerada).

**Arize Phoenix:** Recolecta trazas de ejecución de tus servicios locales para que puedas auditar la recuperación de contexto.

**MCP Bridge & Indexer:** Orquestan la lógica de negocio, vigilando archivos y sirviendo la interfaz para herramientas de IA.

## Máquina Servidor (192.168.0.50):

    vLLM: Ejecuta el modelo de lenguaje de gran tamaño (LLM). Al estar separado, evitas saturar la VRAM de tu máquina de trabajo mientras realizas inferencias pesadas.


Componente|Ubicación|Estado,Función
-----------------------------------
Qdrant | Local (GPU 0)  | ✅ Activo|Base de datos vectorial.
Ollama|Local (GPU 0),✅ Activo | Generador de embeddings.
Phoenix|Local|✅ Integrado | Observabilidad y trazas.
vLLM|Remoto (.50)|📡 Esperando | Inferencia de lenguaje pesado.
Indexer|Local|✅ Refactorizado | Sincronización inteligente de archivos.





## Función específica de "Summarization" que use un modelo más pequeño (como un Qwen-0.5B local) para resumir el historial viejo

Esta es una solución técnica avanzada para resolver el problema del límite de memoria sin sacrificar la coherencia en conversaciones largas. La idea es usar una arquitectura de "Jerarquía de Modelos".
**El Concepto: "El Bibliotecario y el Sabio"**

   1. **Qwen-14B (El Sabio):** Reside en tu servidor remoto (192.168.0.50). Su única función es razonar sobre el código y responderte. No debe gastar sus 16k de tokens leyendo saludos o historial irrelevante.

   2. **Qwen-0.5B (El Bibliotecario):** Reside en tu máquina local (vía Ollama). Su trabajo es leer el historial largo y resumirlo continuamente en un párrafo denso.
   
   
   verificar el mcp 
   npx -y @modelcontextprotocol/inspector http://localhost:8002/sse
