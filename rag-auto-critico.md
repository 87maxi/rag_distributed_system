# Plan de Acción de Refactorización Técnica

Este es el Plan de Acción de Refactorización Técnica diseñado para ser presentado ante un Arquitecto de Software. El plan transforma la estructura actual de un flujo lineal RAG a un ecosistema de IA Agente Distribuido y Autocrítico, optimizando el uso de hardware heterogéneo (RTX 5060 remota vs. GTX 1650 local).

## Plan de Acción: Refactorización hacia Arquitectura MCP Agéntica y Distribuida

1. Visión General de la Refactorización

El objetivo es desacoplar la Inferencia de Alto Nivel (Servidor Remoto) de la Gestión de Contexto y Validación (Máquina Cliente). Se busca implementar un bucle de "Autocrítica" donde modelos locales (SLMs) preparen, limpien y validen los datos antes y después de interactuar con el modelo de lenguaje de gran tamaño (LLM).

2. Fase 1: Infraestructura y Orquestación (Docker & Network)

## Prioridad: Crítica

### Unificación de Redes: Consolidar todos los servicios en el docker-compose.yml bajo una red bridge común para permitir resolución de nombres (ej: http://rag_ollama:11434).

### Aislamiento de Recursos GPU: * Configurar el servicio rag_qdrant para que utilice el driver de NVIDIA con persistencia de vectores en RAM para acelerar la búsqueda HNSW.

### Asignar el servicio ollama exclusivamente a la GPU local (GTX 1650) para tareas de embedding y modelos de soporte (Qwen 0.5B/1.5B).

### Healthcheck Interdependiente: Configurar el rag-mcp-server para que no inicie hasta que el servicio ollama retorne un estado saludable (modelos cargados mediante entrypoint.sh).

## 3. Fase 2: Rediseño del "Cerebro Local" (MCP Server)

### Prioridad: Alta Se propone transformar el main.py de un simple puente a un Orquestador de Tareas.

    Implementación de la Jerarquía de Modelos:

        El Bibliotecario (Qwen 0.5B): Refactorizar get_compressed_memory para que no solo resuma, sino que identifique entidades clave (clases, funciones, archivos) mencionadas en el historial.

**El Crítico Local (Qwen 1.5B):** Crear un nuevo middleware dentro de handle_call_tool que evalúe si el contexto recuperado de Qdrant es suficiente para responder la query.

**Nuevas Herramientas (Tools) en el Protocolo:**

   **get_project_structure:** Escáner de sistema de archivos para entregar al LLM remoto un mapa del árbol de directorios antes de la búsqueda.

   **verify_syntax:** Herramienta de post-procesamiento que use el modelo local para validar que el código generado por el LLM remoto no contenga errores de sintaxis obvios.

## 4. Fase 3: Optimización del Pipeline RAG (Indexer & Search)

### Prioridad: Media

    **Enriquecimiento de Metadatos:** Modificar indexer.py para que incluya en el payload de Qdrant no solo el content y path, sino también los imports y una "firma de función" para mejorar la relevancia en la búsqueda.

**Búsqueda Semántica Mejorada:** Ajustar get_query_vector para realizar "Query Expansion" (usar el modelo local para generar 3 variaciones de la pregunta del usuario antes de buscar en Qdrant).

   **Implementar un sistema de "Reranking" básico en el MCP:** recuperar 10 resultados de Qdrant y usar el modelo de 1.5B para seleccionar los 3 más pertinentes.

## 5. Fase 4: Observabilidad y Trazabilidad (Phoenix Integration)

### Prioridad: Media

**Instrumentación OpenTelemetry:** Integrar trazas en main.py para que cada llamada a search_code envíe los spans correspondientes al servicio rag_phoenix.

**Monitoreo de Calidad:** Utilizar la interfaz de Phoenix para visualizar el flujo: Consulta -> Embedding -> Top K de Qdrant -> Respuesta Remota, facilitando el debug de alucinaciones.

## 6. Resumen de Flujo de Datos Refactorizado

   **Zed IDE (Cliente):** Envía prompt al MCP.

   **MCP (Local):**

   **El Bibliotecario resume historial.**

   **El Crítico analiza la necesidad de búsqueda.**

   **El Indexer provee el mapa de archivos.**

   **Ollama genera embeddings.**

   **Qdrant (Local): Entrega fragmentos de código acelerados por GPU.**

   **vLLM (Remoto .50): Recibe un "Super-Prompt" estructurado y genera la lógica compleja.**

   **MCP (Local): El Crítico valida sintaxis y retorna a Zed.**



