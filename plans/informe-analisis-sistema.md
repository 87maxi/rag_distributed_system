# Informe de Análisis Completo - Sistema RAG Distribuido

**Fecha:** 2026-05-03
**Proyecto:** `rag_distributed_system`
**Arquitectura:** "El Bibliotecario y El Sabio" - RAG para Asistencia de Codificación

---

## 1. VISIÓN GENERAL DEL SISTEMA

### 1.1 Arquitectura Definida

El sistema es un pipeline RAG (Retrieval Augmented Generation) distribuido diseñado para asistir en tareas de codificación. La arquitectura se basa en dos roles conceptuales:

- **El Bibliotecario:** Responsable de resumir el historial de conversación e identificar entidades clave (clases, funciones, archivos). Reside localmente.
- **El Sabio:** LLM principal que genera respuestas. Puede ser intercambiable (Ollama, DMR, vLLM, OpenAI).

### 1.2 Componentes del Sistema

| Componente | Puerto | GPU | Función | Archivo Principal |
|------------|--------|-----|---------|-------------------|
| Qdrant | 6333, 6334 | GPU 1 | Base de datos vectorial | - |
| Ollama | 11434 | GPU 1 | Embeddings / LLM local | `ollama-local/` |
| Phoenix | 6006, 4317, 4318 | CPU | Observabilidad / Traces | - |
| Tabby | 8181 | GPU 1 | Code completion | - |
| Embedding Server | 8000 | GPU 0 | Embeddings dedicados | `embedding_server/app.py` |
| Bibliotecario Server | 8001 | GPU 0 | Resumen de historial | `bibliotecario_server/app.py` |
| Critic Server | 8002/8003 | GPU 0 | Reranking de resultados | `critic_server/app.py` |
| MCP Bridge (SSE) | 8002 | CPU | Orquestador principal | `mcp-bridge/main.py` |
| MCP Bridge (stdio) | N/A | CPU | Modo CLI/Zed | `mcp-bridge/stdio_server.py` |
| Indexer | N/A | CPU | Watchdog de archivos | `indexer/indexer.py` |

### 1.3 Topologías Soportadas

El proyecto mantiene 3 configuraciones Docker:

1. **[`docker-compose.yml`](docker-compose.yml):** Config básica con Ollama como Sabio, sin microservicios locales (Bibliotecario/Critic/Embedding).
2. **[`docker-compose-dmr.yml`](docker-compose-dmr.yml):** Config completa con DMR como Sabio + microservicios locales.
3. **[`docker-compose.local-models.yml`](docker-compose.local-models.yml):** Solo servicios locales de IA (embedding, bibliotecario, critic).

---

## 2. PUNTOS CRÍTICOS (Prioridad Alta)

### CRÍTICO-01: CONFLICTO DE ASIGNACIÓN DE GPU

**Gravedad:** Alta
**Frecuencia:** Siempre

**Descripción:**
Los archivos docker-compose asignan GPUs de forma inconsistente:
- [`docker-compose.yml`](docker-compose.yml:17) usa `device_ids: ["1"]` para Qdrant, Ollama y Tabby
- [`docker-compose-dmr.yml`](docker-compose-dmr.yml:55) usa `device_ids: ["0"]` para embedding_server, bibliotecario y critic
- [`docker-compose.local-models.yml`](docker-compose.local-models.yml:12) también usa `device_ids: ["0"]`

**Problema:**
La GTX 1650 del usuario tiene 4GB VRAM y un solo chip GPU. No existe GPU 1 física.
- TinyLlama-1.1B en bfloat16 = ~2.2 GB
- all-MiniLM-L6-v2 = ~150 MB
- Si se agregan más modelos, se supera la VRAM disponible

**Impacto:**
- OOM (Out of Memory) en containers
- Containers reiniciándose constantemente
- Sistema funcionalmente roto en hardware real

**Ubicación:**
- [`docker-compose.yml`](docker-compose.yml:17): líneas 14-18
- [`docker-compose.yml`](docker-compose.yml:54): líneas 51-54
- [`docker-compose-dmr.yml`](docker-compose-dmr.yml:55): líneas 51-56
- [`docker-compose.local-models.yml`](docker-compose.local-models.yml:12): líneas 9-13

---

### CRÍTICO-02: CÓDIGO SÍNCRONO DENTRO DE CONTEXTO ASYNC

**Gravedad:** Alta
**Frecuencia:** Siempre

**Descripción:**
En [`mcp-bridge/main.py`](mcp-bridge/main.py:238), la función `get_compressed_memory()` es síncrona pero llama a `call_bibliotecario_server()` que usa `httpx.AsyncClient` dentro de un span de tracing:

```python
# Línea 238-264
def get_compressed_memory(history: List[types.ChatMessage]) -> str:
    # ...
    bibliotecario_response = call_bibliotecario_server(full_history_text, last_query)
```

Las funciones `call_bibliotecario_server()`, `generate_search_variations()`, y `call_critic_server()` están declaradas como `async` pero son llamadas de forma síncrona desde `get_compressed_memory()` y `get_query_vector()`.

**Problema:**
Esto bloquea el event loop de uvicorn, haciendo que el servidor responda lentamente o timeout.

**Impacto:**
- Degradación severa de performance
- Timeout en llamadas del LLM
- Sensación de sistema "colgado"

**Ubicación:**
- [`mcp-bridge/main.py`](mcp-bridge/main.py:238): `get_compressed_memory()`
- [`mcp-bridge/main.py`](mcp-bridge/main.py:267): `get_query_vector()`
- [`mcp-bridge/main.py`](mcp-bridge/main.py:122): `call_bibliotecario_server()`
- [`mcp-bridge/main.py`](mcp-bridge/main.py:161): `generate_search_variations()`
- [`mcp-bridge/main.py`](mcp-bridge/main.py:194): `call_critic_server()`

---

### CRÍTICO-03: DUPLICACIÓN MASSIVA DE CÓDIGO

**Gravedad:** Media-Alta
**Frecuencia:** Siempre

**Descripción:**
[`mcp-bridge/stdio_server.py`](mcp-bridge/stdio_server.py) (~478 líneas) duplica la mayoría de la lógica de [`mcp-bridge/main.py`](mcp-bridge/main.py) (~653 líneas):

| Función | main.py | stdio_server.py | Diferencias |
|---------|---------|-----------------|-------------|
| `get_compressed_memory` | Línea 238 | Línea 88 | Prompts diferentes |
| `generate_search_variations` | Línea 161 | Línea 128 | main.py usa bibliotecario, stdio usa sabio directo |
| `get_query_vector` | Línea 267 | Línea 154 | stdio tiene fallback a sabio.embeddings |
| `rerank_search_results` | Línea 194 (via critic) | Línea 203 | stdio usa sabio directo |
| `get_project_structure` | Línea 321 | Línea 249 | Variaciones menores |

**Impacto:**
- Mantener dos copias del mismo código es propenso a errores
- Los prompts divergen entre implementaciones
- Cualquier fix debe aplicarse en DOS lugares

**Ubicación:**
- [`mcp-bridge/stdio_server.py`](mcp-bridge/stdio_server.py): líneas 86-450

---

### CRÍTICO-04: BUG EN IGNORED_PATHS DEL INDEXER

**Gravedad:** Alta
**Frecuencia:** Siempre

**Descripción:**
En [`indexer/indexer.py`](indexer/indexer.py:129-133):

```python
IGNORED_PATHS = {
    "node_modules", "/.","  /data/", "/venv/", "/dist/", "/build/",
    ...
}
```

El string `"/.","  /data/"` tiene:
1. Un espacio extra antes de `/data/`
2. El `"/."` no coincide con ningún path real de sistema de archivos

La función [`_is_path_ignored()`](indexer/indexer.py:136-153) hace checks individuales con `or`:
```python
return (
    "node_modules" in path_str
    or "/.." in path_str
    or "/data/" in path_str
    ...
)
```

**Problema:**
- `"/."` nunca coincidirá con paths reales
- `"/.."` busca dos puntos, pero los paths usan `/../` 
- Los paths relativos dentro del container no tendrán prefijo `/`

**Impacto:**
- Archivos de `node_modules` pueden ser indexados (consumo innecesario de VRAM)
- Directorios `.git/` pueden no ser ignorados correctamente
- Ruido en los embeddings

**Ubicación:**
- [`indexer/indexer.py`](indexer/indexer.py:129-133): `IGNORED_PATHS`
- [`indexer/indexer.py`](indexer/indexer.py:136-153): `_is_path_ignored()`

---

### CRÍTICO-05: INCONSISTENCIA DE NOMBRE DE COLECCIÓN QDRANT

**Gravedad:** Alta
**Frecuencia:** Siempre

**Descripción:**
Diferentes componentes usan diferentes nombres de colección:

| Componente | Valor | Archivo |
|------------|-------|---------|
| MCP Bridge (SSE) | `code_chunks` (default) | [`main.py`](mcp-bridge/main.py:37) |
| MCP Bridge (stdio) | `code_base` (hardcoded) | [`stdio_server.py`](mcp-bridge/stdio_server.py:44) |
| Indexer | `code_chunks` (default) | [`indexer/indexer.py`](indexer/indexer.py:27) |

**Impacto:**
- Si se usa stdio_server con el indexer, las búsquedas retornan 0 resultados
- El indexer crea `code_chunks` pero stdio busca en `code_base`

**Ubicación:**
- [`mcp-bridge/main.py`](mcp-bridge/main.py:37): `COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "code_chunks")`
- [`mcp-bridge/stdio_server.py`](mcp-bridge/stdio_server.py:44): `COLLECTION_NAME = "code_base"`

---

### CRÍTICO-06: ESCAPE INCORRECTO DE NEWLINES EN get_project_structure

**Gravedad:** Media
**Frecuencia:** Siempre

**Descripción:**
En [`mcp-bridge/main.py`](mcp-bridge/main.py:354):
```python
result = "\\n".join(tree)
```

Esto produce el string literal `\n` (backslash + n) en vez de un salto de línea real.

**Comparación con stdio_server.py:**
```python
# stdio_server.py línea 272 - CORRECTO
return "\n".join(structure)
```

**Impacto:**
- El árbol del proyecto se envía al LLM como una línea larga con `\n` literal
- El LLM no puede parsear correctamente la estructura del proyecto
- Búsquedas menos precisas

**Ubicación:**
- [`mcp-bridge/main.py`](mcp-bridge/main.py:354)

---

## 3. PUNTOS INCONSISTENTES (Prioridad Media)

### INCONSISTENTE-01: HEALTHCHECKS INCORRECTOS

**Descripción:**

1. **Ollama healthcheck** en [`docker-compose.yml`](docker-compose.yml:60):
```yaml
test: [ "CMD", "curl", "-f", "http://localhost:11434/api/tags" ]
```
Dentro del container, `localhost` puede no resolver correctamente. Debería usar `0.0.0.0` o el hostname.

2. **Critic Server puerto** en [`docker-compose-dmr.yml`](docker-compose-dmr.yml:88-98):
```yaml
ports:
  - "8003:8002"   # Expuesto en 8003
healthcheck:
  test: [ "CMD", "curl", "-f", "http://localhost:8002/health" ]  # Pero healthcheck mira 8002
```
El healthcheck debería apuntar al puerto expuesto (8002 desde dentro del container es correcto, pero desde el host es 8003).

3. **Tabby** en [`docker-compose.yml`](docker-compose.yml:70-71) usa imagen de registry privado sin healthcheck.

**Ubicación:**
- [`docker-compose.yml`](docker-compose.yml:59-64): Ollama healthcheck
- [`docker-compose-dmr.yml`](docker-compose-dmr.yml:88-103): Critic Server puerto

---

### INCONSISTENTE-02: BIBLIOTECARIO USADO PARA GENERACIÓN DE TEXTO

**Descripción:**
En [`mcp-bridge/main.py`](mcp-bridge/main.py:161-191), `generate_search_variations()` llama al endpoint `/summarize_and_identify/` del Bibliotecario para generar variaciones de búsqueda:

```python
bibliotecario_response = await call_bibliotecario_server(
    history="", query=prompt, timeout=timeout
)
```

El endpoint `/summarize_and_identify/` está diseñado para resumir conversaciones, no para generar texto arbitrario. Se está abusando del endpoint.

**Comparación con stdio_server.py:**
```python
# stdio_server.py línea 141 - CORRECTO
response_text = sabio.complete(prompt, timeout=5)
```

**Impacto:**
- El Bibliotecario (TinyLlama) no está optimizado para esta tarea
- Parsing inconsistente de respuestas
- Latencia innecesaria

**Ubicación:**
- [`mcp-bridge/main.py`](mcp-bridge/main.py:161-191): `generate_search_variations()`
- [`bibliotecario_server/app.py`](bibliotecario_server/app.py:45): `/summarize_and_identify/`

---

### INCONSISTENTE-03: PARSING FRÁGIL DE RESPUESTAS DEL BIBLIOTECARIO

**Descripción:**
En [`bibliotecario_server/app.py`](bibliotecario_server/app.py:85-101), el parsing se basa en tags fijos:

```python
summary_start_tag = "Resumen:\n"
entities_start_tag = "\nEntidades Clave:"

if summary_start_tag in generated_text:
    after_summary_tag = generated_text.split(summary_start_tag, 1)[1]
    if entities_start_tag in after_summary_tag:
        summary_text = after_summary_tag.split(entities_start_tag, 1)[0].strip()
        entities_raw = after_summary_tag.split(entities_start_tag, 1)[1].strip()
```

**Problema:**
- Si TinyLlama no genera exactamente "Resumen:\n" o "\nEntidades Clave:", el parsing falla silenciosamente
- No hay fallback ni validación de estructura
- El modelo puede agregar texto adicional que rompe el split

**Impacto:**
- Resumen vacío o entidades vacías
- Error silencioso sin logging

**Ubicación:**
- [`bibliotecario_server/app.py`](bibliotecario_server/app.py:85-101)

---

### INCONSISTENTE-04: PARSING FRÁGIL DE RESPUESTAS DEL CRÍTIC

**Descripción:**
En [`critic_server/app.py`](critic_server/app.py:97-115), el parsing de índices depende de que el modelo responda exactamente:

```python
response_part = generated_text.split(
    "Índices de los "
    + str(request.top_n)
    + " fragmentos más relevantes (ej: 0, 2, 4):\n"
)[-1].strip()
```

**Problema:**
- Si el modelo agrega un espacio extra, cambia el orden, o responde en inglés, el split no funciona
- No hay validación de que los índices sean consecutivos o lógicos

**Impacto:**
- Reranking retorna documentos aleatorios o vacíos
- Fallback silencioso a documentos originales

**Ubicación:**
- [`critic_server/app.py`](critic_server/app.py:97-115)

---

### INCONSISTENTE-05: PROMPT INYECTA ESTRUCTURA COMPLETA EN CADA REQUEST

**Descripción:**
En [`mcp-bridge/main.py`](mcp-bridge/main.py:574-589), el system_prompt incluye la estructura completa del proyecto:

```python
system_prompt = f"""
...
Estructura del proyecto (si es relevante, puedes obtenerla con get_project_structure):
{get_project_structure(Path(PROJECT_ROOT))}
...
"""
```

**Problema:**
- `get_project_structure()` escanea TODO el árbol de directorios en CADA request
- Para proyectos grandes (ej: un repo de React con 1000+ archivos), esto genera un string de MBs
- Esto consume tokens del context window del LLM innecesariamente
- La función es síncrona y bloqueante

**Impacto:**
- Latencia alta en cada request
- Posible excedencia del context window
- Token waste (dinero si se usa OpenAI)

**Ubicación:**
- [`mcp-bridge/main.py`](mcp-bridge/main.py:583)

---

### INCONSISTENTE-06: NO HAY CIRCUIT BREAKER NI RETRY

**Descripción:**
Todas las llamadas a servicios internos son directas sin retry ni circuit breaker:
- [`call_bibliotecario_server()`](mcp-bridge/main.py:122)
- [`call_critic_server()`](mcp-bridge/main.py:194)
- [`get_embedding_from_server()`](mcp-bridge/main.py:85)

**Problema:**
- Si un servicio cae (común con GPU OOM), todo el pipeline falla
- No hay retry con exponential backoff
- No hay fallback graceful degradation

**Impacto:**
- Un solo servicio caído = sistema completo caído
- No resiliencia ante fallos transitorios

**Ubicación:**
- [`mcp-bridge/main.py`](mcp-bridge/main.py:85-232): Todas las funciones de llamada

---

### INCONSISTENTE-07: MODELAMIENTO DE ERRORES INCONSISTENTE

**Descripción:**
Diferentes servicios manejan errores de forma inconsistente:

| Servicio | Comportamiento |
|----------|----------------|
| `embedding_server` | Lanza HTTP 500 si model es None |
| `bibliotecario_server` | Lanza HTTP 500 si text_generator es None |
| `critic_server` | Fallback silencioso a documentos originales |
| `main.py` | Log + retorna valores por defecto |

**Impacto:**
- Dificulta debugging
- El cliente no puede distinguir entre "error real" y "fallback"

---

## 4. PROBLEMAS DE DISEÑO (Prioridad Media-Baja)

### DISEÑO-01: MEMORY LEAK POTENCIAL EN INDEXER

**Descripción:**
En [`indexer/indexer.py`](indexer/indexer.py:66):
```python
file_hashes = {}
```

Este diccionario crece indefinidamente sin límite ni eviction policy.

**Impacto:**
- En proyectos con miles de archivos, consume memoria significativa
- No hay mecanismo de limpieza

**Ubicación:**
- [`indexer/indexer.py`](indexer/indexer.py:66)

---

### DISEÑO-02: NO HAY RATE LIMITING

**Descripción:**
No hay rate limiting en ningún servicio:
- MCP Bridge acepta requests ilimitados
- Bibliotecario/Critic pueden ser inundados
- Embedding server no limita concurrentes

**Impacto:**
- GPU puede ser saturada
- Respuestas lentas o timeout

---

### DISEÑO-03: TINYLLAMA NO OPTIMIZADO PARA ESPAÑOL/CÓDIGO

**Descripción:**
Tanto [`bibliotecario_server/app.py`](bibliotecario_server/app.py:13) como [`critic_server/app.py`](critic_server/app.py:16) usan `TinyLlama/TinyLlama-1.1B-Chat-v1.0`:
- Entrenado principalmente con datos en inglés
- Los prompts están en español pero el modelo responde mejor en inglés
- No está fine-tuned para análisis de código

**Impacto:**
- Resumen de baja calidad
- Entity identification imprecisa
- Reranking inconsistente

---

### DISEÑO-04: EMBEDDING SERVER SIN BATCH OPTIMIZATION

**Descripción:**
En [`embedding_server/app.py`](embedding_server/app.py:31):
```python
embeddings = model.encode(request.texts, convert_to_numpy=False, convert_to_tensor=True)
```

No hay batching optimizado ni padding para secuencias de diferente longitud.

**Impacto:**
- Performance subóptima en GPU
- No se aprovecha el paralelismo de tensor operations

---

## 5. DIAGRAMA DE ARQUITECTURA ACTUAL

```mermaid
graph TD
    subgraph "Máquina Local"
        A[Usuario / Editor] -->|MCP SSE| B[MCP Bridge:8002]
        A -->|MCP stdio| C[MCP Bridge stdio]
        B --> D[El Bibliotecario:8001]
        B --> E[El Crítico:8002]
        B --> F[Embedding Server:8000]
        B --> G[El Sabio - LLM]
        H[Indexer] --> F
        H --> I[Qdrant:6333]
        J[Ollama:11434] -->|Embeddings| F
        K[Phoenix:6006] -->|Traces| B
        K -->|Traces| H
    end
    
    subgraph "Máquina Remota 192.168.0.50"
        G -->|vLLM| L[vLLM Server]
    end
    
    D -->|Resumen + Entidades| B
    E -->|Reranking| B
    F -->|Embeddings| B
    G -->|Respuesta Final| B
    H -->|Watchdog| M[/app/code]
```

---

## 6. RESUMEN DE PROBLEMAS POR CATEGORÍA

| Categoría | Cantidad | Críticos |
|-----------|----------|----------|
| GPU/Infraestructura | 2 | 1 |
| Código/Concurrency | 2 | 1 |
| Duplicación | 1 | 0 |
| Bugs | 3 | 2 |
| Inconsistencias | 5 | 1 |
| Diseño/Arquitectura | 4 | 0 |
| Observabilidad | 1 | 0 |
| **TOTAL** | **18** | **5** |

---

## 7. MATRIZ DE RIESGO

| Problema | Probabilidad | Impacto | Riesgo Total |
|----------|--------------|---------|--------------|
| GPU OOM (CRÍTICO-01) | Alta | Crítico | **CRÍTICO** |
| Sync en Async (CRÍTICO-02) | Alta | Alto | **CRÍTICO** |
| Duplicación (CRÍTICO-03) | Siempre | Medio | **ALTO** |
| Ignored Paths (CRÍTICO-04) | Alta | Medio | **ALTO** |
| Colección Qdrant (CRÍTICO-05) | Siempre | Alto | **CRÍTICO** |
| Newline Escape (CRÍTICO-06) | Siempre | Medio | **MEDIO** |
| Healthchecks (INCONSISTENTE-01) | Alta | Bajo | **BAJO** |
| Bibliotecario abuse (INCONSISTENTE-02) | Siempre | Medio | **MEDIO** |
| Parsing frágil Bibliotecario (INCONSISTENTE-03) | Alta | Alto | **ALTO** |
| Parsing frágil Critic (INCONSISTENTE-04) | Alta | Alto | **ALTO** |
| Structure injection (INCONSISTENTE-05) | Siempre | Medio | **MEDIO** |
| Sin Circuit Breaker (INCONSISTENTE-06) | Alta | Alto | **ALTO** |
| Error handling (INCONSISTENTE-07) | Siempre | Bajo | **BAJO** |
| Memory leak (DISEÑO-01) | Media | Medio | **MEDIO** |
| Sin Rate Limiting (DISEÑO-02) | Media | Medio | **MEDIO** |
| TinyLlama idioma (DISEÑO-03) | Siempre | Medio | **MEDIO** |
| Embedding batch (DISEÑO-04) | Media | Bajo | **BAJO** |
