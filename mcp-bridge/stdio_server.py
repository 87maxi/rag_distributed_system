"""
stdio_server.py — El Bibliotecario (modo stdio para Zed / Continue.dev local)

Pipeline RAG completo expuesto como servidor MCP via stdio.
NO contiene lógica de inferencia LLM directa: toda la generación pasa
por El Sabio a través de sabio_client.SabioClient.

Configuración (variables de entorno):
    QDRANT_HOST     — host de Qdrant          (default: localhost)
    SABIO_BASE_URL  — URL base de El Sabio    (default: http://localhost:11434)
    SABIO_MODEL     — Modelo de El Sabio      (default: qwen2.5:0.5b)
    SABIO_API_KEY   — API key (si aplica)     (default: none)
    PROJECT_ROOT    — Raíz del proyecto       (default: cwd)
    PHOENIX_ENDPOINT— Endpoint OTEL Phoenix   (default: http://localhost:6006/v1/traces)
"""

import asyncio
import os
import re
import sys
from typing import Dict, List

import mcp.types as types

# MCP Server
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from qdrant_client import QdrantClient

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# El Sabio — abstracción intercambiable del LLM
from sabio_client import SabioClient

# --- CONFIGURACIÓN ---
QDRANT_HOST   = os.getenv("QDRANT_HOST",   "localhost")
COLLECTION_NAME = "code_base"
PROJECT_ROOT  = os.getenv("PROJECT_ROOT",  os.getcwd())

# --- OBSERVABILITY SETUP ---
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
resource = Resource(attributes={"service.name": "rag-mcp-server-stdio"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer("rag.mcp")
otlp_exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

RequestsInstrumentor().instrument()

# --- LOGGING SETUP ---
import logging

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stderr),  # stderr es seguro para el protocolo MCP
    ]
)

logger = logging.getLogger("rag-mcp")

def log(msg):
    logger.info(msg)


# --- CLIENTES ---
qdrant   = QdrantClient(host=QDRANT_HOST, port=6333)
sabio    = SabioClient()          # Configurado via SABIO_* env vars
mcp_server = Server("mcp-rag-bridge-stdio")

log(f"El Sabio → {sabio.base_url}  modelo={sabio.model}")


# --- 1. EL BIBLIOTECARIO — Resumen del historial ---

@tracer.start_as_current_span("get_compressed_memory")
def get_compressed_memory(history: List[Dict[str, str]]) -> str:
    """
    Usa El Sabio para resumir el historial de conversación.
    Identifica entidades clave: clases, funciones, archivos, errores, decisiones.
    """
    span = trace.get_current_span()
    span.set_attribute("rag.history_length", len(history) if history else 0)

    if not history or len(history) < 1:
        log("Bibliotecario: historial corto/vacío, sin resumen")
        return "No hay historial previo relevante."

    # Resumimos todo menos los últimos 2 mensajes para mantener el hilo reciente
    text_to_summarize = "\n".join(
        [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[:-2]]
    )

    prompt = (
        f"Resume esta conversación técnica de forma muy concisa. "
        f"Identifica explícitamente entidades clave (clases, funciones, archivos) mencionadas, "
        f"errores y decisiones de código. Ignora saludos o charla trivial:\n\n"
        f"{text_to_summarize}"
    )

    span.set_attribute("gen_ai.prompt", prompt[:500])
    span.set_attribute("sabio.model",   sabio.model)

    try:
        summary = sabio.complete(prompt, timeout=15)
        span.set_attribute("gen_ai.response", summary[:500])
        return summary
    except Exception as e:
        log(f"Error en Resumen (El Sabio): {e}")
        span.record_exception(e)
        return "Error al procesar la memoria previa."


# --- 2. BÚSQUEDA VECTORIAL — Expansión de queries ---

@tracer.start_as_current_span("generate_search_variations")
def generate_search_variations(query: str) -> List[str]:
    """Genera 3 variaciones de la query para expandir la búsqueda vectorial."""
    span = trace.get_current_span()
    span.set_attribute("rag.input_query", query)

    prompt = (
        f"Genera 3 variaciones de búsqueda técnica breves para: '{query}'. "
        f"Responde solo las 3 frases separadas por línea nueva, sin enumerar."
    )
    span.set_attribute("gen_ai.prompt", prompt)

    try:
        response_text = sabio.complete(prompt, timeout=5)
        span.set_attribute("gen_ai.response", response_text)

        lines = response_text.strip().split("\n")
        variations = [l.strip("- ").strip() for l in lines if l.strip()]

        span.set_attribute("rag.variations", str(variations))
        return variations[:3]
    except Exception as e:
        span.record_exception(e)
        return []


@tracer.start_as_current_span("get_query_vector")
def get_query_vector(text: str):
    """
    Genera embeddings para la query.
    Intenta usar el embedding server dedicado; si no está disponible,
    cae de vuelta a la API de embeddings del propio El Sabio.
    """
    span = trace.get_current_span()
    span.set_attribute("rag.embed_text_len", len(text))

    import requests as _requests

    embedding_host = os.getenv("EMBEDDING_SERVER_HOST", "")
    if embedding_host:
        # Prioridad: embedding server dedicado (nomic-embed-text u otro)
        try:
            url = f"{embedding_host.rstrip('/')}/embed"
            res = _requests.post(url, json={"texts": [text]}, timeout=10)
            res.raise_for_status()
            data = res.json()
            vectors = data.get("embeddings", [])
            if vectors:
                span.set_attribute("rag.vector_dim", len(vectors[0]))
                return vectors[0]
        except Exception as e:
            log(f"Embedding server no disponible ({embedding_host}): {e}. Intentando con El Sabio...")

    # Fallback: /v1/embeddings de El Sabio (compatible con OpenAI)
    try:
        url = f"{sabio.base_url}/v1/embeddings"
        res = _requests.post(
            url,
            json={"model": sabio.model, "input": text},
            headers=sabio.headers,
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()
        vector = data["data"][0]["embedding"]
        span.set_attribute("rag.vector_dim", len(vector))
        return vector
    except Exception as e:
        log(f"Error generando embedding: {e}")
        span.record_exception(e)
        return None


# --- 3. EL CRÍTICO LOCAL — Reranking ---

@tracer.start_as_current_span("rerank_search_results")
def rerank_search_results(query: str, results: List) -> List:
    """Usa El Sabio para ordenar los resultados por relevancia."""
    span = trace.get_current_span()
    span.set_attribute("rag.initial_count", len(results))

    if not results:
        return []

    snippets = []
    for i, r in enumerate(results):
        snippets.append(
            f"[{i}] Path: {r.payload.get('path')} "
            f"Content: {r.payload.get('content', '')[:100]}..."
        )

    prompt = (
        f"Query: {query}\nSnippets:\n" + "\n".join(snippets) +
        "\nSelecciona los índices de los 3 fragmentos más relevantes (ej: '0, 2, 4'). "
        "Responde SOLO los números separados por coma."
    )
    span.set_attribute("gen_ai.prompt", prompt[:500])

    try:
        indices_str = sabio.complete(prompt, timeout=10)
        span.set_attribute("gen_ai.response", indices_str)

        indices = [int(x) for x in re.findall(r"\d+", indices_str)]
        reranked = [results[i] for i in indices if i < len(results)]

        # Rellenar con resultados restantes si faltan
        seen_ids = set(r.id for r in reranked)
        for r in results:
            if r.id not in seen_ids:
                reranked.append(r)

        span.set_attribute("rag.reranked_count", len(reranked))
        return reranked
    except Exception as e:
        log(f"Rerank fallido: {e}")
        span.record_exception(e)
        return results


# --- 4. HERRAMIENTAS AUXILIARES ---

@tracer.start_as_current_span("get_project_structure")
def get_project_structure(root_path: str = ".") -> str:
    """Escanea el árbol de directorios ignorando carpetas irrelevantes."""
    start_dir = os.path.join(PROJECT_ROOT, root_path)
    if not os.path.exists(start_dir):
        return f"Error: Path {start_dir} not found."

    structure = []
    ignored_folders = {
        "node_modules", "__pycache__", "venv", "data", "dist",
        "build", "out", ".git", ".github", ".vscode", ".next",
    }

    try:
        for root, dirs, files in os.walk(start_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ignored_folders]
            level   = root.replace(start_dir, "").count(os.sep)
            indent  = " " * 4 * level
            structure.append(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                if not f.startswith("."):
                    structure.append(f"{subindent}{f}")
        return "\n".join(structure)
    except Exception as e:
        return f"Error scanning structure: {e}"


@tracer.start_as_current_span("verify_code_syntax")
def verify_code_syntax(code: str, language: str) -> str:
    """Usa El Sabio para validar sintaxis de un fragmento de código."""
    span = trace.get_current_span()
    span.set_attribute("rag.language", language)
    span.set_attribute("rag.code_len", len(code))

    prompt = (
        f"Actúa como un linter estricto. Analiza el siguiente código en {language} y detecta SOLO "
        f"errores fatales de sintaxis. Si es válido, responde solo 'VALID'. "
        f"Si hay error, responde 'ERROR: <descripción breve>'. Código:\n\n{code}"
    )
    span.set_attribute("gen_ai.prompt", prompt[:500])

    try:
        result = sabio.complete(prompt, timeout=10)
        span.set_attribute("gen_ai.response", result)
        return result.strip()
    except Exception as e:
        span.record_exception(e)
        return f"System Error: {e}"


# --- 5. MANEJADORES MCP ---

@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Lista las herramientas disponibles."""
    return [
        types.Tool(
            name="search",
            description=(
                "SEARCH_CODE: Busca y analiza la codebase. "
                "Parámetros: 'query' (string). Usa esto para cualquier pregunta técnica."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La consulta de búsqueda."}
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="tree",
            description=(
                "GET_STRUCTURE: Lista el árbol de directorios del proyecto. "
                "Parámetro opcional: 'root_path' (string, default '.')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "root_path": {"type": "string", "description": "Carpeta a listar.", "default": "."}
                },
            },
        ),
        types.Tool(
            name="lint",
            description=(
                "VERIFY_SYNTAX: Verifica si un fragmento de código tiene errores. "
                "Parámetros: 'code' y 'language'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code":     {"type": "string", "description": "Fragmento de código."},
                    "language": {"type": "string", "description": "Lenguaje del código."},
                },
                "required": ["code", "language"],
            },
        ),
    ]


@mcp_server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:

    if arguments is None:
        arguments = {}

    log(f"TOOL CALL: {name} | ARGS: {arguments}")

    with tracer.start_as_current_span(f"tool_call.{name}") as tool_span:
        tool_span.set_attribute("mcp.tool_name", name)

        if name == "tree":
            root_path = arguments.get("root_path", ".")
            tree_out  = get_project_structure(root_path)
            return [types.TextContent(type="text", text=tree_out)]

        elif name == "lint":
            code     = arguments.get("code", "")
            language = arguments.get("language", "python")
            result   = verify_code_syntax(code, language)
            return [types.TextContent(type="text", text=result)]

        elif name == "search":
            query   = arguments.get("query")
            history = arguments.get("history", [])
            tool_span.set_attribute("rag.query", query)

            # PASO A: El Bibliotecario genera el resumen del historial
            summary_text = get_compressed_memory(history)
            log(f"Resumen del historial generado ({len(summary_text)} chars)")

            # PASO B: Expansión de la query y búsqueda vectorial
            variations = generate_search_variations(query)
            log(f"Variaciones generadas: {variations}")

            with tracer.start_as_current_span("vector_search_loop"):
                all_queries = [query] + variations
                hits_map: dict = {}
                for q_var in all_queries:
                    vec = get_query_vector(q_var)
                    if vec:
                        res = qdrant.search(
                            collection_name=COLLECTION_NAME,
                            query_vector=vec,
                            limit=5,
                        )
                        log(f"Search '{q_var}' → {len(res)} hits")
                        for r in res:
                            hits_map[r.id] = r

            unique_results = list(hits_map.values())
            tool_span.set_attribute("rag.total_hits", len(unique_results))
            log(f"Total unique hits: {len(unique_results)}")

            # PASO B.2: Reranking con El Sabio (rol Crítico)
            top_results = rerank_search_results(query, unique_results)[:4]
            log(f"Top results tras rerank: {len(top_results)}")

            # PASO C: Formatear contexto enriquecido
            code_snippets = []
            for res in top_results:
                path    = res.payload.get("path", "desconocido")
                content = res.payload.get("content", "")
                code_snippets.append(f"ARCHIVO: {path}\n```\n{content}\n```")

            code_context = (
                "\n\n".join(code_snippets) if code_snippets else "Sin resultados."
            )
            log(f"Context length: {len(code_context)}")

            # PASO C.2: Evaluación de suficiencia por El Sabio (rol Crítico)
            with tracer.start_as_current_span("critic_eval"):
                critic_prompt = (
                    f"Analiza si el siguiente contexto de código es suficiente para responder a la consulta: '{query}'. "
                    f"Contexto:\n{code_context[:2000]}...\n"
                    f"Responde solo 'SUFFICIENT' o 'INSUFFICIENT' seguido de una explicación de 1 línea."
                )
                tool_span.set_attribute("gen_ai.critic_prompt", critic_prompt[:500])

                critic_judgement = "SKIP"
                try:
                    critic_judgement = sabio.complete(critic_prompt, timeout=15)
                    log(f"Crítico: {critic_judgement}")
                except Exception as e:
                    log(f"Error Crítico: {e}")
                    tool_span.record_exception(e)

                tool_span.set_attribute("gen_ai.critic_judgement", critic_judgement)

            # Prompt final enriquecido para el LLM principal (el editor)
            final_prompt = (
                f"### RESUMEN DE LA CONVERSACIÓN PREVIA\n{summary_text}\n\n"
                f"### EVALUACIÓN DEL CRÍTICO\n{critic_judgement}\n\n"
                f"### CONTEXTO DE CÓDIGO RELEVANTE\n{code_context}\n\n"
                f"### CONSULTA ACTUAL\n{query}"
            )

            return [types.TextContent(type="text", text=final_prompt)]

        else:
            return [types.TextContent(type="text", text=f"Herramienta '{name}' no encontrada.")]


async def func_main():
    log("📡 El Bibliotecario (MCP stdio) iniciado")
    log(f"   El Sabio → {sabio.base_url}  modelo={sabio.model}")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-rag-stdio",
                    server_version="2.0.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability(listChanged=True)
                    ),
                ),
            )
    finally:
        trace.get_tracer_provider().shutdown()


if __name__ == "__main__":
    asyncio.run(func_main())
