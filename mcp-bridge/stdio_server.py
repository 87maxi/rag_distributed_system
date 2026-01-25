import asyncio
import os
import sys
from typing import Dict, List

import mcp.types as types
import requests

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

# --- CONFIGURACIÓN ---
# Adjusted defaults for local usage (Zed editor)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_NAME = "code_base"
LOCAL_MODEL_SUMMARIZER = "qwen2.5:0.5b"
LOCAL_MODEL_CRITIC = "qwen2.5:1.5b"
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd()) 

# --- OBSERVABILITY SETUP ---
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
resource = Resource(attributes={"service.name": "rag-mcp-server-stdio"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer("rag.mcp")
otlp_exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

def log(msg):
    sys.stderr.write(f"SERVER: {msg}\n")
    sys.stderr.flush()

# Clientes
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
mcp_server = Server("mcp-rag-bridge-stdio")

# --- LOGGING SETUP ---
import logging

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests.log")

# Configurar logging basico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stderr) # Keep stderr for MCP protocol safety (logs shouldn't go to stdout)
    ]
)

logger = logging.getLogger("rag-mcp")

def log(msg):
    logger.info(msg)


@tracer.start_as_current_span("generate_search_variations")
def generate_search_variations(query: str) -> List[str]:
    """Genera 3 variaciones de la query para expansión."""
    span = trace.get_current_span()
    span.set_attribute("rag.input_query", query)
    
    prompt = f"Genera 3 variaciones de busqueda tecnica breves para: '{query}'. Responde solo las 3 frases separadas por linea sin enumerar."
    span.set_attribute("gen_ai.prompt", prompt)
    span.set_attribute("gen_ai.model", LOCAL_MODEL_SUMMARIZER)

    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_SUMMARIZER, "prompt": prompt, "stream": False},
            timeout=5,
        )
        response_text = res.json().get("response", "")
        span.set_attribute("gen_ai.response", response_text)
        
        lines = response_text.strip().split("\n")
        variations = [l.strip("- ").strip() for l in lines if l.strip()]
        
        span.set_attribute("rag.variations", str(variations))
        return variations[:3]
    except Exception as e:
        span.record_exception(e)
        return []

@tracer.start_as_current_span("rerank_search_results")
def rerank_search_results(query: str, results: List[any]) -> List[any]:
    """Usa el Crítico (1.5B) para ordenar los resultados."""
    span = trace.get_current_span()
    span.set_attribute("rag.initial_count", len(results))
    
    if not results: return []
    
    snippets = []
    for i, r in enumerate(results):
        snippets.append(f"[{i}] Path: {r.payload.get('path')} Content: {r.payload.get('content')[:100]}...")
    
    prompt = (
        f"Query: {query}\nSnippets:\n" + "\n".join(snippets) + 
        "\nSelect the indices of the 3 most relevant snippets (e.g. '0, 2, 4'). Return ONLY the numbers."
    )
    span.set_attribute("gen_ai.prompt", prompt)
    span.set_attribute("gen_ai.model", LOCAL_MODEL_CRITIC)
    
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_CRITIC, "prompt": prompt, "stream": False},
            timeout=10,
        )
        indices_str = res.json().get("response", "")
        span.set_attribute("gen_ai.response", indices_str)

        import re
        indices = [int(x) for x in re.findall(r"\d+", indices_str)]
        reranked = [results[i] for i in indices if i < len(results)]
        
        # Add remaining unique results if less than 3
        seen_ids = set(r.id for r in reranked)
        for r in results:
            if r.id not in seen_ids:
                reranked.append(r)
        
        span.set_attribute("rag.reranked_count", len(reranked))
        return reranked
    except Exception as e:
        log(f"Rerank failed: {e}")
        span.record_exception(e)
        return results

# --- 1. MÉTODO DEL BIBLIOTECARIO (Resumen del Historial) ---

@tracer.start_as_current_span("get_compressed_memory")
def get_compressed_memory(history: List[Dict[str, str]]):
    """Usa Qwen-0.5B local para resumir el historial largo."""
    span = trace.get_current_span()
    span.set_attribute("rag.history_length", len(history) if history else 0)

    if not history or len(history) < 1:
        log("Iniciando Bibliotecario con historial corto/vacío")
        return "No hay historial previo relevante."

    # Resumimos lo anterior a los últimos 2 mensajes para mantener el hilo vivo
    text_to_summarize = "\n".join(
        [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[:-2]]
    )

    prompt = (
        f"Resume esta conversación técnica de forma muy concisa. "
        f"Identifica explícitamente entidades clave (clases, funciones, archivos) mencionadas, "
        f"errores y decisiones de código. Ignora saludos o charla trivial:\n\n"
        f"{text_to_summarize}"
    )
    
    span.set_attribute("gen_ai.prompt", prompt)
    span.set_attribute("gen_ai.model", LOCAL_MODEL_SUMMARIZER)

    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_SUMMARIZER, "prompt": prompt, "stream": False},
            timeout=15,
        )
        summary = res.json().get("response", "Sin resumen disponible.")
        span.set_attribute("gen_ai.response", summary)
        return summary
    except Exception as e:
        log(f"Error en Resumen Local: {e}")
        span.record_exception(e)
        return "Error al procesar la memoria previa."


# --- 2. LÓGICA DE BÚSQUEDA (RAG) ---

@tracer.start_as_current_span("get_query_vector")
def get_query_vector(text: str):
    """Genera embeddings usando el modelo local nomic."""
    span = trace.get_current_span()
    span.set_attribute("rag.embed_text_len", len(text))
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=10,
        )
        data = res.json()
        vector = data.get("embeddings", [None])[0] or data.get("embedding")
        if vector:
            span.set_attribute("rag.vector_dim", len(vector))
        return vector
    except Exception as e:
        log(f"Error en Ollama Embed: {e}")
        span.record_exception(e)
        return None


# --- 3. NUEVAS HERRAMIENTAS ---

@tracer.start_as_current_span("get_project_structure")
def get_project_structure(root_path: str = ".") -> str:
    """Escanea el árbol de directorios ignorando carpetas irrelevantes."""
    start_dir = os.path.join(PROJECT_ROOT, root_path)
    if not os.path.exists(start_dir):
        return f"Error: Path {start_dir} not found."
    
    structure = []
    for root, dirs, files in os.walk(start_dir):
        # Ignore hidden and node_modules
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules" and d != "__pycache__" and d != "venv"]
        
        level = root.replace(start_dir, "").count(os.sep)
        indent = " " * 4 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            if not f.startswith("."):
                structure.append(f"{subindent}{f}")
                
    return "\n".join(structure)

@tracer.start_as_current_span("verify_code_syntax")
def verify_code_syntax(code: str, language: str) -> str:
    """Usa el modelo local para validar sintaxis."""
    span = trace.get_current_span()
    span.set_attribute("rag.language", language)
    span.set_attribute("rag.code_len", len(code))
    
    prompt = (
        f"Actua como un linter estricto. Analiza el siguiente codigo en {language} y detecta SOLO "
        f"errores fatales de sintaxis. Si es valido, responde solo 'VALID'. "
        f"Si hay error, responde 'ERROR: <descripcion breve>'. Código:\n\n{code}"
    )
    
    span.set_attribute("gen_ai.prompt", prompt)
    
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_SUMMARIZER, "prompt": prompt, "stream": False},
            timeout=10,
        )
        result = res.json().get("response", "Error validation failed").strip()
        span.set_attribute("gen_ai.response", result)
        return result
    except Exception as e:
        span.record_exception(e)
        return f"System Error: {e}"

# --- 4. MANEJADORES MCP ---

@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_code",
            description="Busca código y genera un prompt optimizado con memoria comprimida.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta técnica"},
                    "history": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Historial de chat",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_project_structure",
            description="Devuelve el árbol de archivos del proyecto para entender la estructura.",
            inputSchema={
                "type": "object",
                "properties": {
                    "root_path": {"type": "string", "description": "Ruta relativa opcional (default .)"}
                },
            },
        ),
        types.Tool(
            name="verify_syntax",
            description="Valida la sintaxis de un fragmento de código usando IA local.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Código a validar"},
                    "language": {"type": "string", "description": "Lenguaje (python, js, etc)"}
                },
                "required": ["code", "language"],
            },
        )
    ]


@mcp_server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    
    log(f"TOOL CALL: {name} | ARGS: {arguments}")

    # Iniciar span raíz para la herramienta
    with tracer.start_as_current_span(f"tool_call.{name}") as tool_span:
        tool_span.set_attribute("mcp.tool_name", name)
        
        if name == "get_project_structure":
            root_path = arguments.get("root_path", ".")
            tree = get_project_structure(root_path)
            return [types.TextContent(type="text", text=tree)]

        elif name == "verify_syntax":
            code = arguments.get("code", "")
            language = arguments.get("language", "python")
            result = verify_code_syntax(code, language)
            return [types.TextContent(type="text", text=result)]

        elif name == "search_code":
            query = arguments.get("query")
            history = arguments.get("history", [])
            tool_span.set_attribute("rag.query", query)
            
            # PASO A: El Bibliotecario genera el resumen
            summary_text = get_compressed_memory(history)
            
            # PASO B: Búsqueda Vectorial Expandida
            variations = generate_search_variations(query)
            log(f"Variations: {variations}")
            
            with tracer.start_as_current_span("vector_search_loop"):
                all_queries = [query] + variations
                hits_map = {}
                for q_var in all_queries:
                    vec = get_query_vector(q_var)
                    if vec:
                        res = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=5)
                        log(f"Search '{q_var}' -> {len(res)} hits")
                        for r in res:
                            hits_map[r.id] = r
            
            unique_results = list(hits_map.values())
            tool_span.set_attribute("rag.total_hits", len(unique_results))
            log(f"Total unique hits: {len(unique_results)}")
            
            # PASO B.2: Reranking
            top_results = rerank_search_results(query, unique_results)[:4]
            log(f"Top results after rerank: {len(top_results)}")

            # PASO C: Formatear Contexto
            code_snippets = []
            for res in top_results:
                path = res.payload.get("path", "desconocido")
                content = res.payload.get("content", "")
                code_snippets.append(f"ARCHIVO: {path}\n```\n{content}\n```")

            code_context = (
                "\n\n".join(code_snippets) if code_snippets else "Sin resultados."
            )
            log(f"Context length: {len(code_context)}")

            # PASO C.2: El Crítico Local (Evaluación de Suficiencia)
            with tracer.start_as_current_span("local_critic_eval"):
                critic_prompt = (
                    f"Analiza si el siguiente contexto de código es suficiente para responder a la consulta: '{query}'. "
                    f"Contexto:\n{code_context[:2000]}...\n"
                    f"Responde solo 'SUFFICIENT' o 'INSUFFICIENT' seguido de una explicacion de 1 linea."
                )
                tool_span.set_attribute("gen_ai.critic_prompt", critic_prompt)
                
                critic_judgement = "SKIP (Error)"
                try:
                     res_crit = requests.post(
                        f"{OLLAMA_HOST}/api/generate",
                        json={"model": LOCAL_MODEL_CRITIC, "prompt": critic_prompt, "stream": False},
                        timeout=15,
                    )
                     res_json = res_crit.json()
                     if "error" in res_json:
                         log(f"Ollama Error (Critic): {res_json['error']}")
                         critic_judgement = f"Error: {res_json['error']}"
                     else:
                         critic_judgement = res_json.get("response", "No judgment").strip()
                         log(f"Crítico Local: {critic_judgement}")
                except Exception as e:
                    log(f"Error Crítico: {e}")
                    tool_span.record_exception(e)
                
                tool_span.set_attribute("gen_ai.critic_judgement", critic_judgement)


            final_prompt = (
                f"### RESUMEN DE LA CONVERSACIÓN PREVIA\n{summary_text}\n\n"
                f"### EVALUACIÓN DEL CRÍTICO LOCAL\n{critic_judgement}\n\n"
                f"### CONTEXTO DE CÓDIGO RELEVANTE\n{code_context}\n\n"
                f"### CONSULTA ACTUAL\n{query}"
            )

            return [types.TextContent(type="text", text=final_prompt)]

async def func_main():
    log("📡 Servidor MCP RAG (Stdio) iniciado")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-rag-stdio",
                    server_version="1.0.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability(listChanged=True)
                    ),
                ),
            )
    finally:
        # Asegurar flush de trazas al cerrar
        trace.get_tracer_provider().shutdown()

if __name__ == "__main__":
    asyncio.run(func_main())
