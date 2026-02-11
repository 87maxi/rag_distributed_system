import asyncio
import os
import sys
from typing import Dict, List, Any

import mcp.types as types
import requests

# Servidor Web y MCP
import uvicorn
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport
from qdrant_client import QdrantClient
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.routing import Mount, Route

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# --- CONFIGURACIÓN ---
# Sincronizado con nombres de servicios en docker-compose.yml
QDRANT_HOST = os.getenv("QDRANT_HOST", "rag_qdrant")
LLM_HOST = os.getenv("OLLAMA_HOST", "http://rag_ollama:11434") # Generic host env var (can point to DMR)
LLM_API_TYPE = os.getenv("LLM_API_TYPE", "ollama") # 'ollama' or 'openai' (for DMR)
COLLECTION_NAME = "code_base"
LOCAL_MODEL_SUMMARIZER = "qwen2.5:0.5b"  # El "Bibliotecario" local
LOCAL_MODEL_CRITIC = "qwen2.5:1.5b"  # El "Crítico" local
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/app/code")


def log(msg):
    sys.stderr.write(f"SERVER: {msg}\n")
    sys.stderr.flush()


# Clientes
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
mcp_server = Server("mcp-rag-bridge")
sse = SseServerTransport("/messages")

# --- OBSERVABILITY SETUP (Phase 4) ---
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://rag_phoenix:4318/v1/traces")
resource = Resource(attributes={"service.name": "rag-mcp-server"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer("rag.mcp")
otlp_exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrumentar requests para capturar llamadas a Ollama/DMR automáticamente
RequestsInstrumentor().instrument()

# --- 0. WRAPPER FOR OBSERVABILITY & ABSTRACTION (Phase 4 & Refactor) ---

def call_llm_generic(endpoint_type: str, model: str, payload: Dict[str, Any], timeout=30) -> Dict[str, Any]:
    """
    Abstracción para llamar al LLM soportando Ollama y OpenAI (DMR).
    endpoint_type: 'generate' (completion/chat) or 'embed' (embeddings)
    """
    # Force use of configured model if provided, but function arg takes precedence if specific
    
    if LLM_API_TYPE == "ollama":
        return _call_ollama(endpoint_type, model, payload, timeout)
    elif LLM_API_TYPE == "openai":
        return _call_openai_compatible(endpoint_type, model, payload, timeout)
    else:
        # Tech debt: Default to ollama if unspecified or typo, but maybe log warning
        return _call_ollama(endpoint_type, model, payload, timeout)

def _call_ollama(endpoint_type, model, payload, timeout):
    # Map generic types to Ollama endpoints
    suffix = "/api/generate" if endpoint_type == "generate" else "/api/embed"
    
    # Payload adaptation
    final_payload = payload.copy()
    final_payload["model"] = model
    if endpoint_type == "generate":
         final_payload["stream"] = False
    
    return _execute_http_call(suffix, final_payload, timeout, system="ollama")

def _call_openai_compatible(endpoint_type, model, payload, timeout):
    # Map generic types to OpenAI endpoints
    suffix = "/v1/chat/completions" if endpoint_type == "generate" else "/v1/embeddings"
    
    final_payload = {}
    if endpoint_type == "generate":
        # Adapt Ollama 'prompt' to OpenAI 'messages'
        prompt = payload.get("prompt", "")
        final_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": payload.get("temperature", 0.7)
        }
    else:
        # Embeddings
        final_payload = {
            "model": model, 
            "input": payload.get("input", "")
        }

    response_data = _execute_http_call(suffix, final_payload, timeout, system="openai")
    
    # Normalize response to match Ollama format expected by app logic
    normalized = {}
    if endpoint_type == "generate":
        try:
            content = response_data["choices"][0]["message"]["content"]
            normalized["response"] = content
        except (KeyError, IndexError, TypeError):
            normalized["response"] = "" 
            normalized["error"] = f"Invalid response from DMR: {response_data}"
    else:
        # Embeddings
        try:
            # OpenAI format: data: [{embedding: [...]}]
            data = response_data["data"][0]["embedding"]
            normalized["embeddings"] = [data] # Ollama format emulation
            normalized["embedding"] = data
        except (KeyError, IndexError, TypeError):
             normalized["embedding"] = None

    return normalized

def _execute_http_call(endpoint_suffix, payload, timeout, system="llm"):
    """Realiza la llamada HTTP real envuelta en un span."""
    # Ensure LLM_HOST doesn't end with / if suffix starts with /
    base_host = LLM_HOST.rstrip("/")
    url = f"{base_host}{endpoint_suffix}"
    
    input_value = payload.get("prompt") or payload.get("input") or str(payload)
    model = payload.get("model", "unknown")
    
    with tracer.start_as_current_span(f"{system}_call") as span:
        span.set_attribute("llm.system", system)
        span.set_attribute("llm.request.type", "chat" if "chat" in url or "generate" in url else "embedding")
        span.set_attribute("llm.request.model", model)
        span.set_attribute("input.value", str(input_value))
        
        try:
            # Verify DMR headers if needed (usually not for local)
            res = requests.post(url, json=payload, timeout=timeout)
            res.raise_for_status()
            
            try:
                data = res.json()
                # Log outcome check (truncated)
                output_val = "..." 
                if "choices" in data: 
                    output_val = str(data["choices"][0]["message"]["content"])[:50]
                elif "response" in data: 
                    output_val = str(data["response"])[:50]
                elif "data" in data or "embedding" in data: 
                    output_val = "<embedding_vector>"
                
                span.set_attribute("output.value", output_val)
                return data
            except Exception:
                span.set_attribute("output.value", res.text)
                return res.json() # Try parsing anyway
                
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            # Don't crash server, return error dict
            return {"error": str(e)}

# --- HELPER FUNCTIONS (Phase 3) ---

def generate_search_variations(query: str) -> List[str]:
    """Genera 3 variaciones de la query para expansión."""
    prompt = f"Genera 3 variaciones de busqueda tecnica breves para: '{query}'. Responde solo las 3 frases separadas por linea sin enumerar."
    try:
        res_json = call_llm_generic(
            "generate",
            LOCAL_MODEL_SUMMARIZER,
            {"prompt": prompt},
            timeout=5,
        )
        lines = res_json.get("response", "").strip().split("\n")
        variations = [l.strip("- ").strip() for l in lines if l.strip()]
        return variations[:3]
    except:
        return []

def rerank_search_results(query: str, results: List[any]) -> List[any]:
    """Usa el Crítico (1.5B) para ordenar los resultados."""
    if not results: return []
    
    snippets = []
    for i, r in enumerate(results):
        snippets.append(f"[{i}] Path: {r.payload.get('path')} Content: {r.payload.get('content')[:100]}...")
    
    prompt = (
        f"Query: {query}\nSnippets:\n" + "\n".join(snippets) + 
        "\nSelect the indices of the 3 most relevant snippets (e.g. '0, 2, 4'). Return ONLY the numbers."
    )
    
    try:
        res_json = call_llm_generic(
            "generate",
            LOCAL_MODEL_CRITIC,
            {"prompt": prompt},
            timeout=10,
        )
        indices_str = res_json.get("response", "")
        import re
        indices = [int(x) for x in re.findall(r"\d+", indices_str)]
        reranked = [results[i] for i in indices if i < len(results)]
        # Add remaining unique results if less than 3
        seen_ids = set(r.id for r in reranked)
        for r in results:
            if r.id not in seen_ids:
                reranked.append(r)
        
        return reranked
    except Exception as e:
        log(f"Rerank failed: {e}")
        return results

# --- 1. MÉTODO DEL BIBLIOTECARIO (Resumen del Historial) ---


def get_compressed_memory(history: List[Dict[str, str]]):
    """Usa Qwen-0.5B local para resumir el historial largo."""
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

    try:
        res_json = call_llm_generic(
            "generate",
            LOCAL_MODEL_SUMMARIZER,
            {"prompt": prompt},
            timeout=15,
        )
        return res_json.get("response", "Sin resumen disponible.")
    except Exception as e:
        log(f"Error en Resumen Local: {e}")
        return "Error al procesar la memoria previa."


# --- 2. LÓGICA DE BÚSQUEDA (RAG) ---


def get_query_vector(text: str):
    """Genera embeddings usando el modelo local nomic."""
    try:
        data = call_llm_generic(
            "embed",
            "nomic-embed-text",
            {"input": text},
            timeout=10,
        )
        return data.get("embeddings", [None])[0] or data.get("embedding")
    except Exception as e:
        log(f"Error en Embed: {e}")
        return None


# --- 3. NUEVAS HERRAMIENTAS (PHASE 2) ---

def get_project_structure(root_path: str = ".") -> str:
    """Escanea el árbol de directorios ignorando carpetas irrelevantes."""
    start_dir = os.path.join(PROJECT_ROOT, root_path)
    if not os.path.exists(start_dir):
        return f"Error: Path {start_dir} not found."
    
    structure = []
    # Carpetas a ignorar explícitamente para evitar ruido/blobs enormes
    ignored_folders = {"node_modules", "__pycache__", "venv", "data", "dist", "build", "out", ".git", ".github", ".vscode", ".next"}
    
    try:
        for root, dirs, files in os.walk(start_dir):
            # Ignore hidden and ignored folders
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ignored_folders]
            
            level = root.replace(start_dir, "").count(os.sep)
            indent = " " * 4 * level
            structure.append(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                if not f.startswith("."):
                    structure.append(f"{subindent}{f}")
                    
        return "\n".join(structure)
    except Exception as e:
        return f"Error scanning structure: {e}"

def verify_code_syntax(code: str, language: str) -> str:
    """Usa el modelo local para validar sintaxis."""
    prompt = (
        f"Actua como un linter estricto. Analiza el siguiente codigo en {language} y detecta SOLO "
        f"errores fatales de sintaxis. Si es valido, responde solo 'VALID'. "
        f"Si hay error, responde 'ERROR: <descripcion breve>'. Código:\n\n{code}"
    )
    try:
        res_json = call_llm_generic(
            "generate",
            LOCAL_MODEL_SUMMARIZER,
            {"prompt": prompt},
            timeout=10,
        )
        return res_json.get("response", "Error validation failed").strip()
    except Exception as e:
        return f"System Error: {e}"

# --- 4. MANEJADORES MCP ---


@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Lista las herramientas disponibles con nombres cortos para modelos locales."""
    return [
        types.Tool(
            name="search",
            description=(
                "SEARCH_CODE: Use this to search and analyze the codebase. "
                "Takes 'query' (string). Use this for any technical question."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="tree",
            description=(
                "GET_STRUCTURE: Use this to list the directory tree and find files. "
                "Takes 'root_path' (string, default '.')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "root_path": {"type": "string", "description": "Folder to list.", "default": "."}
                },
            },
        ),
        types.Tool(
            name="lint",
            description=(
                "VERIFY_SYNTAX: Use this to check if a code snippet has errors. "
                "Takes 'code' and 'language'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code snippet."},
                    "language": {"type": "string", "description": "Language name."}
                },
                "required": ["code", "language"],
            },
        )
    ]


@mcp_server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    if arguments is None:
        arguments = {}
    if name == "tree":
        root_path = arguments.get("root_path", ".")
        tree_out = get_project_structure(root_path)
        return [types.TextContent(type="text", text=tree_out)]

    elif name == "lint":
        code = arguments.get("code", "")
        language = arguments.get("language", "python")
        result = verify_code_syntax(code, language)
        return [types.TextContent(type="text", text=result)]

    elif name == "search":
        query = arguments.get("query")
        # history no longer strictly required in schema for search
        history = arguments.get("history", [])
        
        with tracer.start_as_current_span("search_code") as span:
            span.set_attribute("rag.query", query)

            # PASO A: El Bibliotecario genera el resumen (Llamada al método 1)
            summary_text = get_compressed_memory(history)

            # PASO B: Búsqueda Vectorial
            # PASO B: Búsqueda Vectorial Expandida (Phase 3)
            variations = generate_search_variations(query)
            span.set_attribute("rag.variations", str(variations))
            
            all_queries = [query] + variations
            
            hits_map = {}
            for q_var in all_queries:
                vec = get_query_vector(q_var)
                if vec:
                    res = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=5)
                    for r in res:
                        hits_map[r.id] = r # Deduplication by ID
            
            unique_results = list(hits_map.values())
            
            # PASO B.2: Reranking (Phase 3)
            top_results = rerank_search_results(query, unique_results)[:4] # Top 4
            span.set_attribute("rag.results_count", len(unique_results))

            # PASO C: Formatear Contexto (Usando 'path' para coincidir con indexer.py)
            code_snippets = []
            # PASO C: Formatear Contexto
            code_snippets = []
            for res in top_results:
                path = res.payload.get("path", "desconocido")
                content = res.payload.get("content", "")
                code_snippets.append(f"ARCHIVO: {path}\n```\n{content}\n```")

            code_context = (
                "\n\n".join(code_snippets) if code_snippets else "Sin resultados."
            )

            # PASO C.2: El Crítico Local (Evaluación de Suficiencia)
            critic_prompt = (
                f"Analiza si el siguiente contexto de código es suficiente para responder a la consulta: '{query}'. "
                f"Contexto:\n{code_context[:2000]}...\n" # Truncar para no saturar contexto
                f"Responde solo 'SUFFICIENT' o 'INSUFFICIENT' seguido de una explicacion de 1 linea."
            )
            critic_judgement = "SKIP (Error)"
            try:
                 res_json = call_llm_generic(
                    "generate",
                    LOCAL_MODEL_CRITIC,
                    {"prompt": critic_prompt},
                    timeout=15,
                )
                 if "error" in res_json:
                     log(f"LLM Error (Critic): {res_json['error']}")
                     critic_judgement = f"Error: {res_json['error']}"
                 else:
                     critic_judgement = res_json.get("response", "No judgment").strip()
                     log(f"Crítico Local: {critic_judgement}")
            except Exception as e:
                log(f"Error Crítico: {e}")


        # PASO D: El "Super-Prompt" para el modelo remoto
        final_prompt = (
            f"### RESUMEN DE LA CONVERSACIÓN PREVIA\n{summary_text}\n\n"
            f"### EVALUACIÓN DEL CRÍTICO LOCAL\n{critic_judgement}\n\n"
            f"### CONTEXTO DE CÓDIGO RELEVANTE\n{code_context}\n\n"
            f"### CONSULTA ACTUAL\n{query}"
        )

        return [types.TextContent(type="text", text=final_prompt)]


# --- 4. SERVIDOR ASGI ---


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        rs,
        ws,
    ):
        await mcp_server.run(
            rs,
            ws,
            InitializationOptions(
                server_name="mcp-rag-sse",
                server_version="1.0.0",
                capabilities=types.ServerCapabilities(
                    tools=types.ToolsCapability(listChanged=True)
                ),
            ),
        )
    return Response()


async def logged_sse_app(scope, receive, send):
    if scope["type"] == "http" and scope["method"] == "POST":
        body = b""
        more_body = True
        while more_body:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                more_body = msg.get("more_body", False)

        async def new_receive():
            return {"type": "http.request", "body": body, "more_body": False}

        await sse.handle_post_message(scope, new_receive, send)
    else:
        await sse.handle_post_message(scope, receive, send)


app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET", "POST"]),
        Route("/messages", endpoint=logged_sse_app, methods=["POST"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

if __name__ == "__main__":
    log(f"📡 Servidor MCP RAG iniciado en puerto 8002. Modo LLM: {LLM_API_TYPE}")
    uvicorn.run(app, host="0.0.0.0", port=8002)
