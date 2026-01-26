import asyncio
import os
import sys
from typing import Dict, List

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
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# --- CONFIGURACIÓN ---
# Sincronizado con nombres de servicios en docker-compose.yml
QDRANT_HOST = os.getenv("QDRANT_HOST", "rag_qdrant")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://rag_ollama:11434")
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

# Instrumentar requests para capturar llamadas a Ollama automáticamente
RequestsInstrumentor().instrument()

# --- HELPER FUNCTIONS (Phase 3) ---

def generate_search_variations(query: str) -> List[str]:
    """Genera 3 variaciones de la query para expansión."""
    prompt = f"Genera 3 variaciones de busqueda tecnica breves para: '{query}'. Responde solo las 3 frases separadas por linea sin enumerar."
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_SUMMARIZER, "prompt": prompt, "stream": False},
            timeout=5,
        )
        lines = res.json().get("response", "").strip().split("\n")
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
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_CRITIC, "prompt": prompt, "stream": False},
            timeout=10,
        )
        indices_str = res.json().get("response", "")
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
    # CORRECCIÓN: Se agrega return para evitar que el código siga bajando sin datos
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
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_SUMMARIZER, "prompt": prompt, "stream": False},
            timeout=15,
        )
        return res.json().get("response", "Sin resumen disponible.")
    except Exception as e:
        log(f"Error en Resumen Local: {e}")
        return "Error al procesar la memoria previa."


# --- 2. LÓGICA DE BÚSQUEDA (RAG) ---


def get_query_vector(text: str):
    """Genera embeddings usando el modelo local nomic."""
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=10,
        )
        data = res.json()
        return data.get("embeddings", [None])[0] or data.get("embedding")
    except Exception as e:
        log(f"Error en Ollama Embed: {e}")
        return None


# --- 3. NUEVAS HERRAMIENTAS (PHASE 2) ---

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

def verify_code_syntax(code: str, language: str) -> str:
    """Usa el modelo local para validar sintaxis."""
    prompt = (
        f"Actua como un linter estricto. Analiza el siguiente codigo en {language} y detecta SOLO "
        f"errores fatales de sintaxis. Si es valido, responde solo 'VALID'. "
        f"Si hay error, responde 'ERROR: <descripcion breve>'. Código:\n\n{code}"
    )
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LOCAL_MODEL_SUMMARIZER, "prompt": prompt, "stream": False},
            timeout=10,
        )
        return res.json().get("response", "Error validation failed").strip()
    except Exception as e:
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
    log("📡 Servidor MCP RAG iniciado en puerto 8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
