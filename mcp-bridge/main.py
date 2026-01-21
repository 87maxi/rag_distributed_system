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

# --- CONFIGURACIÓN ---
# Sincronizado con nombres de servicios en docker-compose.yml
QDRANT_HOST = os.getenv("QDRANT_HOST", "rag_qdrant")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://rag_ollama:11434")
COLLECTION_NAME = "code_base"
LOCAL_MODEL_SUMMARIZER = "qwen2.5:0.5b"  # El "Bibliotecario" local


def log(msg):
    sys.stderr.write(f"SERVER: {msg}\n")
    sys.stderr.flush()


# Clientes
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
mcp_server = Server("mcp-rag-bridge")
sse = SseServerTransport("/messages")

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
        f"Enfócate en archivos mencionados, errores y decisiones de código:\n\n"
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


# --- 3. MANEJADORES MCP ---


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
        )
    ]


@mcp_server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    if name == "search_code":
        query = arguments.get("query")
        history = arguments.get("history", [])

        # PASO A: El Bibliotecario genera el resumen (Llamada al método 1)
        summary_text = get_compressed_memory(history)

        # PASO B: Búsqueda Vectorial
        vector = get_query_vector(query)
        if not vector:
            return [types.TextContent(type="text", text="Error al generar vectores.")]

        search_results = qdrant.search(
            collection_name=COLLECTION_NAME, query_vector=vector, limit=5
        )

        # PASO C: Formatear Contexto (Usando 'path' para coincidir con indexer.py)
        code_snippets = []
        for res in search_results:
            path = res.payload.get("path", "desconocido")
            content = res.payload.get("content", "")
            code_snippets.append(f"ARCHIVO: {path}\n```\n{content}\n```")

        code_context = (
            "\n\n".join(code_snippets) if code_snippets else "Sin resultados."
        )

        # PASO D: El "Super-Prompt" para el modelo remoto
        final_prompt = (
            f"### RESUMEN DE LA CONVERSACIÓN PREVIA\n{summary_text}\n\n"
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
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=logged_sse_app),
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
