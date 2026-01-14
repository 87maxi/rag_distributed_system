import os
import sys
import asyncio
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
import mcp.types as types
from qdrant_client import QdrantClient

# Configuración
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
COLLECTION_NAME = "code_base"

# Utils
def log(msg):
    sys.stderr.write(f"SERVER: {msg}\n")
    sys.stderr.flush()

qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
mcp_server = Server("mcp-rag-bridge")

def get_query_vector(text):
    try:
        res = requests.post(f"{OLLAMA_HOST}/api/embed", json={"model": "nomic-embed-text", "input": text}, timeout=5)
        data = res.json()
        return data.get("embeddings", [None])[0] or data.get("embedding")
    except:
        return None

@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_code",
            description="Busca fragmentos de código relevantes en el repositorio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Concepto o código a buscar"}
                },
                "required": ["query"]
            }
        )
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name == "search_code":
        query = arguments.get("query")
        log(f"Búsqueda recibida: {query}")
        
        vector = get_query_vector(query)
        if not vector:
            return [types.TextContent(type="text", text="Error: No se pudo conectar con Ollama.")]

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=5
        )

        formatted_results = []
        for res in results:
            path = res.payload.get("path", "desconocido")
            content = res.payload.get("content", "")
            formatted_results.append(f"--- ARCHIVO: {path} ---\n{content}")

        final_text = "\n\n".join(formatted_results)
        return [types.TextContent(type="text", text=final_text or "No se encontraron resultados relevantes.")]

async def main():
    log("Iniciando MCP Server sobre STDIO...")
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-rag-bridge",
                server_version="1.0.0",
                capabilities=types.ServerCapabilities(
                    tools=types.ToolsCapability(listChanged=True)
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())