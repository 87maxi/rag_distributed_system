import os
import requests
import asyncio
import uuid
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# --- Configuración de Observabilidad (Phoenix) ---
resource_attributes = {"service.name": "mcp-bridge-search"}
trace.set_tracer_provider(TracerProvider())
# Usamos el nombre del servicio definido en docker-compose
otlp_exporter = OTLPSpanExporter(endpoint="http://rag_phoenix:4318/v1/traces")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
tracer = trace.get_tracer(__name__)

# Configuración
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
COLLECTION_NAME = "code_base"

qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
mcp_server = Server("code-rag-direct")

@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_code",
            description="Busca código semánticamente en Solidity, TS, Rust y Python",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Término o concepto técnico a buscar"},
                },
                "required": ["query"]
            }
        )
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    if name == "search_code":
        query = arguments.get("query")
        
        with tracer.start_as_current_span("mcp_search_operation") as span:
            span.set_attribute("search.query", query)
            try:
                # 1. Generar Embedding
                res = requests.post(f"{OLLAMA_HOST}/api/embed", 
                                    json={"model": "nomic-embed-text", "input": query})
                vector = res.json()["embeddings"][0]
                
                # 2. Búsqueda en Qdrant
                results = qdrant.search(
                    collection_name=COLLECTION_NAME, 
                    query_vector=vector, 
                    limit=4 # Balance entre contexto y ahorro de tokens
                )
                
                # Formatear respuesta
                if not results:
                    return [TextContent(type="text", text="No se encontraron fragmentos relevantes.")]
                
                text_res = "\n---\n".join([
                    f"Archivo: {r.payload.get('path')}\nContenido: {r.payload.get('content')}" 
                    for r in results
                ])
                
                span.set_attribute("search.results_count", len(results))
                return [TextContent(type="text", text=text_res)]
            except Exception as e:
                span.record_exception(e)
                return [TextContent(type="text", text=f"Error en la búsqueda: {str(e)}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())