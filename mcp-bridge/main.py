import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import mcp.types as types
import httpx

# Servidor Web y MCP
import uvicorn
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from qdrant_client import QdrantClient, models
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

# --- CONFIGURACIÓN ---
# Sincronizado con nombres de servicios en docker-compose.yml
QDRANT_HOST = os.getenv("QDRANT_HOST", "rag_qdrant")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "code_chunks")

# Hosts para los servicios locales de IA
EMBEDDING_SERVER_HOST = os.getenv(
    "EMBEDDING_SERVER_HOST", "http://rag_embedding_server:8000"
)
BIBLIOTECARIO_SERVER_HOST = os.getenv(
    "BIBLIOTECARIO_SERVER_HOST", "http://rag_bibliotecario:8001"
)
CRITIC_SERVER_HOST = os.getenv("CRITIC_SERVER_HOST", "http://rag_critic_server:8002")

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/app/code")


def log(msg):
    sys.stderr.write(f"SERVER: {msg}\n")
    sys.stderr.flush()


# Clientes
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
mcp_server = Server("mcp-rag-bridge")
sse = SseServerTransport("/messages")

# El Sabio — abstracción intercambiable del LLM (configurado via SABIO_* env vars)
from sabio_client import SabioClient  # noqa: E402
sabio = SabioClient()
log(f"El Sabio → {sabio.base_url}  modelo={sabio.model}")

# --- OBSERVABILITY SETUP (Phase 4) ---
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://rag_phoenix:4318/v1/traces")
resource = Resource(attributes={"service.name": "rag-mcp-server"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer("rag.mcp")
otlp_exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrumentar requests para capturar llamadas a servicios automáticamente
RequestsInstrumentor().instrument()

# --- 0. OBSERVABILITY NOTE ---
# Las llamadas al LLM principal (El Sabio) están instrumentadas dentro de SabioClient.
# Las llamadas a los microservicios internos (Bibliotecario, Critic, Embedding) se
# instrumentan en sus propias funciones de llamada.


# --- NEW EMBEDDING SERVER INTERACTION (Phase 2) ---
async def get_embedding_from_server(texts: List[str], timeout=30) -> List[List[float]]:
    """
    Obtiene embeddings de texto desde el servidor de embeddings dedicado.
    Espera una lista de textos y devuelve una lista de listas de floats.
    """
    with tracer.start_as_current_span("get_embedding_from_embedding_server") as span:
        span.set_attribute("embedding.server.host", EMBEDDING_SERVER_HOST)
        span.set_attribute("text.count", len(texts))
        span.set_attribute("text.lengths", [len(t) for t in texts])
        try:
            url = f"{EMBEDDING_SERVER_HOST}/embed"
            payload = {"texts": texts}
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=payload, timeout=timeout)
                res.raise_for_status()
                embeddings_data = res.json()
            if "embeddings" in embeddings_data and isinstance(
                embeddings_data["embeddings"], list
            ):
                return embeddings_data["embeddings"]
            else:
                raise ValueError(
                    f"Invalid embeddings response from server: {embeddings_data}"
                )
        except httpx.HTTPStatusError as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error calling embedding server: {e}")
            raise
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Unexpected error in get_embedding_from_server: {e}")
            raise


# --- NEW BIBLIOTECARIO SERVER INTERACTION (Phase 2) ---
async def call_bibliotecario_server(history: str, query: str, timeout=30) -> Dict[str, Any]:
    """
    Llama al servidor del Bibliotecario para resumir el historial e identificar entidades.
    """
    with tracer.start_as_current_span("call_bibliotecario_server") as span:
        span.set_attribute("bibliotecario.server.host", BIBLIOTECARIO_SERVER_HOST)
        span.set_attribute("history.length", len(history))
        span.set_attribute("query.length", len(query))
        try:
            url = f"{BIBLIOTECARIO_SERVER_HOST}/summarize_and_identify/"
            payload = {"history": history, "query": query}
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=payload, timeout=timeout)
                res.raise_for_status()
                response_data = res.json()
            if "summary" in response_data and "entities" in response_data:
                span.set_attribute(
                    "bibliotecario.summary.length", len(response_data["summary"])
                )
                span.set_attribute(
                    "bibliotecario.entities.count", len(response_data["entities"])
                )
                return response_data
            else:
                raise ValueError(
                    f"Invalid response from bibliotecario server: {response_data}"
                )
        except httpx.HTTPStatusError as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error calling bibliotecario server: {e}")
            raise
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Unexpected error in call_bibliotecario_server: {e}")
            raise


async def generate_search_variations(query: str, timeout=30) -> List[str]:
    """
    Genera variaciones de la query para expansión, utilizando el servidor del Bibliotecario.
    """
    with tracer.start_as_current_span("generate_search_variations") as span:
        span.set_attribute("query.original", query)
        prompt = (
            f"Eres un experto en búsqueda de código. Genera 3 variaciones de búsqueda "
            f"técnica breves y pertinentes para la siguiente consulta: '{query}'. "
            f"Responde solo las 3 frases separadas por una nueva línea, sin enumerar ni añadir prefijos."
        )
        try:
            # Aunque bibliotecario es para resumen, podemos usarlo para generación de texto
            # encuadrando el prompt de forma adecuada.
            # El endpoint /summarize_and_identify/ espera 'history' y 'query'.
            # Usaremos un historial vacío y el prompt como la query.
            bibliotecario_response = await call_bibliotecario_server(
                history="", query=prompt, timeout=timeout
            )
            # La respuesta estará en el campo 'summary' para este tipo de prompt.
            raw_response = bibliotecario_response.get("summary", "").strip()
            lines = raw_response.split("\n")  # Split by actual newline character
            variations = [l.strip().strip("- ") for l in lines if l.strip()]
            span.set_attribute("variations.count", len(variations))
            span.set_attribute("variations.list", json.dumps(variations))
            return variations[:3]  # Devolver hasta 3 variaciones
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error generando variaciones de búsqueda con el Bibliotecario: {e}")
            return []


async def call_critic_server(
    query: str, documents: List[Dict[str, Any]], top_n: int = 3, timeout=30
) -> List[Dict[str, Any]]:
    """
    Llama al servidor del Crítico Local para rerankear documentos.
    """
    with tracer.start_as_current_span("call_critic_server_for_rerank") as span:
        span.set_attribute("critic.server.host", CRITIC_SERVER_HOST)
        span.set_attribute("rerank.query", query)
        span.set_attribute("rerank.documents_count", len(documents))
        span.set_attribute("rerank.top_n", top_n)
        try:
            url = f"{CRITIC_SERVER_HOST}/rerank/"
            payload = {"query": query, "documents": documents, "top_n": top_n}
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=payload, timeout=timeout)
                res.raise_for_status()
                response_data = res.json()
            if "reranked_documents" in response_data and isinstance(
                response_data["reranked_documents"], list
            ):
                span.set_attribute(
                    "rerank.reranked_count", len(response_data["reranked_documents"])
                )
                return response_data["reranked_documents"]
            else:
                raise ValueError(
                    f"Invalid rerank response from critic server: {response_data}"
                )
        except httpx.HTTPStatusError as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error calling critic server for reranking: {e}")
            raise
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Unexpected error in call_critic_server: {e}")
            raise





def get_compressed_memory(history: List[types.ChatMessage]) -> str:
    """
    Refactorizado para usar el servidor del Bibliotecario.
    Resume el historial de chat e identifica entidades clave.
    """
    with tracer.start_as_current_span("get_compressed_memory") as span:
        full_history_text = "\n".join([f"{m.role}: {m.content}" for m in history])
        last_query = history[-1].content if history else ""

        try:
            bibliotecario_response = call_bibliotecario_server(
                full_history_text, last_query
            )
            summary = bibliotecario_response["summary"]
            entities = bibliotecario_response["entities"]

            span.set_attribute("compressed_memory.summary_length", len(summary))
            span.set_attribute("compressed_memory.entities", json.dumps(entities))

            # Puedes formatear la salida como consideres mejor para el LLM remoto
            formatted_output = f"Resumen del historial de conversación:\n{summary}\nEntidades clave identificadas: {', '.join(entities)}"
            return formatted_output
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error en get_compressed_memory al llamar al Bibliotecario: {e}")
            return "No se pudo generar un resumen del historial."


def get_query_vector(query: str, timeout=30) -> List[float]:
    """
    Obtiene el vector de embedding para una consulta, utilizando el servidor de embeddings dedicado.
    Implementa 'Query Expansion' usando el Bibliotecario y devuelve un promedio de los embeddings.
    """
    with tracer.start_as_current_span("get_query_vector") as span:
        span.set_attribute("query.original", query)
        all_queries_to_embed = [query]
        try:
            # Generar variaciones de la query (Query Expansion)
            variations = generate_search_variations(query, timeout=timeout)
            if variations:
                all_queries_to_embed.extend(variations)
            span.set_attribute(
                "query.expanded_queries_count", len(all_queries_to_embed)
            )
            span.set_attribute(
                "query.expanded_queries", json.dumps(all_queries_to_embed)
            )

            # Obtener embeddings para todas las queries
            all_embeddings = get_embedding_from_server(
                texts=all_queries_to_embed, timeout=timeout
            )

            if not all_embeddings:
                raise ValueError("No embeddings returned for the queries.")

            # Calcular el promedio de los embeddings
            # Asegurarse de que todos los embeddings tienen el mismo tamaño
            if not all(len(e) == len(all_embeddings[0]) for e in all_embeddings):
                log(
                    "Advertencia: Embeddings con tamaños inconsistentes. Usando solo el primero."
                )
                span.set_attribute("error", True)
                span.set_attribute(
                    "error.message", "Embeddings con tamaños inconsistentes."
                )
                return all_embeddings[0]

            avg_embedding = [
                sum(dim_values) / len(dim_values) for dim_values in zip(*all_embeddings)
            ]

            span.set_attribute("query.embedding_size", len(avg_embedding))
            return avg_embedding

        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error obteniendo vector de query con expansión: {e}")
            raise


def get_project_structure(root_path: Path) -> str:
    """
    Escáner de sistema de archivos para entregar al LLM remoto un mapa del árbol de directorios antes de la búsqueda.
    """
    with tracer.start_as_current_span("get_project_structure") as span:
        tree = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Ignorar directorios no deseados
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
                and d
                not in [
                    "node_modules",
                    "data",
                    "venv",
                    "dist",
                    "build",
                    "out",
                    "gen",
                    "test_results",
                ]
            ]

            level = dirpath.replace(str(root_path), "").count(os.sep)
            indent = " " * 4 * (level)
            tree.append(f"{indent}{os.path.basename(dirpath)}/")
            subindent = " " * 4 * (level + 1)
            for f in filenames:
                if not f.startswith("."):  # Ignorar archivos ocultos
                    tree.append(f"{subindent}{f}")

        result = "\\n".join(tree)
        span.set_attribute("project.structure.length", len(result))
        return result





async def search_code(
    query: str, limit: int = 5, use_reranker: bool = False
) -> List[Dict[str, Any]]:
    """
    Busca fragmentos de código relevantes en Qdrant.
    Incorpora Query Expansion y Reranking en fases posteriores.
    """
    with tracer.start_as_current_span("search_code") as span:
        span.set_attribute("search.query", query)
        span.set_attribute("search.limit", limit)
        span.set_attribute("search.use_reranker", use_reranker)

        try:
            query_vector = get_query_vector(query)

            # Obtener más resultados si se va a rerankear
            qdrant_limit = limit * (
                3 if use_reranker else 1
            )  # Aumentamos el límite para el reranker
            span.set_attribute("search.qdrant_limit", qdrant_limit)

            search_result = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=qdrant_limit,
                with_payload=True,
            )

            raw_results = []
            for hit in search_result:
                raw_results.append(
                    {
                        "content": hit.payload.get("content"),
                        "path": hit.payload.get("path"),
                        "extension": hit.payload.get("extension"),
                        "score": hit.score,
                        "imports": hit.payload.get("imports"),
                        "signatures": hit.payload.get("signatures"),
                    }
                )

            span.set_attribute("search.qdrant_hits_initial", len(raw_results))

            if use_reranker and raw_results:
                log(
                    f"Realizando reranking de {len(raw_results)} resultados con el Crítico Local..."
                )
                try:
                    reranked_docs = call_critic_server(query, raw_results, top_n=limit)
                    span.set_attribute("search.reranked_hits_count", len(reranked_docs))
                    return reranked_docs
                except Exception as rerank_e:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", f"Reranking failed: {rerank_e}")
                    log(
                        f"Error durante el reranking: {rerank_e}. Volviendo a los resultados brutos."
                    )
                    return raw_results[:limit]  # Fallback a resultados brutos

            return raw_results[
                :limit
            ]  # Devolver solo el límite si no hay reranker o si falla

        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error en search_code: {e}")
            return []


# --- HERRAMIENTAS MCP ---
TOOLS = [
    types.Tool(
        id="search_code",
        name="search_code",
        description="Busca fragmentos de código relevantes de la base de código. Útil para encontrar ejemplos de implementación, definiciones o lógica relacionada con una consulta.",
        parameters=types.Parameters(
            properties={
                "query": types.Parameter(
                    type="string",
                    description="La consulta de búsqueda para encontrar fragmentos de código.",
                ),
                "limit": types.Parameter(
                    type="integer",
                    description="Número máximo de fragmentos de código a devolver (por defecto 5).",
                ),
            },
            required=["query"],
        ),
    ),
    types.Tool(
        id="get_project_structure",
        name="get_project_structure",
        description="Obtiene el mapa del árbol de directorios del proyecto. Útil para entender la organización del código antes de realizar búsquedas más específicas.",
        parameters=types.Parameters(
            properties={},
            required=[],
        ),
    ),
]

# --- MANEJADORES MCP ---


async def handle_list_tools(
    request: types.ListToolsRequest, context: types.UserContext
) -> types.ListToolsResponse:
    with tracer.start_as_current_span("handle_list_tools") as span:
        log(f"ListTools Request: {request.json()}")
        return types.ListToolsResponse(tools=TOOLS)


async def handle_call_tool(
    request: types.CallToolRequest, context: types.UserContext
) -> types.CallToolResponse:
    with tracer.start_as_current_span("handle_call_tool") as span:
        span.set_attribute("tool.id", request.tool_id)
        span.set_attribute("tool.parameters", json.dumps(request.parameters))
        log(f"CallTool Request: {request.json()}")

        tool_id = request.tool_id
        parameters = request.parameters

        # Aquí se implementaría la lógica del Crítico Local para evaluar el contexto
        # antes de ejecutar la herramienta, si fuera necesario.
        # Por ahora, las herramientas se ejecutan directamente.

        if tool_id == "search_code":
            query = parameters.get("query")
            limit = parameters.get("limit", 5)
            if not query:
                return types.CallToolResponse(
                    error="Parámetro 'query' es requerido para search_code."
                )
            results = await search_code(query, limit)
            return types.CallToolResponse(output=json.dumps(results))

        elif tool_id == "get_project_structure":
            project_structure = get_project_structure(Path(PROJECT_ROOT))
            return types.CallToolResponse(output=project_structure)

        else:
            span.set_attribute("error", True)
            span.set_attribute("error.message", f"Herramienta no encontrada: {tool_id}")
            return types.CallToolResponse(error=f"Herramienta no encontrada: {tool_id}")


async def handle_sse(request: Request) -> Response:
    with tracer.start_as_current_span("handle_sse_connection"):
        return await sse.handle_request(request)


# El servidor MCP es un servidor SSE. Para que el agente pueda comunicarse en ambos sentidos, necesitamos
# capturar las peticiones POST y pasárselas al mcp_server.
# Sin embargo, Starlette también tiene que interceptar las peticiones GET para servir el cliente SSE.
# Esto se hace envolviendo el "receive" original con un "new_receive".
async def logged_sse_app(scope, receive, send):
    async def new_receive():
        message = await receive()
        log(f"Received from client: {message.get('type')}")
        return message

    await mcp_server.handle_request(scope, new_receive, send)


# --- SERVIDOR HTTP ---


routes = [
    Route("/", logged_sse_app, methods=["GET", "POST", "PUT", "DELETE"]),
    Mount("/messages", app=sse.app, name="messages"),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
]

app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    # Registrar los manejadores de herramientas en el servidor MCP
    mcp_server.on_list_tools_request(handle_list_tools)
    mcp_server.on_call_tool_request(handle_call_tool)

    # El punto de entrada del LLM es el endpoint de chat de nuestro MCP.
    # Cuando se recibe un mensaje de chat, el MCP lo pasa al LLM remoto.
    @mcp_server.on_chat_completion_request
    async def on_chat_completion_request(
        request: types.ChatCompletionRequest, context: types.UserContext
    ) -> types.ChatCompletionResponse:
        with tracer.start_as_current_span("on_chat_completion_request") as span:
            span.set_attribute("chat.request.model", request.model)
            span.set_attribute("chat.request.messages_count", len(request.messages))
            log(f"ChatCompletion Request: {request.json()}")

            # 1. Obtener memoria comprimida usando El Bibliotecario
            compressed_memory = get_compressed_memory(
                request.messages[:-1]
            )  # Excluir el último mensaje (la query actual)
            current_query = request.messages[-1].content

            span.set_attribute("chat.compressed_memory", compressed_memory)
            span.set_attribute("chat.current_query", current_query)

            # 2. Generar el prompt final para el LLM remoto
            # Este prompt debe incluir el contexto relevante, herramientas, etc.
            system_prompt = f"""
Eres un asistente de codificación experto. Tu tarea es ayudar al usuario a entender y modificar el código.
Utiliza las herramientas disponibles para buscar información relevante en la base de código.
Considera el historial de la conversación y el resumen proporcionado para entender el contexto.

Historial de conversación resumido:
{compressed_memory}

Estructura del proyecto (si es relevante, puedes obtenerla con get_project_structure):
{get_project_structure(Path(PROJECT_ROOT))}

Herramientas disponibles:
{json.dumps([t.dict() for t in TOOLS])}

Responde de forma concisa y útil. Si se te pide generar código, asegúrate de que sea correcto y relevante.
"""

            # Construir mensajes para el LLM remoto
            # El sistema debe ser el primer mensaje. Los mensajes de usuario/asistente deben seguir.
            llm_messages = [{"role": "system", "content": system_prompt}]
            # Añadir mensajes del historial, excluyendo el último (ya manejado como current_query)
            llm_messages.extend(
                [{"role": m.role, "content": m.content} for m in request.messages[:-1]]
            )
            # Añadir la query actual
            llm_messages.append({"role": "user", "content": current_query})

            extra_params = {}
            if request.temperature is not None:
                extra_params["temperature"] = request.temperature
            if request.max_tokens is not None:
                extra_params["max_tokens"] = request.max_tokens

            try:
                # Usar El Sabio para la respuesta final al usuario
                llm_response_content = sabio.chat(
                    messages=llm_messages,
                    timeout=120,
                    **extra_params,
                )

                span.set_attribute(
                    "chat.llm_response_content_length", len(llm_response_content)
                )

                return types.ChatCompletionResponse(
                    id="chatcmpl-" + str(uuid.uuid4()),
                    choices=[
                        types.Choice(
                            delta=types.ChatMessage(
                                role="assistant", content=llm_response_content
                            ),
                            finish_reason="stop",
                        )
                    ],
                    model=sabio.model,
                    object="chat.completion",
                    created=int(time.time()),
                )

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                log(f"Error al llamar al LLM remoto: {e}")
                return types.ChatCompletionResponse(
                    id="error",
                    choices=[
                        types.Choice(
                            delta=types.ChatMessage(
                                role="assistant",
                                content=f"Error interno al procesar tu solicitud: {e}",
                            ),
                            finish_reason="stop",
                        )
                    ],
                )

    # Iniciar el servidor Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
