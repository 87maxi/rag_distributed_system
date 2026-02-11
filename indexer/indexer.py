import hashlib
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

# --- SUPRESIÓN DE LOGS DE LIBRERÍAS ---
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)

# OpenTelemetry
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


# 1. UTILIDADES Y LOGS
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


task_queue = queue.Queue()
# Diccionario en memoria para evitar re-indexar archivos idénticos
file_hashes = {}

# 2. CONFIGURACIÓN
QDRANT_HOST = os.getenv("QDRANT_HOST", "rag_qdrant")
# 4b. LLM CALL WRAPPER & CONFIG
LLM_HOST = os.getenv("OLLAMA_HOST", "http://rag_ollama:11434")
LLM_API_TYPE = os.getenv("LLM_API_TYPE", "ollama")

def call_llm_generic(endpoint_type: str, model: str, payload: dict, timeout=30):
    """Generic wrapper for LLM calls (Ollama/OpenAI)."""
    if LLM_API_TYPE == "ollama":
        return _call_ollama(endpoint_type, model, payload, timeout)
    elif LLM_API_TYPE == "openai":
        return _call_openai_compatible(endpoint_type, model, payload, timeout)
    return _call_ollama(endpoint_type, model, payload, timeout) # Fallback

def _call_ollama(endpoint_type, model, payload, timeout):
    suffix = "/api/generate" if endpoint_type == "generate" else "/api/embed"
    final_payload = payload.copy()
    final_payload["model"] = model
    if endpoint_type == "generate": final_payload["stream"] = False
    return _execute_http_call(suffix, final_payload, timeout, "ollama")

def _call_openai_compatible(endpoint_type, model, payload, timeout):
    suffix = "/v1/chat/completions" if endpoint_type == "generate" else "/v1/embeddings"
    final_payload = {}
    if endpoint_type == "generate":
        final_payload = {
            "model": model,
            "messages": [{"role": "user", "content": payload.get("prompt","")}],
            "temperature": 0.0
        }
    else:
        final_payload = {"model": model, "input": payload.get("input", "")}

    res_data = _execute_http_call(suffix, final_payload, timeout, "openai")
    
    # Normalize result
    normalized = {}
    if endpoint_type == "generate":
        try:
            normalized["response"] = res_data["choices"][0]["message"]["content"]
        except: normalized["response"] = ""
    else:
        try:
            data = res_data["data"][0]["embedding"]
            normalized["embeddings"] = [data]
            normalized["embedding"] = data
        except: normalized["embedding"] = None
    return normalized

def _execute_http_call(endpoint_suffix, payload, timeout, system):
    base_host = LLM_HOST.rstrip("/")
    url = f"{base_host}{endpoint_suffix}"
    
    with tracer.start_as_current_span(f"{system}_call") as span:
        span.set_attribute("llm.system", system)
        try:
            res = requests.post(url, json=payload, timeout=timeout)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            span.set_attribute("error", True)
            raise e

# 4. CONFIGURACIÓN DE SPLITTERS POR LENGUAJE
SPLITTERS = {
    ".ts": RecursiveCharacterTextSplitter.from_language(
        language=Language.TS, chunk_size=1200, chunk_overlap=200
    ),
    ".tsx": RecursiveCharacterTextSplitter.from_language(
        language=Language.TS, chunk_size=1200, chunk_overlap=200
    ),
    ".py": RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1200, chunk_overlap=200
    ),
    ".sol": RecursiveCharacterTextSplitter.from_language(
        language=Language.SOL, chunk_size=1500, chunk_overlap=300
    ),
    ".rs": RecursiveCharacterTextSplitter.from_language(
        language=Language.RUST, chunk_size=1200, chunk_overlap=200
    ),
    ".js": RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=1200, chunk_overlap=200
    ),
}
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def get_file_hash(path: Path):
    """Genera un hash SHA256 para detectar cambios en el contenido."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


@tracer.start_as_current_span("process_file_task")
def process_file_task(file_path: Path):
    """Procesa un archivo individual: Hash -> Split -> Embed -> Index."""
    span = trace.get_current_span()
    span.set_attribute("file.path", str(file_path))
    # Ignore node_modules, hidden files/dirs, data directory, and venv
    path_str = str(file_path)
    if (
        "node_modules" in path_str
        or "/." in path_str
        or "/data/" in path_str
        or "/venv/" in path_str
        or "/dist/" in path_str
        or "/build/" in path_str
        or "/out/" in path_str
        or "/.git/" in path_str
        or "/.github/" in path_str
        or "/.vscode/" in path_str
        or "/.next/" in path_str
        or "/.cache/" in path_str
        or "/broadcast/" in path_str
        or "/forge-std/" in path_str
        or "/openzeppelin-contracts/" in path_str
    ):
        return

    try:
        if not file_path.exists() or (
            file_path.suffix not in SPLITTERS and file_path.suffix != ".js"
        ):
            return

        # Verificar si el archivo ha cambiado realmente
        current_hash = get_file_hash(file_path)
        if not current_hash or file_hashes.get(str(file_path)) == current_hash:
            return

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        if not content:
            return

        # Seleccionar el splitter adecuado
        splitter = SPLITTERS.get(file_path.suffix, DEFAULT_SPLITTER)
        chunks = splitter.split_text(content)

        points = []
        for chunk in chunks:
            # Llamada a LLM para generar el embedding usando el wrapper
            try:
                data = call_llm_generic(
                    "embed",
                    MODEL_NAME,
                    {"input": chunk},
                    timeout=30
                )
                vector = data.get("embeddings", [None])[0] or data.get("embedding")
            except Exception as e:
                log(f"Error embedding chunk: {e}")
                vector = None

            if vector:
                # Extraer metadatos heurísticos
                metro_imports = []
                metro_funcs = []

                # Heurística simple según extensión
                if file_path.suffix == ".py":
                    import re

                    metro_imports = re.findall(
                        r"^(?:from|import) .*", chunk, re.MULTILINE
                    )
                    metro_funcs = re.findall(r"^def .*", chunk, re.MULTILINE)
                elif file_path.suffix in [".ts", ".tsx", ".js"]:
                    import re

                    metro_imports = re.findall(r"^import .*", chunk, re.MULTILINE)
                    metro_funcs = re.findall(
                        r"(?:function|const|let|var) \w+\s*=?\s*\(", chunk
                    )

                payload = {
                    "path": str(file_path.relative_to(INDEX_PATH)),
                    "content": chunk,
                    "extension": file_path.suffix,
                    "imports": metro_imports,
                    "signatures": metro_funcs,
                }

                span.set_attribute("file.payload", str(payload))

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload,
                    )
                )

        if points:
            # Eliminar vectores antiguos del archivo (usando la nueva sintaxis de Filter)
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="path",
                            match=MatchValue(
                                value=str(file_path.relative_to(INDEX_PATH))
                            ),
                        )
                    ]
                ),
            )

            # Upsert de nuevos puntos
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            file_hashes[str(file_path)] = current_hash
            log(f"✅ Indexado: {file_path} ({len(points)} chunks)")

    except Exception as e:
        log(f"❌ Error en {file_path}: {e}")


# 5. SISTEMA DE COLAS (Worker)
def worker():
    log("👷 Trabajador de cola listo.")
    while True:
        file_path = task_queue.get()
        if file_path is None:
            break
        process_file_task(file_path)
        task_queue.task_done()


# 6. OBSERVADOR DE ARCHIVOS (Watchdog)
class CodeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            task_queue.put(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            task_queue.put(Path(event.src_path))


# 7. EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Inicializar Colección con optimización para GPU (Slot 0)
    try:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
                on_disk=False,  # Mantener en RAM para que la GPU procese rápido
            ),
            hnsw_config=HnswConfigDiff(
                on_disk=False  # Acelera la construcción del gráfico de búsqueda en GPU
            ),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20000,
            ),
        )
        log(f"🚀 Colección '{COLLECTION_NAME}' creada con aceleración HNSW-GPU.")
    except Exception:
        log(f"📦 Usando colección existente.")

    # Iniciar hilo de procesamiento
    threading.Thread(target=worker, daemon=True).start()

    # Escaneo inicial del directorio
    log(f"📂 Escaneando {INDEX_PATH}...")
    valid_suffixes = {".py", ".ts", ".tsx", ".sol", ".rs", ".js"}
    for f in INDEX_PATH.rglob("*"):
        if f.is_file() and f.suffix in valid_suffixes:
            path_str = str(f)
            if (
                "node_modules" not in path_str
                and not f.name.startswith(".")
                and "/data/" not in path_str
                and "/venv/" not in path_str
            ):
                task_queue.put(f)

    # Iniciar observación de cambios en tiempo real
    observer = Observer()
    observer.schedule(CodeHandler(), str(INDEX_PATH), recursive=True)
    observer.start()

    log("👁️  Esperando cambios en archivos...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
