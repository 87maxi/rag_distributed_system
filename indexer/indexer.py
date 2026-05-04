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

# OpenTelemetry — inicializado antes de cualquier decorador @tracer.start_as_current_span
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 1. CONFIGURACIÓN
QDRANT_HOST = os.getenv("QDRANT_HOST", "rag_qdrant")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "code_chunks")
INDEX_PATH = Path(os.getenv("INDEX_PATH", "/app/code"))
EMBEDDING_SERVER_HOST = os.getenv("EMBEDDING_SERVER_HOST", "http://rag_embedding_server:8000")

# 2. OBSERVABILIDAD
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://rag_phoenix:4318/v1/traces")
resource = Resource(attributes={"service.name": "rag-indexer"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer("rag.indexer")
otlp_exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

RequestsInstrumentor().instrument()

# Imports que dependen de que OTel ya esté inicializado
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
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


# 3. UTILIDADES Y LOGS
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


task_queue = queue.Queue()
# Diccionario en memoria para evitar re-indexar archivos idénticos
file_hashes = {}

# Clientes
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)


# 4. EMBEDDING VIA EMBEDDING SERVER
def get_embedding(texts: list[str], timeout: int = 30) -> list[list[float]]:
    """Obtiene embeddings del servidor dedicado de embeddings."""
    url = f"{EMBEDDING_SERVER_HOST.rstrip('/')}/embed"
    with tracer.start_as_current_span("get_embedding") as span:
        span.set_attribute("embedding.server.host", EMBEDDING_SERVER_HOST)
        span.set_attribute("text.count", len(texts))
        try:
            res = requests.post(url, json={"texts": texts}, timeout=timeout)
            res.raise_for_status()
            data = res.json()
            if "embeddings" in data and isinstance(data["embeddings"], list):
                return data["embeddings"]
            raise ValueError(f"Respuesta inesperada del embedding server: {data}")
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            log(f"Error llamando al embedding server: {e}")
            raise


# 5. CONFIGURACIÓN DE SPLITTERS POR LENGUAJE
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


IGNORED_PATHS = {
    "node_modules", "/.","  /data/", "/venv/", "/dist/", "/build/",
    "/out/", "/.git/", "/.github/", "/.vscode/", "/.next/", "/.cache/",
    "/broadcast/", "/forge-std/", "/openzeppelin-contracts/",
}


def _is_path_ignored(path_str: str) -> bool:
    return (
        "node_modules" in path_str
        or "/.." in path_str
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
    )


@tracer.start_as_current_span("process_file_task")
def process_file_task(file_path: Path):
    """Procesa un archivo individual: Hash → Split → Embed → Index."""
    span = trace.get_current_span()
    span.set_attribute("file.path", str(file_path))

    if _is_path_ignored(str(file_path)):
        return

    try:
        if not file_path.exists() or file_path.suffix not in SPLITTERS:
            return

        # Verificar si el archivo ha cambiado realmente
        current_hash = get_file_hash(file_path)
        if not current_hash or file_hashes.get(str(file_path)) == current_hash:
            return

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        if not content:
            return

        splitter = SPLITTERS.get(file_path.suffix, DEFAULT_SPLITTER)
        chunks = splitter.split_text(content)

        if not chunks:
            return

        # Generar embeddings para todos los chunks en un solo request
        try:
            embeddings = get_embedding(chunks)
        except Exception as e:
            log(f"Error generando embeddings para {file_path}: {e}")
            return

        points = []
        for chunk, vector in zip(chunks, embeddings):
            if not vector:
                continue

            # Extraer metadatos heurísticos
            import re

            metro_imports: list[str] = []
            metro_funcs: list[str] = []

            if file_path.suffix == ".py":
                metro_imports = re.findall(r"^(?:from|import) .*", chunk, re.MULTILINE)
                metro_funcs = re.findall(r"^def .*", chunk, re.MULTILINE)
            elif file_path.suffix in [".ts", ".tsx", ".js"]:
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

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        if points:
            # Eliminar vectores antiguos del archivo antes de insertar nuevos
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

            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            file_hashes[str(file_path)] = current_hash
            log(f"✅ Indexado: {file_path} ({len(points)} chunks)")

    except Exception as e:
        log(f"❌ Error en {file_path}: {e}")


# 6. SISTEMA DE COLAS (Worker)
def worker():
    log("👷 Trabajador de cola listo.")
    while True:
        file_path = task_queue.get()
        if file_path is None:
            break
        process_file_task(file_path)
        task_queue.task_done()


# 7. OBSERVADOR DE ARCHIVOS (Watchdog)
class CodeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            task_queue.put(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            task_queue.put(Path(event.src_path))


# 8. EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Inicializar Colección con optimización para GPU
    try:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
                on_disk=False,  # Mantener en RAM para mayor rendimiento
            ),
            hnsw_config=HnswConfigDiff(
                on_disk=False  # Acelera la construcción del índice HNSW
            ),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20000,
            ),
        )
        log(f"🚀 Colección '{COLLECTION_NAME}' creada con aceleración HNSW-GPU.")
    except Exception:
        log("📦 Usando colección existente.")

    # Iniciar hilo de procesamiento
    threading.Thread(target=worker, daemon=True).start()

    # Escaneo inicial del directorio
    log(f"📂 Escaneando {INDEX_PATH}...")
    valid_suffixes = set(SPLITTERS.keys())
    for f in INDEX_PATH.rglob("*"):
        if f.is_file() and f.suffix in valid_suffixes and not _is_path_ignored(str(f)):
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
