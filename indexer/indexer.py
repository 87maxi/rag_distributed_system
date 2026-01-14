import os
import sys
import uuid
import time
import queue
import threading
import logging
from pathlib import Path
import requests

# --- SUPRESIÓN DE LOGS DE LIBRERÍAS ---
# Silenciamos OpenTelemetry y urllib3 para que no ensucien la consola con JSON o errores de conexión
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# 1. UTILIDADES Y LOGS
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

task_queue = queue.Queue()

# 2. CONFIGURACIÓN
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
INDEX_PATH = Path(os.getenv("INDEX_PATH", "/app/code"))
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://rag_phoenix:4318/v1/traces")
COLLECTION_NAME = "code_base"
MODEL_NAME = "nomic-embed-text:latest"

# 3. TELEMETRÍA (Modo Silencioso)
resource = Resource(attributes={"service.name": "rag-indexer"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer("indexer.worker")

try:
    # Verificación rápida de Phoenix
    requests.get(PHOENIX_ENDPOINT.replace("/v1/traces", ""), timeout=0.5)
    exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))
    log("📡 Phoenix detectado. Telemetría activa.")
except:
    # Si falla, no añadimos ningún procesador (NoOp automático)
    log("⚠️ Phoenix no disponible. Telemetría desactivada (Modo Silencioso).")

# 4. CONFIGURACIÓN DE PROCESAMIENTO
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)

SPLITTERS = {
    ".ts": RecursiveCharacterTextSplitter.from_language(language=Language.TS, chunk_size=1200, chunk_overlap=200),
    ".tsx": RecursiveCharacterTextSplitter.from_language(language=Language.TS, chunk_size=1200, chunk_overlap=200),
    ".py": RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1200, chunk_overlap=200),
    ".sol": RecursiveCharacterTextSplitter.from_language(language=Language.SOL, chunk_size=1500, chunk_overlap=300),
    ".rs": RecursiveCharacterTextSplitter.from_language(language=Language.RUST, chunk_size=1200, chunk_overlap=200),
}
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def process_file_task(file_path: Path):
    """Procesa un archivo individualmente."""
    # FILTRO CRÍTICO: Ignorar node_modules y archivos ocultos
    if "node_modules" in str(file_path) or "/." in str(file_path):
        return

    with tracer.start_as_current_span("process_file") as span:
        span.set_attribute("file.path", str(file_path))
        try:
            if not file_path.exists() or file_path.suffix not in {'.py', '.ts', '.tsx', '.sol', '.rs', '.js'}:
                return
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            if not content: return

            splitter = SPLITTERS.get(file_path.suffix, DEFAULT_SPLITTER)
            chunks = splitter.split_text(content)
            
            points = []
            for chunk in chunks:
                res = requests.post(f"{OLLAMA_HOST}/api/embed", 
                                 json={"model": MODEL_NAME, "input": chunk}, timeout=15)
                data = res.json()
                vector = data.get("embeddings", [None])[0] or data.get("embedding")
                
                if vector:
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "path": str(file_path.relative_to(INDEX_PATH)),
                            "content": chunk,
                            "extension": file_path.suffix
                        }
                    ))
            
            if points:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                log(f"✅ Indexado: {file_path.name} ({len(points)} chunks)")
        
        except Exception as e:
            log(f"❌ Error en {file_path.name}: {e}")

# 5. SISTEMA DE COLAS (Worker)
def worker():
    log("👷 Trabajador de cola listo.")
    while True:
        file_path = task_queue.get()
        if file_path is None: break
        process_file_task(file_path)
        task_queue.task_done()

# 6. OBSERVADOR DE ARCHIVOS
class CodeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            task_queue.put(Path(event.src_path))
    def on_created(self, event):
        if not event.is_directory:
            task_queue.put(Path(event.src_path))

# 7. MAIN
if __name__ == "__main__":
    # Inicializar Colección con optimización para hardware acelerado
    try:
        from qdrant_client.models import OptimizersConfigDiff

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768, 
                distance=Distance.COSINE,
                # Habilitamos el almacenamiento en memoria para acceso rápido vía GPU
                on_disk=False 
            ),
            # Optimizamos la configuración para permitir que los procesos de indexación
            # aprovechen mejor los recursos del sistema
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20000, # Evita usar disco hasta que la colección sea grande
            )
        )
        log(f"📦 Colección '{COLLECTION_NAME}' creada con optimización de hardware.")
    except Exception as e:
        log(f"📦 Usando colección existente.")

    # Iniciar Worker
    threading.Thread(target=worker, daemon=True).start()

    # Escaneo inicial con exclusión de node_modules
    log(f"📂 Escaneando {INDEX_PATH}...")
    valid_suffixes = {'.py', '.ts', '.tsx', '.sol', '.rs', '.js'}
    for f in INDEX_PATH.rglob("*"):
        if f.is_file() and f.suffix in valid_suffixes:
            if "node_modules" not in str(f) and not f.name.startswith('.'):
                task_queue.put(f)

    # Iniciar Watchdog
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