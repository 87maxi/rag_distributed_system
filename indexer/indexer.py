import os
import sys
import uuid
import time
import queue
import threading
import logging
import hashlib
from pathlib import Path
import requests

# --- SUPRESIÓN DE LOGS DE LIBRERÍAS ---
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    HnswConfigDiff, 
    OptimizersConfigDiff,
    Filter,
    FieldCondition,
    MatchValue
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 1. UTILIDADES Y LOGS
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

task_queue = queue.Queue()
# Diccionario en memoria para evitar re-indexar archivos idénticos
file_hashes = {}

# 2. CONFIGURACIÓN
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
INDEX_PATH = Path(os.getenv("INDEX_PATH", "/app/code"))
COLLECTION_NAME = "code_base_test2"
MODEL_NAME = "nomic-embed-text:latest"

# 3. CLIENTE QDRANT
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)

# 4. CONFIGURACIÓN DE SPLITTERS POR LENGUAJE
SPLITTERS = {
    ".ts": RecursiveCharacterTextSplitter.from_language(language=Language.TS, chunk_size=1200, chunk_overlap=200),
    ".tsx": RecursiveCharacterTextSplitter.from_language(language=Language.TS, chunk_size=1200, chunk_overlap=200),
    ".py": RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1200, chunk_overlap=200),
    ".sol": RecursiveCharacterTextSplitter.from_language(language=Language.SOL, chunk_size=1500, chunk_overlap=300),
    ".rs": RecursiveCharacterTextSplitter.from_language(language=Language.RUST, chunk_size=1200, chunk_overlap=200),
    ".js": RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=1200, chunk_overlap=200),
}
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_file_hash(path: Path):
    """Genera un hash SHA256 para detectar cambios en el contenido."""
    hasher = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def process_file_task(file_path: Path):
    """Procesa un archivo individual: Hash -> Split -> Embed -> Index."""
    if "node_modules" in str(file_path) or "/." in str(file_path):
        return

    try:
        if not file_path.exists() or (file_path.suffix not in SPLITTERS and file_path.suffix != '.js'):
            return
        
        # Verificar si el archivo ha cambiado realmente
        current_hash = get_file_hash(file_path)
        if not current_hash or file_hashes.get(str(file_path)) == current_hash:
            return

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        
        if not content:
            return

        # Seleccionar el splitter adecuado
        splitter = SPLITTERS.get(file_path.suffix, DEFAULT_SPLITTER)
        chunks = splitter.split_text(content)
        
        points = []
        for chunk in chunks:
            # Llamada a Ollama para generar el embedding
            res = requests.post(
                f"{OLLAMA_HOST}/api/embed", 
                json={"model": MODEL_NAME, "input": chunk}, 
                timeout=30
            )
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
            # Eliminar vectores antiguos del archivo (usando la nueva sintaxis de Filter)
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="path", 
                            match=MatchValue(value=str(file_path.relative_to(INDEX_PATH)))
                        )
                    ]
                )
            )
            
            # Upsert de nuevos puntos
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            file_hashes[str(file_path)] = current_hash
            log(f"✅ Indexado: {file_path.name} ({len(points)} chunks)")
            
    except Exception as e:
        log(f"❌ Error en {file_path.name}: {e}")

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
                on_disk=False  # Mantener en RAM para que la GPU procese rápido
            ),
            hnsw_config=HnswConfigDiff(
                on_disk=False  # Acelera la construcción del gráfico de búsqueda en GPU
            ),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20000,
            )
        )
        log(f"🚀 Colección '{COLLECTION_NAME}' creada con aceleración HNSW-GPU.")
    except Exception:
        log(f"📦 Usando colección existente.")

    # Iniciar hilo de procesamiento
    threading.Thread(target=worker, daemon=True).start()

    # Escaneo inicial del directorio
    log(f"📂 Escaneando {INDEX_PATH}...")
    valid_suffixes = {'.py', '.ts', '.tsx', '.sol', '.rs', '.js'}
    for f in INDEX_PATH.rglob("*"):
        if f.is_file() and f.suffix in valid_suffixes:
            if "node_modules" not in str(f) and not f.name.startswith('.'):
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