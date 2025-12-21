#!/usr/bin/env python3
"""
Indexador Inteligente RAG - Optimizado para Solidity, TS y Next.js
Corre en el cliente (RTX 1650), procesa código y guarda en Qdrant.
"""
import os
import sys
import time
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import redis
import json

class SmartIndexer(FileSystemEventHandler):
    def __init__(self, code_path, qdrant_host="qdrant", qdrant_port=6333,
                 ollama_host="ollama-embeddings", ollama_port=11434, redis_host="redis"):
        
        self.code_path = Path(code_path)
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.redis = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        # ✅ Endpoint CORRECTO para embeddings en Ollama
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/embeddings"
        
        # Extensiones optimizadas para tu stack
        self.extensions = {
            '.sol', '.ts', '.tsx', '.js', '.jsx',  # Frontend y Contracts
            '.json', '.yaml', '.yml',              # Configs y ABIs
            '.md', '.py', '.sh'                    # Docs y Scripts
        }
        
        # Directorios a ignorar (vital para no saturar el contexto)
        self.ignore_dirs = {
            'node_modules', '.next', 'build', 'dist', 
            '.git', 'artifacts', 'cache', 'typechain-types'
        }
        
        self.stats = {"indexed": 0, "failed": 0, "skipped": 0}
        self._ensure_collection()
        
        print(f"🚀 Indexador iniciado en: {code_path}")
        print(f"📁 Extensiones: {', '.join(self.extensions)}")
        print(f"🔗 Ollama URL: {self.ollama_url}")
        print(f"📡 Qdrant: {qdrant_host}:{qdrant_port}")

    def _ensure_collection(self):
        """Crea la colección en Qdrant si no existe (768 dims para Nomic)"""
        try:
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == "codebase" for c in collections)
            if not exists:
                self.qdrant.create_collection(
                    collection_name="codebase",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
                print("✅ Colección 'codebase' creada.")
        except Exception as e:
            print(f"❌ Error al conectar con Qdrant: {e}")

    def get_embedding(self, text, is_query=False):
        """Obtiene embedding de Nomic con prefijos recomendados"""
        # Nomic requiere 'search_document: ' para indexar y 'search_query: ' para buscar
        prefix = "search_query: " if is_query else "search_document: "
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "nomic-embed-text",
                    "prompt": prefix + text  # ✅ "prompt", no "input"
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            # ✅ Clave CORRECTA: "embedding" (singular)
            return data["embedding"]
        except Exception as e:
            print(f"❌ Error en Ollama al procesar texto: {str(e)}")
            print(f"   Texto de ejemplo: {prefix + text[:50]}...")
            if 'response' in locals():
                print(f"   Respuesta de Ollama: {response.status_code} {response.text}")
            return None

    def index_file(self, filepath):
        """Procesa e indexa un solo archivo"""
        if filepath.suffix not in self.extensions:
            return
        
        # Ignorar si está en carpetas prohibidas
        if any(part in self.ignore_dirs for part in filepath.parts):
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return

            # Generar ID único basado en la ruta
            file_id = hashlib.md5(str(filepath).encode()).hexdigest()
            
            # Hash del contenido para ver si cambió (usando Redis)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            cached_hash = self.redis.get(f"hash:{file_id}")
            
            if cached_hash == content_hash:
                self.stats["skipped"] += 1
                return

            print(f"📄 Indexando: {filepath.relative_to(self.code_path)}")
            
            # Obtener embedding
            vector = self.get_embedding(content)
            if vector is None:
                self.stats["failed"] += 1
                return

            # Validar dimensión del vector (debería ser 768)
            if len(vector) != 768:
                print(f"⚠️ Dimensión inesperada del embedding: {len(vector)} (esperado: 768)")
                self.stats["failed"] += 1
                return

            self.qdrant.upsert(
                collection_name="codebase",
                points=[
                    PointStruct(
                        id=file_id,
                        vector=vector,
                        payload={
                            "path": str(filepath),
                            "filename": filepath.name,
                            "extension": filepath.suffix,
                            "content_preview": content[:4000],
                            "last_updated": time.time()
                        }
                    )
                ]
            )
            self.redis.set(f"hash:{file_id}", content_hash)
            self.stats["indexed"] += 1
            print(f"✅ Insertado: {filepath.name}")

        except Exception as e:
            print(f"⚠️ Error procesando {filepath}: {e}")
            self.stats["failed"] += 1

    def index_directory(self):
        """Escaneo completo inicial"""
        print("🔍 Escaneando directorio completo...")
        for root, dirs, files in os.walk(self.code_path):
            # Modificar dirs in-place para que os.walk no entre en carpetas ignoradas
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            for file in files:
                self.index_file(Path(root) / file)
        print(f"✨ Fin del escaneo. Indexados: {self.stats['indexed']}, Saltados: {self.stats['skipped']}, Fallidos: {self.stats['failed']}")

    # Handlers para cambios en tiempo real
    def on_modified(self, event):
        if not event.is_directory:
            self.index_file(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            self.index_file(Path(event.src_path))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=os.getenv("NFS_MOUNT_PATH", "/mnt/codigo_principal"))
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"❌ El directorio no existe: {args.path}")
        sys.exit(1)

    indexer = SmartIndexer(
        code_path=args.path,
        qdrant_host=os.getenv("QDRANT_HOST", "qdrant"),
        ollama_host=os.getenv("OLLAMA_HOST", "ollama-embeddings")
    )

    # Ejecución inicial
    indexer.index_directory()

    # Mantener monitoreo activo
    observer = Observer()
    observer.schedule(indexer, args.path, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo indexador...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()