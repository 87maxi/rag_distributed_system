#!/usr/bin/env python3
"""
Indexador RAG Automático - Ignora OpenZeppelin problemático, preserva tu código
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

class AutoProjectIndexer(FileSystemEventHandler):
    def __init__(self, code_path):
        self.code_path = Path(code_path)
        self.qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=6333)
        self.redis = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, decode_responses=True)
        self.ollama_url = f"http://{os.getenv('OLLAMA_HOST', 'ollama-embeddings')}:11434/api/embeddings"
        
        # Extensiones relevantes para desarrollo
        self.extensions = {'.sol', '.ts', '.tsx', '.js', '.jsx', '.json', '.yaml', '.yml', '.md', '.py', '.sh'}
        
        # Directorios a ignorar (sin 'utils' para preservar tu código)
        self.ignore_dirs = {
            'node_modules', '.git', 'build', 'dist', '.next', 'out', 'public',
            'artifacts', 'cache', 'broadcast', 'deployments', 'typechain-types',
            'test', 'tests', '__tests__', '__pycache__', 'venv', '.venv', 'coverage', 'logs'
        }
        
        # Límites seguros
        self.MAX_FILE_SIZE_BYTES = 300_000  # 300 KB (más estricto)
        self.stats = {"indexed": 0, "failed": 0, "skipped": 0}

    def get_project_collection(self, filepath):
        """Obtiene colección por proyecto (carpeta raíz)"""
        try:
            rel_path = filepath.relative_to(self.code_path)
            project_name = rel_path.parts[0].lower().replace('-', '_').replace('.', '_')
            return f"project_{project_name}"
        except:
            return "project_default"

    def ensure_collection(self, collection_name):
        """Crea colección si no existe"""
        try:
            collections = [c.name for c in self.qdrant.get_collections().collections]
            if collection_name not in collections:
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                print(f"✅ Colección creada: {collection_name}")
        except Exception as e:
            print(f"❌ Error colección {collection_name}: {e}")

    def get_embedding(self, text):
        """Genera embedding con límite seguro de 20k chars"""
        try:
            # Truncar para evitar errores de contexto
            safe_text = (text[:20000] + "...") if len(text) > 20000 else text
            response = requests.post(
                self.ollama_url,
                json={"model": "nomic-embed-text", "prompt": "search_document: " + safe_text},
                timeout=45
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"❌ Error Ollama: {e}")
            return None

    def index_file(self, filepath):
        """Indexa archivo con reglas específicas para evitar errores"""
        if filepath.suffix not in self.extensions:
            return

        # 👇 IGNORAR SOLO utils de OpenZeppelin (preserva tus utils)
        path_str = str(filepath)
        if 'openzeppelin-contracts/contracts/utils' in path_str:
            self.stats["skipped"] += 1
            return

        # Ignorar directorios generales
        if any(part in self.ignore_dirs for part in filepath.parts):
            return

        try:
            # Reglas de tamaño por tipo de archivo
            file_size = filepath.stat().st_size
            if filepath.suffix == '.json' and file_size > 50_000:  # 50 KB
                self.stats["skipped"] += 1
                return
            if filepath.suffix in ['.js', '.ts', '.tsx', '.sol'] and file_size > 200_000:  # 200 KB
                self.stats["skipped"] += 1
                return
            if file_size > self.MAX_FILE_SIZE_BYTES:  # 300 KB global
                self.stats["skipped"] += 1
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                return

        except (OSError, UnicodeDecodeError):
            self.stats["skipped"] += 1
            return

        try:
            # Obtener colección y verificar caché
            collection_name = self.get_project_collection(filepath)
            self.ensure_collection(collection_name)
            
            file_id = hashlib.md5(str(filepath).encode()).hexdigest()
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            redis_key = f"hash:{collection_name}:{file_id}"
            
            if self.redis.get(redis_key) == content_hash:
                self.stats["skipped"] += 1
                return

            print(f"📄 [{collection_name}] Procesando: {filepath.name}")
            
            # Generar embedding y guardar
            vector = self.get_embedding(content)
            if vector and len(vector) == 768:
                self.qdrant.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(
                        id=file_id,
                        vector=vector,
                        payload={
                            "path": str(filepath),
                            "filename": filepath.name,
                            "project": collection_name,
                            "content_preview": content[:2000]
                        }
                    )]
                )
                self.redis.set(redis_key, content_hash)
                self.stats["indexed"] += 1
            else:
                self.stats["failed"] += 1

        except Exception as e:
            print(f"⚠️ Error procesando {filepath}: {e}")
            self.stats["failed"] += 1

    def index_all(self):
        """Indexa todos los proyectos automáticamente"""
        print(f"🔍 Iniciando indexación en: {self.code_path}")
        for item in self.code_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                print(f"📁 Proyecto: {item.name}")
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
                    for file in files:
                        self.index_file(Path(root) / file)
        print(f"✨ Finalizado. Indexados: {self.stats['indexed']}, Saltados: {self.stats['skipped']}, Fallidos: {self.stats['failed']}")

    def on_modified(self, event):
        if not event.is_directory:
            self.index_file(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            self.index_file(Path(event.src_path))

def main():
    path = os.getenv("NFS_MOUNT_PATH", "/mnt/codigo_principal")
    if not Path(path).exists():
        print(f"❌ Directorio no existe: {path}")
        sys.exit(1)
    
    indexer = AutoProjectIndexer(path)
    indexer.index_all()
    
    observer = Observer()
    observer.schedule(indexer, path, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()