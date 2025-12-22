#!/usr/bin/env python3
"""
Indexador Inteligente RAG - Optimizado para Solidity, TS, JS, Rust, Python, YAML, JSON, Markdown y Shell
Con chunking inteligente por lenguaje para evitar errores de contexto largo.
"""
import os
import sys
import time
import hashlib
import re
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import redis
from typing import List


class SmartIndexer(FileSystemEventHandler):
    def __init__(self, code_path, qdrant_host="qdrant", qdrant_port=6333,
                 ollama_host="ollama-embeddings", ollama_port=11434, redis_host="redis"):
        
        self.code_path = Path(code_path)
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.redis = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/embeddings"
        
        # Extensiones soportadas
        self.extensions = {
            '.sol', '.ts', '.tsx', '.js', '.jsx',  # Frontend y Contracts
            '.json', '.yaml', '.yml',              # Configs y ABIs
            '.md', '.py', '.sh'                    # Docs y Scripts
        }
        
        # Directorios y archivos a ignorar
        self.ignore_dirs = {
            'node_modules', 'target', 'build', 'dist', '.next',
            '.git', 'artifacts', 'cache', 'typechain-types',
            '__pycache__', 'venv', '.venv', 'coverage', 'public',
            'broadcast', 'out', 'deployments',
            'test', 'tests', '__tests__' 
        }
        self.ignore_files = {
            'package-lock.json', 'yarn.lock', 'Cargo.lock',
            'pnpm-lock.yaml', 'go.sum', 'go.mod'
        }

        # 🔒 Límites seguros para nomic-embed-text (8k tokens)
        self.MAX_FILE_SIZE_BYTES = 500_000    # 500 KB
        self.MAX_CONTENT_CHARS = 24_000       # 24k chars ≈ 6k tokens
        
        self.stats = {"indexed": 0, "failed": 0, "skipped": 0}
        self._ensure_collection()
        
        print(f"🚀 Indexador RAG iniciado en: {code_path}")
        print(f"📁 Extensiones: {', '.join(sorted(self.extensions))}")
        print(f"🔗 Ollama: {self.ollama_url} | 📡 Qdrant: {qdrant_host}:{qdrant_port}")
        print(f"✂️ Límites: {self.MAX_FILE_SIZE_BYTES//1000} KB | {self.MAX_CONTENT_CHARS} chars")

    def _ensure_collection(self):
        """Crea colección en Qdrant si no existe"""
        try:
            collections = {c.name for c in self.qdrant.get_collections().collections}
            if "codebase" not in collections:
                self.qdrant.create_collection(
                    collection_name="codebase",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
                print("✅ Colección 'codebase' creada en Qdrant.")
        except Exception as e:
            print(f"❌ Error al inicializar Qdrant: {e}")

    def get_embedding(self, text):
        """Genera embedding con nomic-embed-text"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "nomic-embed-text",
                    "prompt": "search_document: " + text
                },
                timeout=45
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"❌ Error en Ollama: {e}")
            return None

    def _smart_chunk_code(self, content: str, file_ext: str) -> List[str]:
        """Divide código en chunks respetando estructura lógica por tipo de archivo"""
        if len(content) <= self.MAX_CONTENT_CHARS:
            return [content]
        
        lines = content.splitlines(keepends=True)
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Patrones específicos por extensión
        if file_ext == '.sol':
            # Solidity: contratos, librerías, funciones
            safe_patterns = [
                r'^\s*(contract|library|interface)\s+\w+',
                r'^\s*function\s+\w+\s*\(',
                r'^\s*}\s*(//.*|/\*.*\*/)?\s*$',
                r'^\s*//\s*-----',
                r'^\s*$'
            ]
        elif file_ext in ['.ts', '.tsx', '.js', '.jsx']:
            # TypeScript/JavaScript: funciones, clases, componentes
            safe_patterns = [
                r'^\s*(export\s+)?(async\s+)?function\s+\w+',
                r'^\s*const\s+\w+\s*=\s*(async\s+)?\([^)]*\)\s*=>',
                r'^\s*(interface|type|class)\s+\w+',
                r'^\s*}\s*(//.*|/\*.*\*/)?\s*$',
                r'^\s*//\s*-----',
                r'^\s*$'
            ]
        elif file_ext == '.py':
            # Python: funciones, clases, decoradores
            safe_patterns = [
                r'^\s*(async\s+)?def\s+\w+\s*\(',
                r'^\s*class\s+\w+',
                r'^\s*@.*',
                r'^\s*}\s*(#.*|/\*.*\*/)?\s*$',
                r'^\s*#.*-----',
                r'^\s*$'
            ]
        elif file_ext == '.rs':
            # Rust: funciones, structs, impls
            safe_patterns = [
                r'^\s*(pub(\s+\w+)?\s+)?(fn|struct|enum|trait|impl)\s+\w+',
                r'^\s*}\s*(//.*|/\*.*\*/)?\s*$',
                r'^\s*//\s*-----',
                r'^\s*$'
            ]
        elif file_ext in ['.yaml', '.yml']:
            # YAML: cortar en raíces de nivel superior
            safe_patterns = [
                r'^[a-zA-Z_][a-zA-Z0-9_]*\s*:',
                r'^\s*-\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:',
                r'^\s*$'
            ]
        elif file_ext == '.json':
            # JSON: cortar en objetos/arrays raíz (cuidado: raro dividir JSON)
            safe_patterns = [
                r'^\s*\}\s*$',
                r'^\s*\]\s*$',
                r'^\s*\{\s*$',
                r'^\s*\[\s*$',
                r'^\s*$'
            ]
        elif file_ext == '.md':
            # Markdown: cortar en encabezados
            safe_patterns = [
                r'^#{1,6}\s+.*',
                r'^\s*$'
            ]
        elif file_ext == '.sh':
            # Shell: cortar en funciones o secciones
            safe_patterns = [
                r'^\s*function\s+\w+\s*\(\s*\)\s*\{',
                r'^\s*\w+\s*\(\s*\)\s*\{',
                r'^\s*#\s*-----',
                r'^\s*$'
            ]
        else:
            # Fallback genérico
            safe_patterns = [
                r'^\s*}\s*(//.*|/\*.*\*/|#.*|\s*)?$',
                r'^\s*//\s*-----',
                r'^\s*#\s*-----',
                r'^\s*$'
            ]
        
        compiled_patterns = [re.compile(p) for p in safe_patterns]
        
        for line in lines:
            if current_size + len(line) > self.MAX_CONTENT_CHARS and current_chunk:
                # Buscar punto seguro en las últimas 100 líneas
                split_at = -1
                search_start = max(0, len(current_chunk) - 100)
                for i in range(len(current_chunk) - 1, search_start, -1):
                    if any(pat.match(current_chunk[i]) for pat in compiled_patterns):
                        split_at = i
                        break
                
                if split_at != -1:
                    chunks.append(''.join(current_chunk[:split_at + 1]))
                    current_chunk = current_chunk[split_at + 1:] + [line]
                    current_size = sum(len(l) for l in current_chunk)
                else:
                    # Forzar corte si no hay punto seguro
                    chunks.append(''.join(current_chunk))
                    current_chunk = [line]
                    current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks

    def index_file(self, filepath):
        """Indexa un archivo con chunking inteligente"""
        if filepath.suffix not in self.extensions:
            return
        
        rel_path = filepath.relative_to(self.code_path)
        if any(part in self.ignore_dirs for part in filepath.parts):
            return
        if filepath.name in self.ignore_files:
            return

        try:
            # Saltar archivos JSON grandes (no deberían dividirse)
            if filepath.suffix == '.json' and filepath.stat().st_size > 10_000:
                print(f"⏭️ Saltando JSON grande: {rel_path}")
                self.stats["skipped"] += 1
                return
                
            if filepath.stat().st_size > self.MAX_FILE_SIZE_BYTES:
                print(f"⏭️ Saltando {rel_path} (>500 KB)")
                self.stats["skipped"] += 1
                return
        except OSError:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                return
        except (UnicodeDecodeError, OSError):
            print(f"⏭️ Binario o ilegible: {rel_path}")
            self.stats["skipped"] += 1
            return

        file_id_base = hashlib.md5(str(filepath).encode()).hexdigest()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if self.redis.get(f"hash:{file_id_base}") == content_hash:
            self.stats["skipped"] += 1
            return

        print(f"📄 Procesando: {rel_path}")
        chunks = self._smart_chunk_code(content, filepath.suffix) if len(content) > self.MAX_CONTENT_CHARS else [content]
        
        all_valid = True
        for i, chunk in enumerate(chunks):
            vector = self.get_embedding(chunk)
            if not vector or len(vector) != 768:
                self.stats["failed"] += 1
                all_valid = False
                break

            point_id = f"{file_id_base}_c{i}" if len(chunks) > 1 else file_id_base
            self.qdrant.upsert(
                collection_name="codebase",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "path": str(filepath),
                            "filename": filepath.name,
                            "extension": filepath.suffix,
                            "content_preview": chunk[:2000],
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "last_updated": time.time()
                        }
                    )
                ]
            )
            self.stats["indexed"] += 1

        if all_valid:
            self.redis.set(f"hash:{file_id_base}", content_hash)
            status = f" ({len(chunks)} chunks)" if len(chunks) > 1 else ""
            print(f"✅ Indexado: {rel_path}{status}")

    def index_directory(self):
        """Escanea recursivamente el directorio"""
        print("🔍 Iniciando indexación completa...")
        for root, dirs, files in os.walk(self.code_path):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            for file in files:
                self.index_file(Path(root) / file)
        print(f"✨ Indexación finalizada: {self.stats['indexed']} indexados, "
              f"{self.stats['skipped']} saltados, {self.stats['failed']} fallidos")

    def on_modified(self, event):
        if not event.is_directory:
            self.index_file(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            self.index_file(Path(event.src_path))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Indexador RAG para múltiples lenguajes")
    parser.add_argument("--path", default=os.getenv("NFS_MOUNT_PATH", "/mnt/codigo_principal"))
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"❌ Directorio no encontrado: {args.path}")
        sys.exit(1)

    indexer = SmartIndexer(
        code_path=args.path,
        qdrant_host=os.getenv("QDRANT_HOST", "qdrant"),
        ollama_host=os.getenv("OLLAMA_HOST", "ollama-embeddings")
    )

    indexer.index_directory()

    observer = Observer()
    observer.schedule(indexer, args.path, recursive=True)
    observer.start()
    print("👀 Monitoreando cambios en tiempo real...")

    try:
        while True:
            time.sleep(1, timeout=10)
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo indexador...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()