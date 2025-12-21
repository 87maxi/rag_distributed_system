#!/usr/bin/env python3
"""
Cliente RAG distribuido - Usa recursos locales + LLM remoto
"""
import requests
import json
import hashlib
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import redis
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RAGConfig:
    """Configuración del sistema RAG distribuido"""
    # Servicios locales (Cliente)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    ollama_host: str = "localhost"
    ollama_port: int = 11435
    redis_host: str = "localhost"
    
    # LLM remoto (Principal)
    llm_host: str = "192.168.1.50"
    llm_port: int = 8080
    
    # Parámetros RAG
    top_k: int = 5
    similarity_threshold: float = 0.3
    max_context_tokens: int = 4000

class DistributedRAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        
        # Conectar a servicios locales
        self.qdrant = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            timeout=10
        )
        
        self.redis = redis.Redis(
            host=self.config.redis_host,
            port=6379,
            decode_responses=True,
            socket_timeout=5
        )
        
        # URLs
        self.ollama_url = f"http://{self.config.ollama_host}:{self.config.ollama_port}/api/embed"
        self.llm_url = f"http://{self.config.llm_host}:{self.config.llm_port}/completion"
        
        # Estadísticas
        self.stats = {
            "queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "avg_response_time": 0
        }
        
        print("🚀 Sistema RAG Distribuido Iniciado")
        print(f"   • Cliente: Qdrant + Embeddings (local)")
        print(f"   • Principal: LLM en {self.config.llm_host}:{self.config.llm_port}")
    
    def ask(self, question: str, language: str = None) -> Dict:
        """Pregunta principal con RAG"""
        import time
        start_time = time.time()
        self.stats["queries"] += 1
        
        # 1. Verificar cache
        cache_key = self._get_cache_key(question, language)
        cached = self.redis.get(cache_key)
        
        if cached:
            self.stats["cache_hits"] += 1
            response_time = time.time() - start_time
            self._update_stats(response_time)
            
            result = json.loads(cached)
            result["cached"] = True
            result["response_time"] = response_time
            return result
        
        # 2. Generar embedding local
        question_embedding = self._generate_embedding(question)
        if not question_embedding:
            return {
                "error": "No se pudo generar embedding",
                "answer": "",
                "sources": []
            }
        
        # 3. Buscar en Qdrant local
        context_results = self._search_context(
            question_embedding, 
            language_filter=language
        )
        
        # 4. Construir prompt optimizado
        prompt = self._build_rag_prompt(question, context_results)
        
        # 5. LLM remoto (Principal)
        llm_response = self._call_llm(prompt)
        self.stats["llm_calls"] += 1
        
        # 6. Preparar respuesta
        result = {
            "question": question,
            "answer": llm_response,
            "sources": context_results,
            "context_count": len(context_results),
            "cached": False,
            "response_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # 7. Cachear respuesta (1 hora)
        self.redis.setex(
            cache_key,
            3600,
            json.dumps({k: v for k, v in result.items() if k != 'response_time'})
        )
        
        self._update_stats(result["response_time"])
        
        return result
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generar embedding usando Ollama local"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "nomic-embed-text",
                    "prompt": text[:4000],
                    "options": {"num_gpu": 15}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
        except Exception as e:
            print(f"Error en embedding: {e}")
        
        return None
    
    def _search_context(self, embedding: List[float], language_filter: str = None) -> List[Dict]:
        """Buscar contexto relevante en Qdrant"""
        from qdrant_client.models import Filter
        
        # Construir filtro si hay lenguaje específico
        search_filter = None
        if language_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="extension",
                        match=MatchValue(value=f".{language_filter}")
                    )
                ]
            )
        
        try:
            results = self.qdrant.search(
                collection_name="codebase",
                query_vector=embedding,
                query_filter=search_filter,
                limit=self.config.top_k,
                with_payload=True,
                with_vectors=False
            )
            
            formatted_results = []
            for hit in results:
                if hit.score >= self.config.similarity_threshold:
                    formatted_results.append({
                        "score": float(hit.score),
                        "filename": hit.payload.get("filename", ""),
                        "path": hit.payload.get("path", ""),
                        "preview": hit.payload.get("content_preview", ""),
                        "extension": hit.payload.get("extension", "")
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error buscando en Qdrant: {e}")
            return []
    
    def _build_rag_prompt(self, question: str, context_results: List[Dict]) -> str:
        """Construir prompt optimizado para el LLM"""
        if not context_results:
            return f"Pregunta: {question}\n\nResponde:"
        
        # Agrupar por tipo de archivo
        context_by_type = {}
        for result in context_results:
            ext = result["extension"]
            if ext not in context_by_type:
                context_by_type[ext] = []
            context_by_type[ext].append(result)
        
        # Construir prompt estructurado
        prompt_parts = ["Tienes acceso al siguiente código relevante:"]
        
        for ext, files in context_by_type.items():
            prompt_parts.append(f"\n[{ext.upper()} FILES]:")
            for i, file in enumerate(files[:3], 1):  # Máximo 3 por tipo
                prompt_parts.append(f"\n--- File {i}: {file['filename']} (score: {file['score']:.3f}) ---")
                prompt_parts.append(file['preview'])
        
        prompt_parts.append(f"\n\nPREGUNTA: {question}")
        prompt_parts.append("\nINSTRUCCIONES:")
        prompt_parts.append("1. Responde basándote PRINCIPALMENTE en el código proporcionado")
        prompt_parts.append("2. Si el código no contiene información suficiente, indícalo claramente")
        prompt_parts.append("3. Sé preciso y técnico en tu respuesta")
        prompt_parts.append("4. Incluye referencias a los archivos relevantes cuando sea posible")
        prompt_parts.append("\nRESPUESTA:")
        
        return "\n".join(prompt_parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Llamar al LLM en la máquina principal"""
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "prompt": prompt,
                    "n_predict": 1024,
                    "temperature": 0.1,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["</s>", "```", "---", "PREGUNTA:"],
                    "stream": False
                },
                timeout=45
            )
            
            if response.status_code == 200:
                return response.json().get("content", "").strip()
            else:
                return f"Error del LLM: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "El LLM tardó demasiado en responder. Intenta con una pregunta más específica."
        except Exception as e:
            return f"Error conectando con el LLM: {str(e)}"
    
    def _get_cache_key(self, question: str, language: str) -> str:
        """Generar clave de cache única"""
        base = f"rag:{question}:{language or 'all'}"
        return hashlib.md5(base.encode()).hexdigest()
    
    def _update_stats(self, response_time: float):
        """Actualizar estadísticas"""
        total_time = self.stats["avg_response_time"] * (self.stats["queries"] - 1)
        self.stats["avg_response_time"] = (total_time + response_time) / self.stats["queries"]
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del sistema"""
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["queries"], 1),
            "services": {
                "qdrant": self._check_service("qdrant", self.config.qdrant_host, self.config.qdrant_port),
                "ollama": self._check_service("ollama", self.config.ollama_host, self.config.ollama_port),
                "llm": self._check_service("llm", self.config.llm_host, self.config.llm_port),
                "redis": self._check_service("redis", self.config.redis_host, 6379)
            }
        }
    
    def _check_service(self, name: str, host: str, port: int) -> bool:
        """Verificar si un servicio está disponible"""
        import socket
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return True
        except:
            return False

# ==================== INTERFAZ DE LÍNEA DE COMANDOS ====================

if __name__ == "__main__":
    import argparse
    import sys
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    console = Console()
    
    parser = argparse.ArgumentParser(description="Cliente RAG Distribuido")
    parser.add_argument("--question", "-q", help="Pregunta a hacer")
    parser.add_argument("--language", "-l", help="Filtrar por lenguaje (py, js, rs, etc)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interactivo")
    parser.add_argument("--stats", "-s", action="store_true", help="Mostrar estadísticas")
    parser.add_argument("--config", "-c", help="Archivo de configuración JSON")
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = RAGConfig()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            console.print(f"[red]Error cargando configuración: {e}[/red]")
    
    # Crear cliente RAG
    rag = DistributedRAG(config)
    
    if args.stats:
        # Mostrar estadísticas
        stats = rag.get_stats()
        
        table = Table(title="📊 Estadísticas del Sistema RAG")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")
        
        table.add_row("Consultas totales", str(stats["queries"]))
        table.add_row("Cache hits", str(stats["cache_hits"]))
        table.add_row("LLM calls", str(stats["llm_calls"]))
        table.add_row("Tiempo promedio", f"{stats['avg_response_time']:.2f}s")
        table.add_row("Cache hit rate", f"{stats['cache_hit_rate']*100:.1f}%")
        
        console.print(table)
        
        # Estado de servicios
        console.print("\n[bold]🔧 Estado de servicios:[/bold]")
        for service, status in stats["services"].items():
            icon = "✅" if status else "❌"
            console.print(f"  {icon} {service}")
        
        sys.exit(0)
    
    if args.interactive:
        # Modo interactivo
        console.print(Panel.fit("💬 [bold cyan]RAG Distribuido - Modo Interactivo[/bold cyan]"))
        console.print("Escribe 'quit' para salir, 'stats' para estadísticas, 'clear' para limpiar")
        
        while True:
            try:
                question = console.input("\n[bold yellow]❓ Pregunta: [/bold yellow]")
                
                if question.lower() == 'quit':
                    break
                elif question.lower() == 'stats':
                    stats = rag.get_stats()
                    console.print(f"\n📊 Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
                    continue
                elif question.lower() == 'clear':
                    console.clear()
                    continue
                
                if not question.strip():
                    continue
                
                # Detectar lenguaje en la pregunta
                language = None
                question_lower = question.lower()
                if "python" in question_lower or ".py" in question_lower:
                    language = "py"
                elif "javascript" in question_lower or ".js" in question_lower:
                    language = "js"
                elif "rust" in question_lower or ".rs" in question_lower:
                    language = "rs"
                elif "solidity" in question_lower or ".sol" in question_lower:
                    language = "sol"
                elif "typescript" in question_lower or ".ts" in question_lower:
                    language = "ts"
                
                with console.status("[bold green]Procesando...[/bold green]"):
                    result = rag.ask(question, language)
                
                # Mostrar respuesta
                console.print("\n" + "="*60)
                
                if "error" in result:
                    console.print(f"[red]Error: {result['error']}[/red]")
                else:
                    # Fuentes
                    if result["sources"]:
                        console.print("[bold cyan]📚 Fuentes encontradas:[/bold cyan]")
                        for source in result["sources"][:3]:
                            console.print(f"  • {source['filename']} (score: {source['score']:.3f})")
                        console.print()
                    
                    # Respuesta
                    md = Markdown(result["answer"])
                    console.print(md)
                    
                    # Metadata
                    console.print(f"\n[dim]⏱️  {result['response_time']:.2f}s • 📝 {result['context_count']} contextos • {'💾 cached' if result['cached'] else '🔄 live'}[/dim]")
                
                console.print("="*60)
                
            except KeyboardInterrupt:
                console.print("\n\n👋 Saliendo...")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    elif args.question:
        # Modo single query
        result = rag.ask(args.question, args.language)
        
        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            sys.exit(1)
        
        console.print(result["answer"])
        
        if args.language:
            console.print(f"\n[dim](Filtrado por: {args.language})[/dim]")
    
    else:
        parser.print_help()