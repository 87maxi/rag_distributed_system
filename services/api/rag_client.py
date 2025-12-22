import os
import time
import requests
from dataclasses import dataclass
from typing import Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


@dataclass
class RAGConfig:
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    ollama_host: str = "ollama-embeddings"
    ollama_port: int = 11434
    llm_host: str = "192.168.0.50"
    llm_port: int = 8080
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))


class DistributedRAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.qdrant_client = QdrantAdapter(
            host=config.qdrant_host,
            port=config.qdrant_port
        )
        self.ollama_adapter = OllamaAdapter(
            host=config.ollama_host,
            port=config.ollama_port
        )
        self.llm_adapter = LLMAdapter(
            host=config.llm_host,
            port=config.llm_port
        )

    def ask(self, query: str) -> Dict[str, Any]:
        # 1. Obtener embedding de la consulta
        query_embedding = self.ollama_adapter.get_embedding(query, is_query=True)
        
        # 2. Buscar en Qdrant
        search_results = self.qdrant_client.search(
            collection_name="codebase",
            query_vector=query_embedding,
            limit=15  # Buscar más para filtrar después
        )
        
        # 3. Limitar contexto a MAX_CONTEXT_TOKENS
        context = self._build_context(search_results, query)
        
        # 4. Construir prompt
        prompt = self._build_prompt(query, context)
        
        # 5. Llamar al LLM
        answer = self.llm_adapter.generate(prompt)
        
        return {
            "answer": answer,
            "context_used": len(context["chunks"]),
            "total_tokens_est": context["total_tokens"],
            "llm_host": f"{self.config.llm_host}:{self.config.llm_port}"
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimación conservadora: 1 token ≈ 4 caracteres (para código)"""
        return max(1, len(text) // 4)

    def _build_context(self, search_results: List[PointStruct], query: str) -> Dict[str, Any]:
        """Construye un contexto que no exceda MAX_CONTEXT_TOKENS"""
        max_tokens = self.config.max_context_tokens
        query_tokens = self._estimate_tokens(query)
        available_tokens = max_tokens - query_tokens - 500  # margen para instrucciones y respuesta
        
        chunks = []
        total_tokens = 0
        
        for point in search_results:
            content = point.payload.get("content_preview", "")
            if not content.strip():
                continue
                
            content_tokens = self._estimate_tokens(content)
            if total_tokens + content_tokens > available_tokens:
                break
                
            chunks.append(content)
            total_tokens += content_tokens
            
        return {
            "chunks": chunks,
            "total_tokens": total_tokens,
            "max_allowed": max_tokens
        }

    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Construye un prompt optimizado para código"""
        context_text = "\n\n".join(context["chunks"])
        return f"""Eres un experto en Solidity, TypeScript, Rust y Next.js. 
Responde la pregunta usando SOLO la información del contexto proporcionado.

Contexto:
{context_text}

Pregunta:
{query}

Instrucciones:
- Sé conciso y preciso.
- Si el contexto no contiene la respuesta, di "No tengo suficiente información".
- Para código, incluye ejemplos completos y seguros.

Respuesta:"""

    def get_stats(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "qdrant": f"{self.config.qdrant_host}:{self.config.qdrant_port}",
            "ollama": f"{self.config.ollama_host}:{self.config.ollama_port}",
            "llm": f"{self.config.llm_host}:{self.config.llm_port}",
            "max_context_tokens": self.config.max_context_tokens
        }


# === ADAPTADORES ===

class QdrantAdapter:
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)
    
    def search(self, collection_name: str, query_vector: List[float], limit: int = 5):
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )


class OllamaAdapter:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}/api/embeddings"
    
    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        prefix = "search_query: " if is_query else "search_document: "
        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": "nomic-embed-text",
                    "prompt": prefix + text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Error en Ollama embeddings: {e}")


class LLMAdapter:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}/v1/chat/completions"
    
    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": "qwen-14b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Error en LLM: {e}")