# ./services/api/main.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os

# Importamos tu clase DistributedRAG desde la carpeta de scripts
sys.path.append("/app/scripts")
from rag_client import DistributedRAG, RAGConfig

app = FastAPI(title="RAG Distributed API Gateway")  # ← ESTA LÍNEA ES CLAVE

# Configuración desde variables de entorno
config = RAGConfig(
    qdrant_host=os.getenv("QDRANT_HOST", "qdrant"),
    ollama_host=os.getenv("OLLAMA_HOST", "ollama-embeddings"),
    llm_host=os.getenv("LLAMA_PRINCIPAL_IP", "192.168.0.50"),
    llm_port=int(os.getenv("LLAMA_PRINCIPAL_PORT", 8080))
)

rag_system = DistributedRAG(config)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, x_api_key: Optional[str] = Header(None)):
    # Verificación simple de API Key
    if os.getenv("API_KEY") and x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Extraemos la última pregunta del usuario
    last_message = request.messages[-1].content
    
    # Ejecutamos tu lógica de RAG
    result = rag_system.ask(last_message)
    
    # Formateamos la respuesta
    return {
        "id": "chatcmpl-rag",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["answer"]
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

@app.get("/health")
async def health():
    return rag_system.get_stats()