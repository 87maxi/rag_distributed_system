import json
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

# Check for GPU availability and use it if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the TinyLlama model for the 'Critico Local' (or Qwen 1.5B)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Placeholder, assuming similar size as Qwen 1.5B
tokenizer = None
model = None
text_generator = None

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for modern GPUs, or float16 for older
    ).to(device)
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,  # Use GPU 0 if available, else CPU
        torch_dtype=torch.bfloat16,
    )
    print(f"'{model_name}' model loaded successfully for Critico Local.")
except Exception as e:
    print(f"Error loading '{model_name}' model for Critico Local: {e}")


class RerankRequest(BaseModel):
    query: str
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de documentos a rerankear, cada uno con 'content' y 'path'.",
    )
    top_n: int = Field(3, description="Número de documentos más relevantes a devolver.")


class RerankResponse(BaseModel):
    reranked_documents: List[Dict[str, Any]]


@app.post("/rerank/")
async def rerank_documents(request: RerankRequest):
    if text_generator is None:
        raise HTTPException(status_code=500, detail="Critico Local model not loaded.")

    if not request.documents:
        return RerankResponse(reranked_documents=[])

    snippets_for_llm = []
    # Create a mapping from index to original document for easy retrieval
    doc_map = {i: doc for i, doc in enumerate(request.documents)}

    for i, doc in enumerate(request.documents):
        content_snippet = doc.get("content", "")[:200]  # Take a snippet
        path = doc.get("path", "N/A")
        snippets_for_llm.append(f"[{i}] Path: {path} Content: {content_snippet}...")

    # Prompt the LLM to select the most relevant snippets based on the query
    prompt_template = f"""
Eres un crítico experto en código. Tu tarea es rerankear fragmentos de código.
Dada la consulta del usuario y una lista de fragmentos de código, selecciona los {request.top_n} fragmentos más relevantes.
Responde ÚNICAMENTE con una lista de los índices (números entre corchetes) de los {request.top_n} fragmentos más relevantes, separados por comas.

Consulta: {request.query}

Fragmentos:
{"\n".join(snippets_for_llm)}

Índices de los {request.top_n} fragmentos más relevantes (ej: 0, 2, 4):
"""

    try:
        generation_result = text_generator(
            prompt_template,
            max_new_tokens=60,  # Small output expected
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.1,  # Keep it deterministic for reranking
            truncation=True,
        )

        generated_text = generation_result[0]["generated_text"]
        # Extract the part after the prompt
        response_part = generated_text.split(
            "Índices de los "
            + str(request.top_n)
            + " fragmentos más relevantes (ej: 0, 2, 4):\n"
        )[-1].strip()

        # Parse the indices
        import re

        selected_indices = []
        for match in re.finditer(r"\d+", response_part):
            try:
                index = int(match.group(0))
                if 0 <= index < len(request.documents):
                    selected_indices.append(index)
                if len(selected_indices) >= request.top_n:
                    break
            except ValueError:
                continue

        # Retrieve the original documents based on selected indices
        reranked_docs = [doc_map[idx] for idx in selected_indices]

        # If reranker didn't return enough docs, fill with highest scoring remaining ones
        if len(reranked_docs) < request.top_n:
            existing_indices = set(selected_indices)
            remaining_docs = [
                doc for idx, doc in doc_map.items() if idx not in existing_indices
            ]
            # Assuming documents might have a 'score' from initial retrieval, sort them if available
            # For simplicity, we'll just append the next best if reranker failed to pick enough
            # A more robust solution would re-query Qdrant with reranking score or resort
            reranked_docs.extend(remaining_docs[: request.top_n - len(reranked_docs)])

        return RerankResponse(reranked_documents=reranked_docs)

    except Exception as e:
        print(f"Error during reranking: {e}")
        # Fallback: if reranking fails, return the original documents (or top_n of them)
        return RerankResponse(reranked_documents=request.documents[: request.top_n])


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": text_generator is not None,
        "device": device,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
