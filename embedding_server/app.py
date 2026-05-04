from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model once when the application starts
try:
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    # Depending on the criticality, you might want to raise the exception or handle it
    # For now, we'll let it proceed, but subsequent calls to /embed might fail.
    model = None

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    try:
        embeddings = model.encode(request.texts, convert_to_numpy=False, convert_to_tensor=True)
        # Convert embeddings to a list of Python lists for JSON serialization
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None, "device": device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
