import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

# Check for GPU availability and use it if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the TinyLlama model for the 'Bibliotecario'
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = None
model = None
text_generator = None

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,  # Use GPU 0 if available, else CPU
        torch_dtype=torch.bfloat16,
    )
    print("TinyLlama model loaded successfully for Bibliotecario.")
except Exception as e:
    print(f"Error loading TinyLlama model for Bibliotecario: {e}")


class BibliotecarioRequest(BaseModel):
    history: str  # Represents the conversation history to summarize
    query: str  # The current query


class SummaryResponse(BaseModel):
    summary: str
    entities: list[str]


@app.post("/summarize_and_identify/")
async def summarize_and_identify(request: BibliotecarioRequest):
    if text_generator is None:
        raise HTTPException(status_code=500, detail="Bibliotecario model not loaded.")

    full_context = (
        f"Conversation History: {request.history}\nUser Query: {request.query}"
    )

    # Prompt for summarization and entity identification
    prompt_template = f"""
    Eres un bibliotecario experto. Tu tarea es resumir la siguiente conversación e identificar las entidades clave (como clases, funciones, archivos, módulos) mencionadas.

    Contexto:
    {full_context}

    Instrucciones:
    1. Proporciona un resumen conciso del contexto.
    2. Lista las entidades clave encontradas.

    Resumen:
    """

    try:
        # Generate text using the loaded model
        generation_result = text_generator(
            prompt_template,
            max_new_tokens=256,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            truncation=True,
        )

        generated_text = generation_result[0]["generated_text"]

        # Extract summary and entities from the generated text
        # This part might need fine-tuning based on actual model output
        summary_start_tag = "Resumen:\n"
        entities_start_tag = "\nEntidades Clave:"

        summary_text = ""
        entities_list = []

        if summary_start_tag in generated_text:
            after_summary_tag = generated_text.split(summary_start_tag, 1)[1]
            if entities_start_tag in after_summary_tag:
                summary_text = after_summary_tag.split(entities_start_tag, 1)[0].strip()
                entities_raw = after_summary_tag.split(entities_start_tag, 1)[1].strip()
                entities_list = [
                    e.strip() for e in entities_raw.split(",") if e.strip()
                ]
            else:
                summary_text = after_summary_tag.strip()

        return SummaryResponse(summary=summary_text, entities=entities_list)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during summarization and entity identification: {str(e)}",
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": text_generator is not None,
        "device": device,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
