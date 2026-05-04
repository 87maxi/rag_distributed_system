"""
sabio_client.py — El Sabio: capa de abstracción para cualquier LLM compatible con OpenAI.

El Bibliotecario (pipeline RAG / servidor MCP) SIEMPRE habla con El Sabio a través de
este módulo. Nunca llama a ningún backend LLM directamente.

Configuración via variables de entorno:
    SABIO_BASE_URL  — URL base del LLM (default: http://localhost:11434)
    SABIO_MODEL     — Nombre del modelo (default: qwen2.5:0.5b)
    SABIO_API_KEY   — API key del proveedor (default: none — Ollama/local no la necesita)
    SABIO_TIMEOUT   — Timeout HTTP en segundos (default: 30)

Proveedores soportados (cualquiera que implemente /v1/chat/completions):
    - Ollama        → SABIO_BASE_URL=http://localhost:11434
    - Docker Model Runner (DMR) → SABIO_BASE_URL=http://localhost:12434
    - vLLM          → SABIO_BASE_URL=http://localhost:8080
    - llama.cpp     → SABIO_BASE_URL=http://localhost:8080
    - OpenAI        → SABIO_BASE_URL=https://api.openai.com  SABIO_API_KEY=sk-...
    - Anthropic proxy, Gemini proxy, etc.
"""

import os
import logging
from typing import List, Dict, Any, Optional

import httpx
from opentelemetry import trace

logger = logging.getLogger("rag-mcp.sabio")

# --- Configuración ---
SABIO_BASE_URL = os.getenv("SABIO_BASE_URL", "http://localhost:11434")
SABIO_MODEL    = os.getenv("SABIO_MODEL",    "qwen2.5:0.5b")
SABIO_API_KEY  = os.getenv("SABIO_API_KEY",  "none")
SABIO_TIMEOUT  = int(os.getenv("SABIO_TIMEOUT", "30"))

# Tracer (reutiliza el provider configurado en el proceso principal)
tracer = trace.get_tracer("rag.sabio")


class SabioClient:
    """
    Cliente para 'El Sabio': el servidor LLM intercambiable.

    Habla con cualquier proveedor que exponga la OpenAI Chat Completions API:
        POST {base_url}/v1/chat/completions

    Ejemplo de uso:
        sabio = SabioClient()
        respuesta = sabio.complete("¿Qué hace la función foo()?")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model:    Optional[str] = None,
        api_key:  Optional[str] = None,
        timeout:  Optional[int] = None,
    ):
        self.base_url = (base_url or SABIO_BASE_URL).rstrip("/")
        self.model    = model   or SABIO_MODEL
        self.timeout  = timeout or SABIO_TIMEOUT
        api_key_val   = api_key or SABIO_API_KEY
        self.headers  = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key_val}",
        }
        logger.info(
            f"SabioClient inicializado → base_url={self.base_url}  model={self.model}"
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, str]],
        timeout:  Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Llama a /v1/chat/completions con una lista de mensajes OpenAI-style.

        Args:
            messages: Lista de dicts {"role": "user"|"assistant"|"system", "content": str}
            timeout:  Timeout específico para esta llamada (override del default).
            **kwargs: Parámetros extra pasados al payload (temperature, max_tokens, etc.)

        Returns:
            El contenido del mensaje de respuesta del asistente (str).

        Raises:
            httpx.HTTPStatusError: Si el servidor devuelve un código de error HTTP.
            KeyError: Si la respuesta no tiene el formato esperado.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            **kwargs,
        }

        with tracer.start_as_current_span("sabio.chat") as span:
            span.set_attribute("sabio.base_url",       self.base_url)
            span.set_attribute("sabio.model",          self.model)
            span.set_attribute("sabio.messages_count", len(messages))

            # Loguear el prompt del último mensaje (usuarios) para observabilidad
            if messages:
                last_content = messages[-1].get("content", "")
                span.set_attribute("gen_ai.prompt", str(last_content)[:500])

            try:
                async with httpx.AsyncClient() as client:
                    res = await client.post(
                        url,
                        json=payload,
                        headers=self.headers,
                        timeout=timeout or self.timeout,
                    )
                    res.raise_for_status()
                    data = res.json()

                content = data["choices"][0]["message"]["content"]
                span.set_attribute("gen_ai.response", str(content)[:500])
                logger.debug(f"Sabio response ({len(content)} chars)")
                return content

            except httpx.HTTPStatusError as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                logger.error(f"Error HTTP al llamar a El Sabio en {url}: {e}")
                raise
            except (KeyError, IndexError) as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", f"Respuesta inesperada: {e}")
                logger.error(
                    f"Respuesta de El Sabio no tiene el formato esperado: {e}"
                )
                raise

    def complete(
        self,
        prompt:  str,
        system:  Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Conveniencia: envía un prompt simple (sin historial previo).

        Args:
            prompt:  Texto del mensaje del usuario.
            system:  Mensaje de sistema opcional (instrucciones al modelo).
            timeout: Timeout HTTP override.
            **kwargs: Parámetros extra (temperature, max_tokens, etc.)

        Returns:
            Respuesta del modelo como string.
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, timeout=timeout, **kwargs)

    async def health_check(self) -> bool:
        """
        Verifica que El Sabio está respondiendo (sin cargar un modelo).
        Intenta GET /v1/models que la mayoría de APIs OpenAI-compat. exponen.
        """
        url = f"{self.base_url}/v1/models"
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(url, headers=self.headers, timeout=5)
                is_ok = res.status_code < 500
                logger.info(f"Sabio health check → {self.base_url} → {'OK' if is_ok else 'FAIL'}")
                return is_ok
        except Exception as e:
            logger.warning(f"Sabio no disponible en {self.base_url}: {e}")
            return False
