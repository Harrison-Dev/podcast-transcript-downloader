"""
Ollama API Client - HTTP client for Ollama LLM service.

Replaces llama-cpp-python with HTTP calls to Ollama API.
Supports retry mechanism and exponential backoff.
"""
import logging
import time
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    HTTP client for Ollama API service.

    Supports generate and chat completions with retry mechanism.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ):
        self.api_url = (api_url or settings.OLLAMA_API).rstrip('/')
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT
        self.max_retries = max_retries or getattr(settings, 'API_MAX_RETRIES', 3)
        self.retry_delay = retry_delay or getattr(settings, 'API_RETRY_DELAY', 2.0)

    def health_check(self) -> bool:
        """Check if Ollama API is available."""
        try:
            response = httpx.get(f"{self.api_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = httpx.get(f"{self.api_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def is_model_available(self, model: Optional[str] = None) -> bool:
        """Check if a specific model is available."""
        model = model or self.model
        models = self.list_models()
        # Check both exact match and prefix match (e.g., qwen3:8b matches qwen3:8b-q4_0)
        return any(m == model or m.startswith(f"{model}-") for m in models)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Call Ollama /api/generate endpoint.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Generation temperature (default from config)
            num_predict: Max tokens to generate (default from config)
            model: Model to use (default from config)

        Returns:
            Generated text, empty string on failure
        """
        url = f"{self.api_url}/api/generate"
        model = model or self.model

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature or getattr(settings, 'OLLAMA_TEMPERATURE', 0.3),
                "num_predict": num_predict or getattr(settings, 'OLLAMA_NUM_PREDICT', 4096),
            }
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    return result.get("response", "")

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"Ollama API error: {e.response.status_code}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(delay)

            except httpx.RequestError as e:
                last_error = e
                logger.error(f"Ollama request failed: {e}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(delay)

        logger.error(f"Ollama generate failed after {self.max_retries} attempts: {last_error}")
        return ""

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Call Ollama /api/chat endpoint for multi-turn conversation.

        Args:
            messages: List of {"role": "system/user/assistant", "content": "..."}
            temperature: Generation temperature
            num_predict: Max tokens to generate
            model: Model to use

        Returns:
            Assistant's response text
        """
        url = f"{self.api_url}/api/chat"
        model = model or self.model

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or getattr(settings, 'OLLAMA_TEMPERATURE', 0.3),
                "num_predict": num_predict or getattr(settings, 'OLLAMA_NUM_PREDICT', 4096),
            }
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    return result.get("message", {}).get("content", "")

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(delay)

        logger.error(f"Ollama chat failed after {self.max_retries} attempts: {last_error}")
        return ""


# Global client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get the global Ollama client instance (singleton)."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def check_ollama_available() -> bool:
    """Check if Ollama is available and model is ready."""
    client = get_ollama_client()
    if not client.health_check():
        return False
    return client.is_model_available()
