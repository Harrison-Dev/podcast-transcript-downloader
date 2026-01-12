"""
Podcast Transcript Downloader - Configuration

Supports containerized architecture:
- Whisper API: Containerized faster-whisper with GPU support
- Ollama API: External LLM service for text polishing
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # ========== Whisper API Configuration ==========
    # Containerized Whisper service endpoint
    WHISPER_API: str = "http://localhost:8207"  # Docker whisper-api service
    WHISPER_MODEL: str = "large-v3"  # Model size: tiny/base/small/medium/large-v3

    # Legacy settings (still used for local transcriber fallback)
    WHISPER_DEVICE: Literal["cuda", "cpu"] = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"  # float16 for GPU, int8 for lower memory
    CONVERT_TRADITIONAL: bool = True  # Convert Simplified to Traditional Chinese

    # VAD Parameters (used by Whisper API container)
    VAD_THRESHOLD: float = 0.5  # Speech probability threshold
    VAD_MIN_SPEECH_DURATION_MS: int = 250  # Filter out short noises
    VAD_MIN_SILENCE_DURATION_MS: int = 1500  # Longer silence = better paragraph breaks
    VAD_SPEECH_PAD_MS: int = 400  # Pad speech chunks for better context

    # ========== Ollama LLM Configuration ==========
    # Ollama API endpoint
    OLLAMA_API: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"  # Model name (must be pulled in Ollama)
    OLLAMA_TEMPERATURE: float = 0.3  # Lower = more deterministic
    OLLAMA_NUM_PREDICT: int = 4096  # Max tokens per generation

    # LLM Processing toggles
    LLM_ENABLED: bool = True
    LLM_TIMEOUT: int = 300  # Request timeout in seconds (longer for large batches)

    # Two-pass batch processing
    BATCH_SIZE_PASS1: int = 8   # Pass 1: Add punctuation, fix errors
    BATCH_SIZE_PASS2: int = 20  # Pass 2: Polish into fluent article

    # API retry settings
    API_MAX_RETRIES: int = 3
    API_RETRY_DELAY: float = 2.0  # Base delay for exponential backoff

    # ========== Pipeline Configuration ==========
    MAX_DOWNLOAD_WORKERS: int = 3
    TRANSCRIBE_WORKERS: int = 1  # Usually 1 due to GPU memory constraints

    # ========== Paths ==========
    OUTPUT_BASE_DIR: Path = Path("./transcripts")
    TEMP_AUDIO_DIR: Path = Path("./temp_audio")

    # ========== Server ==========
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
settings.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
