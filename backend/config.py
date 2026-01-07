"""
Podcast Transcript Downloader - Configuration
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Whisper Configuration
    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: Literal["cuda", "cpu"] = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"  # float16 for GPU, int8 for lower memory
    CONVERT_TRADITIONAL: bool = True  # Convert Simplified to Traditional Chinese
    
    # SOTA VAD Parameters (optimized for reducing hallucination and better segmentation)
    VAD_THRESHOLD: float = 0.5  # Speech probability threshold
    VAD_MIN_SPEECH_DURATION_MS: int = 250  # Filter out short noises
    VAD_MIN_SILENCE_DURATION_MS: int = 1500  # Longer silence = better paragraph breaks
    VAD_SPEECH_PAD_MS: int = 400  # Pad speech chunks for better context
    
    # LLM Post-Processing Configuration
    LLM_ENABLED: bool = True
    LLM_PROVIDER: Literal["ollama", "none"] = "ollama"
    LLM_MODEL: str = "qwen3:8b"  # or "mistral", "qwen3:4b"
    LLM_OLLAMA_HOST: str = "http://localhost:11434"
    LLM_BATCH_SIZE: int = 5  # Process N segments at a time
    LLM_TIMEOUT: int = 60  # Request timeout in seconds
    
    # Pipeline Configuration
    MAX_DOWNLOAD_WORKERS: int = 3
    TRANSCRIBE_WORKERS: int = 1  # Usually 1 due to GPU memory constraints
    
    # Paths
    OUTPUT_BASE_DIR: Path = Path("./transcripts")
    TEMP_AUDIO_DIR: Path = Path("./temp_audio")
    
    # Server
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
