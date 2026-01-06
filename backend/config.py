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
