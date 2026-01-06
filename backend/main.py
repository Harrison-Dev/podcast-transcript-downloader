"""
Podcast Transcript Downloader - FastAPI Backend

Main application entry point with CORS and routing configuration.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import transcription_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    logger.info("Starting Podcast Transcript Downloader API...")
    logger.info(f"Output directory: {settings.OUTPUT_BASE_DIR.absolute()}")
    logger.info(f"Whisper model: {settings.WHISPER_MODEL} ({settings.WHISPER_DEVICE})")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Podcast Transcript Downloader",
    description="Download podcasts from RSS feeds and transcribe using Whisper AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transcription_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "Podcast Transcript Downloader",
        "version": "1.0.0",
        "status": "running",
        "whisper_model": settings.WHISPER_MODEL,
        "device": settings.WHISPER_DEVICE,
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "config": {
            "whisper_model": settings.WHISPER_MODEL,
            "device": settings.WHISPER_DEVICE,
            "compute_type": settings.WHISPER_COMPUTE_TYPE,
            "output_dir": str(settings.OUTPUT_BASE_DIR.absolute()),
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
