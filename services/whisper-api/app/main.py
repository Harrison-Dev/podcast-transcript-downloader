"""FastAPI web server for transcription service."""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from .models import (
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionStatus,
    OutputFormat,
    ModelSize,
)
from .transcriber import (
    transcribe_file,
    preload_model,
    get_loaded_models,
    unload_model,
)
from .utils import (
    is_audio_file,
    is_video_file,
    ensure_dir,
    cleanup_temp_files,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 500 * 1024 * 1024))  # 500MB default
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "large-v3")

# Ensure directories exist
ensure_dir(UPLOAD_DIR)
ensure_dir(OUTPUT_DIR)

# Job storage (in production, use Redis or database)
jobs: dict[str, TranscriptionStatus] = {}

# Create FastAPI app
app = FastAPI(
    title="Faster-Whisper Transcription API",
    description="High-performance speech-to-text API using faster-whisper with CUDA acceleration",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Preload model on startup for faster first inference."""
    if PRELOAD_MODEL:
        logger.info(f"Preloading model: {PRELOAD_MODEL}")
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                preload_model,
                ModelSize(PRELOAD_MODEL)
            )
            logger.info("Model preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "faster-whisper-api",
        "loaded_models": get_loaded_models(),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    """List available and loaded models."""
    return {
        "available": [m.value for m in ModelSize],
        "loaded": get_loaded_models(),
    }


@app.post("/models/{model_size}/load")
async def load_model(model_size: ModelSize):
    """Preload a specific model into memory."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, preload_model, model_size)
        return {"status": "loaded", "model": model_size.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_size}")
async def delete_model(model_size: ModelSize):
    """Unload a model from memory."""
    if unload_model(model_size):
        return {"status": "unloaded", "model": model_size.value}
    raise HTTPException(status_code=404, detail="Model not loaded")


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    file: UploadFile = File(..., description="Audio or video file to transcribe"),
    language: Optional[str] = Form(default=None, description="Language code (auto-detect if not specified)"),
    model_size: ModelSize = Form(default=ModelSize.LARGE_V3, description="Model size"),
    output_format: OutputFormat = Form(default=OutputFormat.BOTH, description="Output format"),
    beam_size: int = Form(default=5, ge=1, le=10, description="Beam size"),
    vad_filter: bool = Form(default=True, description="Enable VAD filter"),
    word_timestamps: bool = Form(default=False, description="Include word timestamps"),
    initial_prompt: Optional[str] = Form(default=None, description="Initial prompt"),
    batch_size: int = Form(default=16, ge=1, le=64, description="Batch size for inference"),
):
    """
    Transcribe an audio/video file synchronously.
    Returns the transcription result directly.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not (is_audio_file(file.filename) or is_video_file(file.filename)):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: mp3, wav, flac, aac, ogg, m4a, mp4, mkv, avi, mov, webm"
        )

    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")

    try:
        # Stream file to disk
        async with aiofiles.open(upload_path, "wb") as f:
            total_size = 0
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
                    )
                await f.write(chunk)

        logger.info(f"File saved: {upload_path} ({total_size / 1024 / 1024:.2f}MB)")

        # Create request object
        request = TranscriptionRequest(
            language=language,
            model_size=model_size,
            output_format=output_format,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            batch_size=batch_size,
        )

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            transcribe_file,
            upload_path,
            request,
            None,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        cleanup_temp_files(upload_path)


@app.post("/transcribe/async")
async def transcribe_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    model_size: ModelSize = Form(default=ModelSize.LARGE_V3),
    output_format: OutputFormat = Form(default=OutputFormat.BOTH),
    beam_size: int = Form(default=5, ge=1, le=10),
    vad_filter: bool = Form(default=True),
    word_timestamps: bool = Form(default=False),
    initial_prompt: Optional[str] = Form(default=None),
    batch_size: int = Form(default=16, ge=1, le=64),
):
    """
    Start an asynchronous transcription job.
    Returns a job ID that can be used to check status.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not (is_audio_file(file.filename) or is_video_file(file.filename)):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Generate job ID
    job_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")

    # Save file
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        await f.write(content)

    # Create job status
    jobs[job_id] = TranscriptionStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
    )

    # Create request
    request = TranscriptionRequest(
        language=language,
        model_size=model_size,
        output_format=output_format,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        batch_size=batch_size,
    )

    # Add background task
    background_tasks.add_task(
        process_transcription_job,
        job_id,
        upload_path,
        request,
    )

    return {"job_id": job_id, "status": "pending"}


async def process_transcription_job(
    job_id: str,
    file_path: str,
    request: TranscriptionRequest,
):
    """Background task to process transcription."""
    try:
        jobs[job_id].status = "processing"

        def progress_callback(progress: float):
            jobs[job_id].progress = progress

        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            transcribe_file,
            file_path,
            request,
            progress_callback,
        )

        jobs[job_id].status = "completed"
        jobs[job_id].progress = 1.0
        jobs[job_id].result = result

        # Save output files
        output_base = os.path.join(OUTPUT_DIR, job_id)

        if result.srt_content:
            async with aiofiles.open(f"{output_base}.srt", "w", encoding="utf-8") as f:
                await f.write(result.srt_content)

        if result.txt_content:
            async with aiofiles.open(f"{output_base}.txt", "w", encoding="utf-8") as f:
                await f.write(result.txt_content)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id].status = "failed"
        jobs[job_id].message = str(e)
    finally:
        cleanup_temp_files(file_path)


@app.get("/jobs/{job_id}", response_model=TranscriptionStatus)
async def get_job_status(job_id: str):
    """Get the status of a transcription job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs/{job_id}/download/{format}")
async def download_result(job_id: str, format: str):
    """Download transcription result in specified format."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status}")

    if format not in ("srt", "txt"):
        raise HTTPException(status_code=400, detail="Invalid format. Use 'srt' or 'txt'")

    file_path = os.path.join(OUTPUT_DIR, f"{job_id}.{format}")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Output file not found for format: {format}")

    return FileResponse(
        file_path,
        media_type="text/plain; charset=utf-8",
        filename=f"{job.result.filename if job.result else job_id}.{format}",
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Cleanup files
    for ext in ("srt", "txt"):
        file_path = os.path.join(OUTPUT_DIR, f"{job_id}.{ext}")
        cleanup_temp_files(file_path)

    del jobs[job_id]
    return {"status": "deleted", "job_id": job_id}
