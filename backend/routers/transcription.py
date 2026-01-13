"""
API Router for transcription endpoints.
"""
import asyncio
import logging
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from models import (
    TranscriptionRequest, JobResponse, JobProgress, PreviewResponse
)
from services.rss_parser import parse_rss_feed
from services.file_manager import transcript_exists, get_transcript_path
from services.pipeline import get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transcription"])


@router.post("/transcribe", response_model=JobResponse)
async def start_transcription(request: TranscriptionRequest):
    """
    Start a new transcription job.
    Returns a job ID for tracking progress.
    """
    orchestrator = get_orchestrator()
    
    job = orchestrator.create_job(
        rss_url=str(request.rss_url),
        episode_count=request.episode_count,
        output_dir=request.output_dir,
        skip_existing=request.skip_existing,
        language=request.language,
    )
    
    # Start processing in background
    await orchestrator.start_job(job)
    
    return JobResponse(
        job_id=job.job_id,
        message=f"Transcription job started. Track progress via WebSocket /ws/progress/{job.job_id}",
    )


@router.get("/jobs/{job_id}", response_model=JobProgress)
async def get_job_status(job_id: str):
    """Get the current status of a transcription job."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.get_progress()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running transcription job."""
    orchestrator = get_orchestrator()
    success = await orchestrator.cancel_job(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": "Job cancelled", "job_id": job_id}


@router.post("/preview", response_model=PreviewResponse)
async def preview_rss(request: TranscriptionRequest):
    """
    Preview what would be processed without starting a job.
    Shows which episodes would be downloaded and which would be skipped.
    """
    try:
        show = parse_rss_feed(str(request.rss_url), request.episode_count)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse RSS: {e}")
    
    output_dir = Path(request.output_dir) if request.output_dir else None
    
    will_process = []
    will_skip = []
    
    for idx, episode in enumerate(show.episodes):
        if request.skip_existing and transcript_exists(
            show.title, episode, idx, output_dir
        ):
            path = get_transcript_path(show.title, episode, idx, output_dir)
            will_skip.append(path.name)
        else:
            will_process.append(episode)
    
    return PreviewResponse(
        show=show,
        will_process=will_process,
        will_skip=will_skip,
    )


# WebSocket connections for real-time progress
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
    
    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.
    """
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    await manager.connect(job_id, websocket)

    # Send initial status
    try:
        await websocket.send_json(job.get_progress().model_dump())
    except Exception:
        manager.disconnect(job_id, websocket)
        return

    # Get the current event loop for thread-safe callback scheduling
    loop = asyncio.get_running_loop()

    # Set up progress callback (called from ThreadPoolExecutor)
    def progress_callback(progress: JobProgress):
        """Thread-safe callback that schedules WebSocket send on the main event loop."""
        async def send_progress():
            try:
                await websocket.send_json(progress.model_dump())
            except Exception:
                pass

        # Schedule the coroutine on the main event loop (thread-safe)
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(send_progress(), loop=loop)
            )
        except RuntimeError:
            # Loop might be closed
            pass

    job.add_progress_callback(progress_callback)

    try:
        while True:
            # Keep connection alive, handle any client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                # Handle ping/pong or other client messages
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json(job.get_progress().model_dump())
                except Exception:
                    break

    except WebSocketDisconnect:
        pass

    finally:
        job.remove_progress_callback(progress_callback)
        manager.disconnect(job_id, websocket)
