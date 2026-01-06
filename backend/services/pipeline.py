"""
Pipeline Orchestrator - Manage the download-transcribe pipeline with async workers.
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from models import (
    EpisodeInfo, ShowInfo, JobStatus, JobProgress, EpisodeProgress
)
from services.rss_parser import parse_rss_feed
from services.downloader import download_with_retry, get_temp_audio_path, cleanup_temp_file
from services.transcriber import get_transcriber
from services.file_manager import transcript_exists, save_transcript, get_transcript_path
from config import settings

logger = logging.getLogger(__name__)


class TranscriptionJob:
    """Represents a single transcription job with its state."""
    
    def __init__(
        self,
        job_id: str,
        rss_url: str,
        episode_count: int,
        output_dir: Optional[Path],
        skip_existing: bool,
        language: Optional[str],
    ):
        self.job_id = job_id
        self.rss_url = rss_url
        self.episode_count = episode_count
        self.output_dir = output_dir or settings.OUTPUT_BASE_DIR
        self.skip_existing = skip_existing
        self.language = language
        
        # State
        self.status = JobStatus.PENDING
        self.show_info: Optional[ShowInfo] = None
        self.total_episodes = 0
        self.completed_episodes = 0
        self.skipped_episodes = 0
        self.failed_episodes = 0
        self.current_episode: Optional[EpisodeProgress] = None
        self.error: Optional[str] = None
        self.cancelled = False
        
        # Callbacks for progress updates
        self._progress_callbacks: list[Callable[[JobProgress], None]] = []
    
    def add_progress_callback(self, callback: Callable[[JobProgress], None]):
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[JobProgress], None]):
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def _notify_progress(self):
        """Notify all registered callbacks of progress update."""
        progress = self.get_progress()
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def get_progress(self) -> JobProgress:
        return JobProgress(
            job_id=self.job_id,
            status=self.status,
            show_title=self.show_info.title if self.show_info else None,
            total_episodes=self.total_episodes,
            completed_episodes=self.completed_episodes,
            skipped_episodes=self.skipped_episodes,
            failed_episodes=self.failed_episodes,
            current_episode=self.current_episode,
            error=self.error,
        )


class PipelineOrchestrator:
    """Manages transcription jobs and workers."""
    
    def __init__(self):
        self.jobs: Dict[str, TranscriptionJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)  # For CPU-bound transcription
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    def create_job(
        self,
        rss_url: str,
        episode_count: int = 5,
        output_dir: Optional[str] = None,
        skip_existing: bool = True,
        language: Optional[str] = None,
    ) -> TranscriptionJob:
        """Create a new transcription job."""
        job_id = str(uuid.uuid4())[:8]
        
        job = TranscriptionJob(
            job_id=job_id,
            rss_url=rss_url,
            episode_count=episode_count,
            output_dir=Path(output_dir) if output_dir else None,
            skip_existing=skip_existing,
            language=language,
        )
        
        self.jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[TranscriptionJob]:
        return self.jobs.get(job_id)
    
    async def start_job(self, job: TranscriptionJob) -> None:
        """Start processing a job asynchronously."""
        task = asyncio.create_task(self._process_job(job))
        self._running_tasks[job.job_id] = task
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        job.cancelled = True
        job.status = JobStatus.CANCELLED
        job._notify_progress()
        
        task = self._running_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        
        return True
    
    async def _process_job(self, job: TranscriptionJob) -> None:
        """Main job processing pipeline."""
        try:
            # Step 1: Parse RSS feed
            job.status = JobStatus.PARSING
            job._notify_progress()
            
            logger.info(f"[{job.job_id}] Parsing RSS: {job.rss_url}")
            job.show_info = parse_rss_feed(str(job.rss_url), job.episode_count)
            job.total_episodes = len(job.show_info.episodes)
            job._notify_progress()
            
            logger.info(
                f"[{job.job_id}] Found {job.total_episodes} episodes "
                f"for '{job.show_info.title}'"
            )
            
            # Step 2: Process each episode
            for idx, episode in enumerate(job.show_info.episodes):
                if job.cancelled:
                    break
                
                await self._process_episode(job, episode, idx)
            
            # Step 3: Complete
            if not job.cancelled:
                job.status = JobStatus.COMPLETED
                job.current_episode = None
                job._notify_progress()
                logger.info(f"[{job.job_id}] Job completed successfully")
        
        except Exception as e:
            logger.exception(f"[{job.job_id}] Job failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job._notify_progress()
        
        finally:
            self._running_tasks.pop(job.job_id, None)
    
    async def _process_episode(
        self,
        job: TranscriptionJob,
        episode: EpisodeInfo,
        index: int,
    ) -> None:
        """Process a single episode: check, download, transcribe, save."""
        
        show_title = job.show_info.title
        
        # Check if already exists
        if job.skip_existing and transcript_exists(
            show_title, episode, index, job.output_dir
        ):
            job.skipped_episodes += 1
            job.current_episode = EpisodeProgress(
                episode_title=episode.title,
                status="skipped",
                progress=100,
                message="Transcript already exists",
            )
            job._notify_progress()
            logger.info(f"[{job.job_id}] Skipping existing: {episode.title}")
            return
        
        temp_audio_path = None
        
        try:
            # Download
            job.status = JobStatus.DOWNLOADING
            job.current_episode = EpisodeProgress(
                episode_title=episode.title,
                status="downloading",
                progress=0,
            )
            job._notify_progress()
            
            def on_download_progress(pct: float):
                job.current_episode = EpisodeProgress(
                    episode_title=episode.title,
                    status="downloading",
                    progress=pct,
                )
                job._notify_progress()
            
            # Generate unique temp filename - strip query string from URL
            temp_id = f"{job.job_id}_{index}"
            parsed_url = urlparse(episode.audio_url)
            ext = Path(parsed_url.path).suffix or ".mp3"
            temp_audio_path = get_temp_audio_path(temp_id, ext)
            
            logger.info(f"[{job.job_id}] Downloading: {episode.title}")
            await download_with_retry(
                episode.audio_url,
                temp_audio_path,
                progress_callback=on_download_progress,
            )
            
            # Transcribe (run in thread pool to not block event loop)
            job.status = JobStatus.TRANSCRIBING
            job.current_episode = EpisodeProgress(
                episode_title=episode.title,
                status="transcribing",
                progress=0,
                message="Running Whisper AI...",
            )
            job._notify_progress()
            
            logger.info(f"[{job.job_id}] Transcribing: {episode.title}")
            
            loop = asyncio.get_event_loop()
            transcriber = get_transcriber()
            
            result = await loop.run_in_executor(
                self._executor,
                lambda: transcriber.transcribe(temp_audio_path, job.language),
            )
            
            # Save transcript
            output_path = save_transcript(
                show_title,
                episode,
                result.text,
                index,
                job.output_dir,
            )
            
            job.completed_episodes += 1
            job.current_episode = EpisodeProgress(
                episode_title=episode.title,
                status="completed",
                progress=100,
                message=f"Saved to {output_path.name}",
            )
            job._notify_progress()
            
            logger.info(f"[{job.job_id}] Completed: {episode.title} -> {output_path}")
        
        except Exception as e:
            logger.exception(f"[{job.job_id}] Failed episode {episode.title}: {e}")
            job.failed_episodes += 1
            job.current_episode = EpisodeProgress(
                episode_title=episode.title,
                status="failed",
                progress=0,
                message=str(e),
            )
            job._notify_progress()
        
        finally:
            # Cleanup temp file
            if temp_audio_path:
                await cleanup_temp_file(temp_audio_path)


# Global orchestrator instance
_orchestrator: Optional[PipelineOrchestrator] = None


def get_orchestrator() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator
