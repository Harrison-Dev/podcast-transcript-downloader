"""
Pydantic models for API and internal data structures.
"""
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, HttpUrl
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EpisodeInfo(BaseModel):
    """Parsed episode information from RSS feed."""
    title: str
    episode_number: Optional[int] = None
    publish_date: Optional[datetime] = None
    audio_url: str
    duration: Optional[int] = None  # seconds
    description: Optional[str] = None


class ShowInfo(BaseModel):
    """Parsed show/podcast information."""
    title: str
    description: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None
    episodes: list[EpisodeInfo] = []


class TranscriptionRequest(BaseModel):
    """Request to start a transcription job."""
    rss_url: HttpUrl
    episode_count: int = 5
    output_dir: Optional[str] = None
    skip_existing: bool = True
    language: Optional[str] = None  # Auto-detect if None


class EpisodeProgress(BaseModel):
    """Progress update for a single episode."""
    episode_title: str
    status: str  # downloading, transcribing, completed, skipped, failed
    progress: float = 0.0  # 0-100
    message: Optional[str] = None


class JobProgress(BaseModel):
    """Overall job progress."""
    job_id: str
    status: JobStatus
    show_title: Optional[str] = None
    total_episodes: int = 0
    completed_episodes: int = 0
    skipped_episodes: int = 0
    failed_episodes: int = 0
    current_episode: Optional[EpisodeProgress] = None
    error: Optional[str] = None


class JobResponse(BaseModel):
    """Response when creating a job."""
    job_id: str
    message: str


class PreviewResponse(BaseModel):
    """Response for RSS preview."""
    show: ShowInfo
    will_process: list[EpisodeInfo]
    will_skip: list[str]  # Already existing file names
