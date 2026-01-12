"""Pydantic models for the transcription API."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    SRT = "srt"
    TXT = "txt"
    BOTH = "both"


class ModelSize(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class TranscriptionRequest(BaseModel):
    """Request parameters for transcription."""
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'zh', 'en', 'ja'). Auto-detect if not specified."
    )
    model_size: ModelSize = Field(
        default=ModelSize.LARGE_V3,
        description="Whisper model size to use"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.BOTH,
        description="Output format: srt, txt, or both"
    )
    beam_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Beam size for decoding"
    )
    vad_filter: bool = Field(
        default=True,
        description="Enable VAD filter to remove silence"
    )
    word_timestamps: bool = Field(
        default=False,
        description="Include word-level timestamps"
    )
    initial_prompt: Optional[str] = Field(
        default=None,
        description="Initial prompt to guide transcription"
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Batch size for batched inference"
    )


class TranscriptionSegment(BaseModel):
    """A single transcription segment."""
    id: int
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    """Result of transcription."""
    filename: str
    language: str
    duration: float
    segments: list[TranscriptionSegment]
    srt_content: Optional[str] = None
    txt_content: Optional[str] = None


class TranscriptionStatus(BaseModel):
    """Status of a transcription job."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[TranscriptionResult] = None
