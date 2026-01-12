"""Core transcription module using faster-whisper with batched inference."""
import os
import logging
from typing import Optional, Generator
from pathlib import Path

from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import Segment

from .models import (
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
    ModelSize,
    OutputFormat,
)
from .utils import (
    extract_audio,
    segments_to_srt,
    segments_to_txt,
    is_video_file,
    cleanup_temp_files,
    get_audio_duration,
)

logger = logging.getLogger(__name__)

# Global model cache
_model_cache: dict[str, WhisperModel] = {}
_batched_pipeline_cache: dict[str, BatchedInferencePipeline] = {}


def get_model(model_size: ModelSize) -> WhisperModel:
    """Get or create a cached WhisperModel instance."""
    model_key = model_size.value

    if model_key not in _model_cache:
        logger.info(f"Loading model: {model_key}")

        # Check if CUDA is available
        device = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        _model_cache[model_key] = WhisperModel(
            model_key,
            device=device,
            compute_type=compute_type,
            download_root="/app/models",  # Cache models in container
        )
        logger.info(f"Model {model_key} loaded on {device} with {compute_type}")

    return _model_cache[model_key]


def get_batched_pipeline(model_size: ModelSize) -> BatchedInferencePipeline:
    """Get or create a cached BatchedInferencePipeline instance."""
    model_key = model_size.value

    if model_key not in _batched_pipeline_cache:
        model = get_model(model_size)
        _batched_pipeline_cache[model_key] = BatchedInferencePipeline(model=model)
        logger.info(f"Batched pipeline created for {model_key}")

    return _batched_pipeline_cache[model_key]


def transcribe_file(
    file_path: str,
    request: TranscriptionRequest,
    progress_callback: Optional[callable] = None,
) -> TranscriptionResult:
    """
    Transcribe an audio/video file using faster-whisper with batched inference.

    Args:
        file_path: Path to the audio/video file
        request: Transcription parameters
        progress_callback: Optional callback for progress updates

    Returns:
        TranscriptionResult with segments and formatted output
    """
    temp_audio_path: Optional[str] = None

    try:
        # Extract audio if video file
        if is_video_file(file_path):
            logger.info(f"Extracting audio from video: {file_path}")
            temp_audio_path = extract_audio(file_path)
            audio_path = temp_audio_path
        else:
            audio_path = file_path

        # Get duration for progress tracking
        duration = get_audio_duration(audio_path)
        logger.info(f"Audio duration: {duration:.2f}s")

        # Get batched pipeline for inference
        pipeline = get_batched_pipeline(request.model_size)

        # Prepare transcription options
        transcribe_options = {
            "language": request.language,
            "beam_size": request.beam_size,
            "vad_filter": request.vad_filter,
            "word_timestamps": request.word_timestamps,
            "batch_size": request.batch_size,
        }

        if request.initial_prompt:
            transcribe_options["initial_prompt"] = request.initial_prompt

        logger.info(f"Starting transcription with options: {transcribe_options}")

        # Run batched transcription
        segments_result, info = pipeline.transcribe(
            audio_path,
            **transcribe_options,
        )

        # Collect segments
        segments: list[TranscriptionSegment] = []
        for i, segment in enumerate(segments_result):
            segments.append(TranscriptionSegment(
                id=i + 1,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            ))

            # Progress callback
            if progress_callback and duration > 0:
                progress = min(segment.end / duration, 1.0)
                progress_callback(progress)

        logger.info(f"Transcription completed: {len(segments)} segments")

        # Format output
        srt_content = None
        txt_content = None

        if request.output_format in (OutputFormat.SRT, OutputFormat.BOTH):
            srt_content = segments_to_srt(segments)

        if request.output_format in (OutputFormat.TXT, OutputFormat.BOTH):
            txt_content = segments_to_txt(segments)

        return TranscriptionResult(
            filename=Path(file_path).name,
            language=info.language,
            duration=duration,
            segments=segments,
            srt_content=srt_content,
            txt_content=txt_content,
        )

    finally:
        # Cleanup temp files
        if temp_audio_path:
            cleanup_temp_files(temp_audio_path)


def transcribe_stream(
    file_path: str,
    request: TranscriptionRequest,
) -> Generator[TranscriptionSegment, None, None]:
    """
    Stream transcription segments as they are processed.
    Useful for real-time progress updates.
    """
    temp_audio_path: Optional[str] = None

    try:
        # Extract audio if video file
        if is_video_file(file_path):
            temp_audio_path = extract_audio(file_path)
            audio_path = temp_audio_path
        else:
            audio_path = file_path

        # Get batched pipeline
        pipeline = get_batched_pipeline(request.model_size)

        # Run transcription
        segments_result, info = pipeline.transcribe(
            audio_path,
            language=request.language,
            beam_size=request.beam_size,
            vad_filter=request.vad_filter,
            word_timestamps=request.word_timestamps,
            batch_size=request.batch_size,
            initial_prompt=request.initial_prompt,
        )

        # Yield segments as they come
        for i, segment in enumerate(segments_result):
            yield TranscriptionSegment(
                id=i + 1,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )

    finally:
        if temp_audio_path:
            cleanup_temp_files(temp_audio_path)


def preload_model(model_size: ModelSize) -> None:
    """Preload a model into memory for faster first inference."""
    logger.info(f"Preloading model: {model_size.value}")
    get_batched_pipeline(model_size)
    logger.info(f"Model {model_size.value} preloaded successfully")


def get_loaded_models() -> list[str]:
    """Get list of currently loaded models."""
    return list(_model_cache.keys())


def unload_model(model_size: ModelSize) -> bool:
    """Unload a model from memory."""
    model_key = model_size.value

    if model_key in _model_cache:
        del _model_cache[model_key]
        if model_key in _batched_pipeline_cache:
            del _batched_pipeline_cache[model_key]
        logger.info(f"Model {model_key} unloaded")
        return True

    return False
