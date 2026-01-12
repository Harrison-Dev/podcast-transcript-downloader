"""Utility functions for audio/video processing."""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_audio_duration(file_path: str) -> float:
    """Get duration of audio/video file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting duration: {e}")
        return 0.0


def extract_audio(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000
) -> str:
    """
    Extract audio from video file or convert audio to proper format.
    Returns path to the processed audio file.
    """
    if output_path is None:
        # Create temp file with .wav extension
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", input_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", str(sample_rate),  # Sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                output_path
            ],
            capture_output=True,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list) -> str:
    """Convert transcription segments to SRT format."""
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between entries

    return "\n".join(srt_lines)


def segments_to_txt(segments: list) -> str:
    """Convert transcription segments to plain text format."""
    return "\n".join(segment.text.strip() for segment in segments)


def is_video_file(filename: str) -> bool:
    """Check if file is a video based on extension."""
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    return Path(filename).suffix.lower() in video_extensions


def is_audio_file(filename: str) -> bool:
    """Check if file is an audio based on extension."""
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'}
    return Path(filename).suffix.lower() in audio_extensions


def cleanup_temp_files(*file_paths: str) -> None:
    """Remove temporary files."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logger.debug(f"Cleaned up temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
