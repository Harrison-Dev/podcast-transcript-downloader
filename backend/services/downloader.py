"""
Podcast Audio Downloader - Async download with progress tracking.
"""
import aiohttp
import aiofiles
import asyncio
from pathlib import Path
from typing import Callable, Optional
import logging

from config import settings

logger = logging.getLogger(__name__)


async def download_audio(
    url: str,
    output_path: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> Path:
    """
    Download audio file from URL with progress tracking.
    
    Args:
        url: Audio file URL
        output_path: Path to save the downloaded file
        progress_callback: Optional callback for progress updates (0-100)
        chunk_size: Download chunk size in bytes
    
    Returns:
        Path to downloaded file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for large files
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            async with aiofiles.open(output_path, "wb") as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    await f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback and total_size > 0:
                        progress = (downloaded / total_size) * 100
                        progress_callback(progress)
    
    logger.info(f"Downloaded: {output_path} ({downloaded / 1024 / 1024:.1f} MB)")
    return output_path


async def download_with_retry(
    url: str,
    output_path: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
    max_retries: int = 3,
) -> Path:
    """
    Download with automatic retry on failure.
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await download_audio(url, output_path, progress_callback)
        except aiohttp.ClientError as e:
            last_error = e
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise last_error


def get_temp_audio_path(episode_id: str, extension: str = ".mp3") -> Path:
    """Generate a temporary path for downloaded audio."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in episode_id)
    return settings.TEMP_AUDIO_DIR / f"{safe_id}{extension}"


async def cleanup_temp_file(file_path: Path) -> None:
    """Remove temporary audio file after transcription."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
