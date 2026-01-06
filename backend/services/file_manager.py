"""
File Manager - Handle output file naming, organization, and deduplication.
"""
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import settings
from models import EpisodeInfo


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    Sanitize a string for use as a filename.
    Removes invalid characters while preserving Chinese and other Unicode.
    """
    # Normalize unicode
    name = unicodedata.normalize("NFC", name)
    
    # Remove or replace invalid filename characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    name = re.sub(invalid_chars, "", name)
    
    # Replace multiple spaces/underscores with single underscore
    name = re.sub(r"[\s_]+", "_", name)
    
    # Remove leading/trailing spaces and dots
    name = name.strip(" ._")
    
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length].rstrip("_")
    
    return name or "untitled"


def generate_transcript_filename(
    episode: EpisodeInfo,
    index: int = 0,
) -> str:
    """
    Generate a transcript filename from episode info.
    Format: {episode_number}_{title}_{YYYY-MM-DD}.txt
    """
    # Episode number
    ep_num = episode.episode_number or (index + 1)
    ep_str = f"{ep_num:03d}"
    
    # Title (sanitized)
    title = sanitize_filename(episode.title, max_length=80)
    
    # Date
    if episode.publish_date:
        date_str = episode.publish_date.strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    return f"{ep_str}_{title}_{date_str}.txt"


def get_show_directory(show_title: str, base_dir: Optional[Path] = None) -> Path:
    """Get or create the directory for a podcast show."""
    base = base_dir or settings.OUTPUT_BASE_DIR
    show_dir_name = sanitize_filename(show_title, max_length=100)
    show_dir = base / show_dir_name
    show_dir.mkdir(parents=True, exist_ok=True)
    return show_dir


def get_transcript_path(
    show_title: str,
    episode: EpisodeInfo,
    index: int = 0,
    base_dir: Optional[Path] = None,
) -> Path:
    """Get the full path for a transcript file."""
    show_dir = get_show_directory(show_title, base_dir)
    filename = generate_transcript_filename(episode, index)
    return show_dir / filename


def transcript_exists(
    show_title: str,
    episode: EpisodeInfo,
    index: int = 0,
    base_dir: Optional[Path] = None,
) -> bool:
    """Check if a transcript already exists for this episode."""
    path = get_transcript_path(show_title, episode, index, base_dir)
    return path.exists()


def save_transcript(
    show_title: str,
    episode: EpisodeInfo,
    transcript_text: str,
    index: int = 0,
    base_dir: Optional[Path] = None,
    include_metadata: bool = True,
) -> Path:
    """
    Save transcript to file with optional metadata header.
    
    Returns:
        Path to the saved file
    """
    path = get_transcript_path(show_title, episode, index, base_dir)
    
    content_parts = []
    
    if include_metadata:
        content_parts.append(f"# {episode.title}")
        content_parts.append(f"# Show: {show_title}")
        if episode.publish_date:
            content_parts.append(f"# Date: {episode.publish_date.strftime('%Y-%m-%d')}")
        if episode.duration:
            minutes = episode.duration // 60
            content_parts.append(f"# Duration: {minutes} minutes")
        content_parts.append("")
        content_parts.append("-" * 50)
        content_parts.append("")
    
    content_parts.append(transcript_text)
    
    full_content = "\n".join(content_parts)
    
    path.write_text(full_content, encoding="utf-8")
    
    return path


def list_existing_transcripts(
    show_title: str,
    base_dir: Optional[Path] = None,
) -> list[str]:
    """List all existing transcript filenames for a show."""
    show_dir = get_show_directory(show_title, base_dir)
    if not show_dir.exists():
        return []
    return [f.name for f in show_dir.glob("*.txt")]
