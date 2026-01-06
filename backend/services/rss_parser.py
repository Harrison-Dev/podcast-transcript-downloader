"""
RSS Feed Parser - Extract podcast episode information from RSS/Atom feeds.
"""
import feedparser
from datetime import datetime
from typing import Optional
import re
import requests

from models import EpisodeInfo, ShowInfo


def parse_duration(duration_str: Optional[str]) -> Optional[int]:
    """Parse duration string to seconds. Handles HH:MM:SS, MM:SS, or seconds."""
    if not duration_str:
        return None
    
    try:
        # Try parsing as integer seconds
        return int(duration_str)
    except ValueError:
        pass
    
    # Try parsing as HH:MM:SS or MM:SS
    parts = duration_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    
    return None


def extract_episode_number(title: str, index: int) -> int:
    """Try to extract episode number from title, fallback to index."""
    # Common patterns: EP01, Ep.01, #01, Episode 01, 第1集, etc.
    patterns = [
        r'[Ee][Pp]\.?\s*(\d+)',
        r'#(\d+)',
        r'[Ee]pisode\s*(\d+)',
        r'第\s*(\d+)\s*[集期話话]',
        r'^(\d+)\.',
        r'\[(\d+)\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            return int(match.group(1))
    
    return index + 1  # Fallback to 1-indexed position


def fetch_feed_content(rss_url: str) -> str:
    """Fetch RSS feed content with proper headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        'Accept-Encoding': 'identity',  # Disable compression to avoid encoding issues
    }
    
    response = requests.get(rss_url, headers=headers, timeout=30)
    response.raise_for_status()
    
    # Try to detect encoding
    response.encoding = response.apparent_encoding or 'utf-8'
    return response.text


def parse_rss_feed(rss_url: str, max_episodes: Optional[int] = None) -> ShowInfo:
    """
    Parse an RSS/Atom feed and extract show and episode information.
    
    Args:
        rss_url: URL of the RSS feed
        max_episodes: Maximum number of episodes to return (newest first)
    
    Returns:
        ShowInfo with parsed data
    """
    # Fetch content first to handle encoding properly
    try:
        content = fetch_feed_content(rss_url)
        feed = feedparser.parse(content)
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch RSS feed: {e}")
    
    if feed.bozo and not feed.entries:
        raise ValueError(f"Failed to parse RSS feed: {feed.bozo_exception}")
    
    # Extract show info
    show_title = feed.feed.get("title", "Unknown Podcast")
    show_description = feed.feed.get("description") or feed.feed.get("subtitle")
    show_author = feed.feed.get("author") or feed.feed.get("itunes_author")
    show_image = None
    
    if hasattr(feed.feed, "image") and feed.feed.image:
        show_image = feed.feed.image.get("href")
    elif hasattr(feed.feed, "itunes_image"):
        show_image = feed.feed.itunes_image.get("href")
    
    # Extract episodes
    episodes = []
    entries = feed.entries[:max_episodes] if max_episodes else feed.entries
    
    for idx, entry in enumerate(entries):
        # Find audio URL from enclosures
        audio_url = None
        for enclosure in entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("href")
                break
        
        # Fallback: check links
        if not audio_url:
            for link in entry.get("links", []):
                if link.get("type", "").startswith("audio/"):
                    audio_url = link.get("href")
                    break
        
        if not audio_url:
            continue  # Skip entries without audio
        
        # Parse publish date
        publish_date = None
        if entry.get("published_parsed"):
            try:
                publish_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pass
        
        # Parse duration
        duration = parse_duration(
            entry.get("itunes_duration") or entry.get("duration")
        )
        
        # Create episode info
        episode = EpisodeInfo(
            title=entry.get("title", f"Episode {idx + 1}"),
            episode_number=extract_episode_number(entry.get("title", ""), idx),
            publish_date=publish_date,
            audio_url=audio_url,
            duration=duration,
            description=entry.get("description") or entry.get("summary"),
        )
        episodes.append(episode)
    
    return ShowInfo(
        title=show_title,
        description=show_description,
        author=show_author,
        image_url=show_image,
        episodes=episodes,
    )
