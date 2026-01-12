"""
Whisper API Client - HTTP client for the containerized Whisper API service.

Replaces direct faster-whisper usage with HTTP calls to Whisper API container.
"""
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import httpx

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    duration: float  # seconds
    segments: list[dict]  # Detailed segments with timestamps
    srt_content: Optional[str] = None
    txt_content: Optional[str] = None


class WhisperClient:
    """
    HTTP client for Whisper API service.

    Supports both sync and async transcription modes.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: int = 3600,  # 1 hour for long files
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_url = (api_url or settings.WHISPER_API).rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def health_check(self) -> bool:
        """Check if Whisper API is available."""
        try:
            response = httpx.get(f"{self.api_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Whisper API health check failed: {e}")
            return False

    def transcribe_sync(
        self,
        file_path: Path,
        language: Optional[str] = None,
        model_size: str = "large-v3",
        progress_callback: Optional[callable] = None,
    ) -> TranscriptionResult:
        """
        Synchronous transcription - upload and wait for result.

        Args:
            file_path: Path to the audio file
            language: Language code (auto-detect if None)
            model_size: Whisper model size
            progress_callback: Optional callback for progress (0-100)

        Returns:
            TranscriptionResult with text, segments, etc.
        """
        url = f"{self.api_url}/transcribe"

        logger.info(f"Uploading to Whisper API: {file_path}")

        if progress_callback:
            progress_callback(5)  # Starting upload

        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            data = {
                'model_size': model_size,
                'output_format': 'both',
                'vad_filter': 'true',
            }
            if language:
                data['language'] = language

            last_error = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Reset file position for retry
                    f.seek(0)

                    with httpx.Client(timeout=self.timeout) as client:
                        response = client.post(url, files=files, data=data)
                        response.raise_for_status()

                    result = response.json()

                    if progress_callback:
                        progress_callback(100)

                    logger.info(
                        f"Transcription complete: language={result.get('language')}, "
                        f"duration={result.get('duration', 0):.1f}s, "
                        f"segments={len(result.get('segments', []))}"
                    )

                    return self._parse_result(result)

                except httpx.HTTPStatusError as e:
                    last_error = e
                    logger.error(f"Whisper API error: {e.response.status_code} - {e.response.text}")
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** (attempt - 1))
                        logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
                        time.sleep(delay)

                except httpx.RequestError as e:
                    last_error = e
                    logger.error(f"Whisper API request failed: {e}")
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** (attempt - 1))
                        logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
                        time.sleep(delay)

            raise RuntimeError(f"Transcription failed after {self.max_retries} attempts: {last_error}")

    def transcribe_async_start(
        self,
        file_path: Path,
        language: Optional[str] = None,
        model_size: str = "large-v3",
    ) -> str:
        """
        Start async transcription job.

        Returns:
            job_id for polling status
        """
        url = f"{self.api_url}/transcribe/async"

        logger.info(f"Starting async transcription: {file_path}")

        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            data = {
                'model_size': model_size,
                'output_format': 'both',
                'vad_filter': 'true',
            }
            if language:
                data['language'] = language

            with httpx.Client(timeout=300) as client:  # 5 min for upload
                response = client.post(url, files=files, data=data)
                response.raise_for_status()

            result = response.json()
            job_id = result.get('job_id')

            logger.info(f"Async job started: {job_id}")
            return job_id

    def transcribe_async_poll(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        progress_callback: Optional[callable] = None,
    ) -> TranscriptionResult:
        """
        Poll async transcription job until completion.

        Args:
            job_id: Job ID from transcribe_async_start
            poll_interval: Seconds between polls
            progress_callback: Optional callback for progress (0-100)

        Returns:
            TranscriptionResult when complete
        """
        url = f"{self.api_url}/jobs/{job_id}"

        while True:
            try:
                response = httpx.get(url, timeout=30)
                response.raise_for_status()
                status = response.json()

                progress = status.get('progress', 0) * 100
                state = status.get('status', 'unknown')

                if progress_callback:
                    progress_callback(progress)

                logger.debug(f"Job {job_id}: {state} ({progress:.1f}%)")

                if state == 'completed':
                    result = status.get('result', {})
                    logger.info(
                        f"Async transcription complete: language={result.get('language')}, "
                        f"duration={result.get('duration', 0):.1f}s"
                    )
                    return self._parse_result(result)

                elif state == 'failed':
                    raise RuntimeError(f"Transcription failed: {status.get('message')}")

                time.sleep(poll_interval)

            except httpx.RequestError as e:
                logger.warning(f"Poll request failed, retrying: {e}")
                time.sleep(poll_interval)

    def transcribe_async(
        self,
        file_path: Path,
        language: Optional[str] = None,
        model_size: str = "large-v3",
        progress_callback: Optional[callable] = None,
    ) -> TranscriptionResult:
        """
        Full async transcription: upload, poll, return result.

        Useful for long audio files to avoid HTTP timeout.
        """
        job_id = self.transcribe_async_start(file_path, language, model_size)
        return self.transcribe_async_poll(job_id, progress_callback=progress_callback)

    def _parse_result(self, data: dict) -> TranscriptionResult:
        """Parse API response into TranscriptionResult."""
        segments = []
        for seg in data.get('segments', []):
            segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', '').strip(),
            })

        # Build text from segments if txt_content not provided
        txt_content = data.get('txt_content')
        if not txt_content and segments:
            txt_content = '\n'.join(seg['text'] for seg in segments)

        return TranscriptionResult(
            text=txt_content or '',
            language=data.get('language', 'unknown'),
            duration=data.get('duration', 0),
            segments=segments,
            srt_content=data.get('srt_content'),
            txt_content=txt_content,
        )


# Global client instance
_whisper_client: Optional[WhisperClient] = None


def get_whisper_client() -> WhisperClient:
    """Get the global Whisper client instance (singleton)."""
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = WhisperClient()
    return _whisper_client
