"""
Whisper Transcription Service - GPU-accelerated transcription using faster-whisper.
With Traditional Chinese conversion and improved formatting.
"""
import logging
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded transcriber instance
_transcriber = None

# OpenCC converter for Simplified -> Traditional Chinese
_opencc_converter = None


def get_opencc_converter():
    """Get or create OpenCC converter for s2t conversion."""
    global _opencc_converter
    if _opencc_converter is None:
        try:
            from opencc import OpenCC
            _opencc_converter = OpenCC('s2t')  # Simplified to Traditional
            logger.info("OpenCC converter initialized (Simplified -> Traditional)")
        except ImportError:
            logger.warning("OpenCC not installed, skipping Chinese conversion")
            _opencc_converter = False  # Mark as unavailable
    return _opencc_converter if _opencc_converter else None


def convert_to_traditional(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese."""
    converter = get_opencc_converter()
    if converter:
        return converter.convert(text)
    return text


def format_transcript_text(segments: list[dict], include_timestamps: bool = False) -> str:
    """
    Format transcript with better sentence structure.
    Creates readable paragraphs by:
    1. Breaking based on time gaps between segments (pause > 1.5s = new paragraph)
    2. Breaking at sentence-ending punctuation if available (。！？等)
    3. Fallback to character count limits
    """
    if not segments:
        return ""
    
    # Strategy: Use time gaps between segments to create natural paragraphs
    # If there's a pause > PAUSE_THRESHOLD, start a new paragraph
    PAUSE_THRESHOLD = 1.5  # seconds - silence gap that indicates paragraph break
    MAX_PARAGRAPH_CHARS = 500  # Force break if paragraph gets too long
    MIN_PARAGRAPH_CHARS = 100  # Don't break if paragraph is too short
    
    paragraphs = []
    current_paragraph_texts = []
    current_paragraph_start_time = 0
    current_length = 0
    last_end_time = 0
    
    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue
        
        start_time = seg.get("start", 0)
        end_time = seg.get("end", 0)
        
        # Calculate gap from previous segment
        gap = start_time - last_end_time if i > 0 else 0
        
        # Decide if we should start a new paragraph
        should_break = False
        
        # Break on significant pause (natural paragraph boundary)
        if gap > PAUSE_THRESHOLD and current_length >= MIN_PARAGRAPH_CHARS:
            should_break = True
        
        # Break if current paragraph is getting too long
        if current_length + len(text) > MAX_PARAGRAPH_CHARS and current_length >= MIN_PARAGRAPH_CHARS:
            should_break = True
        
        # Also check for sentence-ending punctuation at the end of last segment
        if current_paragraph_texts:
            last_text = current_paragraph_texts[-1]
            if last_text and last_text[-1] in '。！？!?.':
                if current_length >= MIN_PARAGRAPH_CHARS and current_length + len(text) > MAX_PARAGRAPH_CHARS * 0.7:
                    should_break = True
        
        if should_break and current_paragraph_texts:
            # Save current paragraph with optional timestamp
            para_text = "".join(current_paragraph_texts)
            if include_timestamps:
                minutes = int(current_paragraph_start_time // 60)
                seconds = int(current_paragraph_start_time % 60)
                para_text = f"[{minutes:02d}:{seconds:02d}] {para_text}"
            paragraphs.append(para_text)
            
            # Start new paragraph
            current_paragraph_texts = []
            current_paragraph_start_time = start_time
            current_length = 0
        
        # Add text to current paragraph
        current_paragraph_texts.append(text)
        current_length += len(text)
        last_end_time = end_time
        
        # Track start time for first segment of paragraph
        if len(current_paragraph_texts) == 1:
            current_paragraph_start_time = start_time
    
    # Don't forget the last paragraph
    if current_paragraph_texts:
        para_text = "".join(current_paragraph_texts)
        if include_timestamps:
            minutes = int(current_paragraph_start_time // 60)
            seconds = int(current_paragraph_start_time % 60)
            para_text = f"[{minutes:02d}:{seconds:02d}] {para_text}"
        paragraphs.append(para_text)
    
    return "\n\n".join(paragraphs)


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    duration: float  # seconds
    segments: list[dict]  # Detailed segments with timestamps


class Transcriber:
    """
    Whisper transcription service using faster-whisper.
    Lazy-loads the model on first use.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        compute_type: str = None,
        convert_traditional: bool = True,
    ):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self.convert_traditional = convert_traditional
        self._model = None
    
    def _load_model(self):
        """Load the Whisper model (lazy initialization)."""
        if self._model is None:
            logger.info(
                f"Loading Whisper model: {self.model_name} "
                f"(device={self.device}, compute_type={self.compute_type})"
            )
            from faster_whisper import WhisperModel
            
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model loaded successfully")
        return self._model
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",  # or "translate" for English translation
        include_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'zh', 'en'). Auto-detect if None.
            task: 'transcribe' or 'translate'
            include_timestamps: Whether to include timestamps in output
        
        Returns:
            TranscriptionResult with full text and segments
        """
        model = self._load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=5,
            vad_filter=True,  # Voice activity detection to skip silence
            vad_parameters=dict(
                threshold=settings.VAD_THRESHOLD,
                min_speech_duration_ms=settings.VAD_MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=settings.VAD_MIN_SILENCE_DURATION_MS,
                speech_pad_ms=settings.VAD_SPEECH_PAD_MS,
            ),
        )
        
        # Collect all segments
        all_segments = []
        
        for segment in segments:
            text = segment.text.strip()
            
            # Convert to Traditional Chinese if enabled and detected language is Chinese
            if self.convert_traditional and info.language in ('zh', 'yue'):
                text = convert_to_traditional(text)
            
            all_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })
        
        # Format the full text with better paragraph structure
        full_text = format_transcript_text(all_segments, include_timestamps)
        
        logger.info(
            f"Transcription complete: {len(all_segments)} segments, "
            f"language={info.language}, duration={info.duration:.1f}s"
        )
        
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            duration=info.duration,
            segments=all_segments,
        )
    
    def transcribe_with_timestamps(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe and format with timestamps.
        Format: [MM:SS] Paragraph text
        """
        result = self.transcribe(audio_path, language, include_timestamps=True)
        return result.text
    
    def unload_model(self):
        """Explicitly unload Whisper model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("Whisper model unloaded, GPU memory released")


def get_transcriber() -> Transcriber:
    """Get the global transcriber instance (singleton)."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def release_transcriber_gpu():
    """Release GPU memory used by the transcriber."""
    global _transcriber
    if _transcriber is not None:
        _transcriber.unload_model()
        logger.info("Transcriber GPU memory released for LLM usage")
