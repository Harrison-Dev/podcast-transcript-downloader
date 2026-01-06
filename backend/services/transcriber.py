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
    Groups segments by natural breaks (punctuation, pauses).
    """
    if not segments:
        return ""
    
    paragraphs = []
    current_paragraph = []
    last_end_time = 0
    
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        
        start_time = seg["start"]
        
        # Start new paragraph if:
        # 1. Long pause (> 2 seconds)
        # 2. Previous text ended with sentence-ending punctuation
        pause_duration = start_time - last_end_time
        
        if current_paragraph:
            prev_text = current_paragraph[-1]["text"]
            ends_sentence = prev_text and prev_text[-1] in '。！？.!?'
            
            if pause_duration > 2.0 or (ends_sentence and pause_duration > 0.8):
                # Save current paragraph
                if include_timestamps:
                    para_start = current_paragraph[0]["start"]
                    minutes = int(para_start // 60)
                    seconds = int(para_start % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}] "
                else:
                    timestamp = ""
                
                para_text = "".join(s["text"] for s in current_paragraph)
                paragraphs.append(timestamp + para_text)
                current_paragraph = []
        
        current_paragraph.append(seg)
        last_end_time = seg["end"]
    
    # Don't forget the last paragraph
    if current_paragraph:
        if include_timestamps:
            para_start = current_paragraph[0]["start"]
            minutes = int(para_start // 60)
            seconds = int(para_start % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}] "
        else:
            timestamp = ""
        
        para_text = "".join(s["text"] for s in current_paragraph)
        paragraphs.append(timestamp + para_text)
    
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
                min_silence_duration_ms=500,
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


def get_transcriber() -> Transcriber:
    """Get the global transcriber instance (singleton)."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber
