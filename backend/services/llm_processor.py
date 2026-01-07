"""
LLM Text Processor - GPU-accelerated text polishing using llama-cpp-python.

Uses local LLM models to:
1. Add proper punctuation (。，！？)
2. Remove filler words (呃、嗯、那個)
3. Fix grammar and improve readability

Designed for sequential GPU execution: Whisper releases GPU → LLM uses GPU
"""
import logging
import gc
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded client instance
_text_polisher = None

# Default model path (user should download GGUF model here)
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"


@dataclass
class PolishResult:
    """Result of text polishing."""
    original: str
    polished: str
    success: bool
    error: Optional[str] = None


# System prompt for text polishing
POLISH_SYSTEM_PROMPT = """你是一個專業的中文文字編輯，專門處理 podcast 逐字稿的潤飾工作。

你的任務是潤飾輸入的文字，遵循以下規則：
1. 添加適當的標點符號（。，！？、）
2. 移除重複的語氣詞（呃、嗯、那個、就是、對對對、然後）
3. 修正明顯的錯別字
4. 保持說話者的原意和語氣，不要改寫內容
5. 保持專有名詞不變

重要：只輸出潤飾後的文字，不要加任何解釋或前綴。不要使用 markdown 格式。"""


class LlamaCppClient:
    """
    Client for llama-cpp-python with GPU support.
    Designed for sequential execution after Whisper releases GPU.
    """
    
    def __init__(
        self,
        model_path: str = None,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_ctx: int = 4096,
    ):
        self.model_path = model_path or str(DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm = None
        self._available = None
    
    def _check_model_exists(self) -> bool:
        """Check if model file exists."""
        return Path(self.model_path).exists()
    
    def _load_model(self):
        """Load the LLM model (lazy initialization)."""
        if self._llm is None:
            if not self._check_model_exists():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}\n"
                    f"Please download a GGUF model to: {DEFAULT_MODEL_DIR}\n"
                    f"Recommended: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF"
                )
            
            logger.info(f"Loading LLM model: {self.model_path}")
            
            try:
                from llama_cpp import Llama
                
                self._llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=self.n_gpu_layers,
                    n_ctx=self.n_ctx,
                    verbose=False,
                )
                logger.info("LLM model loaded successfully")
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed.\n"
                    "For CPU: pip install llama-cpp-python\n"
                    "For GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall"
                )
        return self._llm
    
    def unload_model(self):
        """Explicitly unload model to free GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
            
            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("LLM model unloaded, GPU memory released")
    
    def generate(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text using llama-cpp.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text response
        """
        llm = self._load_model()
        
        # Format as chat messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more consistent output
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if LLM is available (model exists)."""
        if self._available is None:
            self._available = self._check_model_exists()
            if not self._available:
                logger.warning(
                    f"LLM model not found at {self.model_path}. "
                    f"Download a GGUF model to enable text polishing."
                )
        return self._available


class TextPolisher:
    """
    Text polisher using local LLM for transcript enhancement.
    """
    
    def __init__(self, client: LlamaCppClient = None):
        self.client = client or LlamaCppClient()
    
    def is_available(self) -> bool:
        """Check if text polishing is available."""
        return settings.LLM_ENABLED and self.client.is_available()
    
    def polish(self, text: str) -> PolishResult:
        """
        Polish a single text segment.
        
        Args:
            text: Raw transcript text
        
        Returns:
            PolishResult with original and polished text
        """
        if not text or not text.strip():
            return PolishResult(original=text, polished=text, success=True)
        
        if not self.is_available():
            return PolishResult(original=text, polished=text, success=True)
        
        try:
            prompt = f"請潤飾以下文字：\n\n{text}"
            polished = self.client.generate(
                prompt=prompt,
                system=POLISH_SYSTEM_PROMPT,
            )
            
            # Clean up the response
            polished = self._clean_response(polished)
            
            return PolishResult(
                original=text,
                polished=polished.strip(),
                success=True,
            )
        except Exception as e:
            logger.warning(f"Text polishing failed, using original: {e}")
            return PolishResult(
                original=text,
                polished=text,
                success=False,
                error=str(e),
            )
    
    def _clean_response(self, text: str) -> str:
        """Remove any model-specific artifacts from response."""
        import re
        # Remove <think>...</think> blocks from some models
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove markdown code blocks if present
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        return text.strip()
    
    def polish_segments(
        self,
        segments: list[dict],
        batch_size: int = None,
    ) -> list[dict]:
        """
        Polish multiple transcript segments.
        
        Args:
            segments: List of segment dicts with 'text' key
            batch_size: Number of segments to process together
        
        Returns:
            Segments with polished text
        """
        if not self.is_available():
            logger.info("LLM polishing disabled or unavailable, skipping")
            return segments
        
        batch_size = batch_size or settings.LLM_BATCH_SIZE
        polished_segments = []
        total = len(segments)
        
        logger.info(f"Starting LLM polish for {total} segments...")
        
        # Process in batches to reduce API calls
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Combine batch texts for single LLM call
            combined_text = "\n---\n".join(seg["text"] for seg in batch if seg.get("text"))
            result = self.polish(combined_text)
            
            if result.success and result.polished != result.original:
                # Split polished text back into segments
                polished_parts = result.polished.split("\n---\n")
                
                for j, seg in enumerate(batch):
                    new_seg = seg.copy()
                    if j < len(polished_parts):
                        new_seg["text"] = polished_parts[j].strip()
                    polished_segments.append(new_seg)
            else:
                # On failure, keep original segments
                polished_segments.extend(batch)
            
            logger.info(f"Polished batch {batch_num}/{(total + batch_size - 1) // batch_size}")
        
        logger.info(f"LLM polishing complete: {len(polished_segments)} segments")
        return polished_segments
    
    def cleanup(self):
        """Release GPU memory after polishing."""
        self.client.unload_model()


def get_text_polisher() -> TextPolisher:
    """Get the global text polisher instance (singleton)."""
    global _text_polisher
    if _text_polisher is None:
        _text_polisher = TextPolisher()
    return _text_polisher


def release_polisher_gpu():
    """Release GPU memory used by the text polisher."""
    global _text_polisher
    if _text_polisher is not None:
        _text_polisher.cleanup()
        _text_polisher = None
        logger.info("Text polisher released")
