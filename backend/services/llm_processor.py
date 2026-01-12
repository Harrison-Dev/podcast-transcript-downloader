"""
LLM Text Processor - Two-pass text polishing using Ollama API.

Uses Ollama for:
1. Pass 1: Add punctuation, fix recognition errors (batch size: 8)
2. Pass 2: Polish into fluent article (batch size: 20)

Features:
- Checkpoint/resume support for interrupted processing
- Exponential backoff retry mechanism
- Progress estimation with ETA
"""
import json
import logging
import time
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from config import settings
from services.ollama_client import get_ollama_client, check_ollama_available

logger = logging.getLogger(__name__)

# Podcast keywords config file
KEYWORDS_FILE = Path(__file__).parent.parent / "data" / "podcast_keywords.json"


# ========== Prompts ==========

SYSTEM_PROMPT_PASS1 = """你是逐字稿打字員。你要把語音辨識的結果轉成正確標點的逐字稿。
請使用繁體中文（台灣正體字）輸出，不要使用簡體字。

重要原則：這是「逐字稿」，必須一字不漏保留說話者說的每一句話。

你可以做：
- 加標點符號（，。？！）
- 修正語音辨識的錯字（同音字錯誤）
- 把所有簡體字轉成繁體字
- 刪除結巴重複（「我我我」→「我」）
- 每 3-5 句換行分段

你不能做（絕對禁止）：
- 刪除任何句子
- 把多句濃縮成一句
- 改寫或重組內容
- 寫摘要、總結、重點整理
- 加標題
- 使用 Markdown（禁止 ** # --- > - 等符號）

輸入是什麼內容，輸出就要是什麼內容，只是加上標點和分段。
廣告、閒聊、開場白都要保留。
保留口語詞（就是、然後、那個、欸、喔）。

直接輸出處理後的文字，不要加任何說明。"""

SYSTEM_PROMPT_PASS2 = """你是逐字稿編輯。合併過短的段落讓文章更好讀。
請使用繁體中文（台灣用語）輸出。

你可以做：
- 合併太短的段落（少於 2 句的）
- 修正錯字
- 把簡體字轉成繁體字

你不能做：
- 刪除任何句子
- 寫摘要或重點
- 加標題
- 使用 Markdown 格式

直接輸出，不要加說明。"""

USER_PROMPT_PASS1 = """請將以下逐字稿整理成通順的文章，加入標點符號和適當分段：

{text}

只輸出整理後的文章，不要加入任何說明："""

USER_PROMPT_PASS2 = """請將以下已初步整理的文字，進一步潤飾成完整流暢的文章。合併過短的單行，形成完整的段落：

{text}

只輸出整理後的文章，不要加入任何說明："""


# ========== Checkpoint ==========

@dataclass
class Checkpoint:
    """Checkpoint state for resume capability."""
    pass_num: int  # 1 or 2
    completed_batches: int
    total_batches: int
    partial_results: list[str]
    remaining_lines: list[str]


def save_checkpoint(checkpoint: Checkpoint, output_dir: Path, basename: str) -> Path:
    """Save checkpoint to output directory."""
    checkpoint_path = output_dir / f".{basename}_checkpoint.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(checkpoint), f, ensure_ascii=False, indent=2)
    return checkpoint_path


def load_checkpoint(output_dir: Path, basename: str) -> Optional[Checkpoint]:
    """Load checkpoint from output directory."""
    checkpoint_path = output_dir / f".{basename}_checkpoint.json"
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Checkpoint(**data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def clear_checkpoint(output_dir: Path, basename: str) -> None:
    """Clear checkpoint file."""
    checkpoint_path = output_dir / f".{basename}_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


# ========== Keywords ==========

_podcast_keywords = None


def load_podcast_keywords() -> dict:
    """Load podcast keywords from JSON file."""
    global _podcast_keywords
    if _podcast_keywords is None:
        if KEYWORDS_FILE.exists():
            try:
                with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
                    _podcast_keywords = json.load(f)
                logger.info(f"Loaded keywords for {len(_podcast_keywords)} podcasts")
            except Exception as e:
                logger.warning(f"Failed to load keywords: {e}")
                _podcast_keywords = {}
        else:
            _podcast_keywords = {}
    return _podcast_keywords


def get_podcast_keywords(show_name: str) -> dict:
    """Get keywords for a specific podcast."""
    keywords = load_podcast_keywords()
    return keywords.get(show_name, {})


def build_system_prompt_pass1(show_name: str = None) -> str:
    """Build Pass 1 system prompt with optional keyword corrections."""
    base_prompt = SYSTEM_PROMPT_PASS1

    if show_name:
        kw = get_podcast_keywords(show_name)
        if kw:
            additions = []

            if kw.get("corrections"):
                corrections_text = "\n".join(
                    f"  - {wrong} -> {right}"
                    for wrong, right in kw["corrections"].items()
                )
                additions.append(f"\n常見錯誤修正：\n{corrections_text}")

            if kw.get("terms"):
                terms_text = "、".join(kw["terms"])
                additions.append(f"\n本節目專有名詞：{terms_text}")

            if additions:
                base_prompt = base_prompt + "\n" + "\n".join(additions)

    return base_prompt


# ========== Utilities ==========

def format_time(seconds: float) -> str:
    """Format seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.0f} sec"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hr"


def clean_response(text: str) -> str:
    """Remove model-specific artifacts from response."""
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove markdown code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    return text.strip()


# ========== Batch Processor ==========

class BatchProcessor:
    """
    Two-pass batch processor with checkpoint support.

    Pass 1: Add punctuation, fix recognition errors
    Pass 2: Polish into fluent article
    """

    def __init__(
        self,
        batch_size_pass1: Optional[int] = None,
        batch_size_pass2: Optional[int] = None,
        output_dir: Optional[Path] = None,
        basename: Optional[str] = None,
        show_name: Optional[str] = None,
    ):
        self.batch_size_pass1 = batch_size_pass1 or getattr(settings, 'BATCH_SIZE_PASS1', 8)
        self.batch_size_pass2 = batch_size_pass2 or getattr(settings, 'BATCH_SIZE_PASS2', 20)
        self.output_dir = output_dir
        self.basename = basename
        self.show_name = show_name

        # Progress tracking
        self.batch_times: list[float] = []

        # Ollama client
        self.client = get_ollama_client()

    def _estimate_remaining_time(self, completed: int, total: int) -> str:
        """Estimate remaining time."""
        if not self.batch_times or completed == 0:
            return "calculating..."

        avg_time = sum(self.batch_times) / len(self.batch_times)
        remaining = (total - completed) * avg_time
        return format_time(remaining)

    def _create_batches(self, lines: list[str], batch_size: int) -> list[list[str]]:
        """Create batches from lines."""
        if not lines:
            return []
        return [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]

    def _merge_results(self, results: list[str]) -> str:
        """Merge batch results."""
        return "\n\n".join(results)

    def process_pass1(
        self,
        lines: list[str],
        checkpoint: Optional[Checkpoint] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Pass 1: Add punctuation, fix errors."""
        logger.info("Pass 1: Adding punctuation and fixing errors")

        # Resume from checkpoint or create new batches
        if checkpoint and checkpoint.pass_num == 1:
            batches = self._create_batches(checkpoint.remaining_lines, self.batch_size_pass1)
            results = checkpoint.partial_results.copy()
            start_batch = checkpoint.completed_batches
            logger.info(f"Resuming from checkpoint: {start_batch} batches completed")
        else:
            batches = self._create_batches(lines, self.batch_size_pass1)
            results = []
            start_batch = 0

        total = len(batches) + start_batch
        system_prompt = build_system_prompt_pass1(self.show_name)

        for i, batch in enumerate(batches, start_batch + 1):
            text = "\n".join(batch)
            prompt = USER_PROMPT_PASS1.format(text=text)

            eta = self._estimate_remaining_time(i - 1, total)
            logger.info(f"[Pass 1] Batch {i}/{total} (ETA: {eta})")

            start_time = time.time()
            result = self.client.generate(prompt, system_prompt)
            elapsed = time.time() - start_time
            self.batch_times.append(elapsed)

            if result:
                results.append(clean_response(result))
                logger.info(f"[Pass 1] Batch {i} complete ({elapsed:.1f}s)")
            else:
                results.append(text)  # Fallback to original
                logger.warning(f"[Pass 1] Batch {i} failed, using original")

            # Update progress
            if progress_callback:
                progress = int(50 * i / total)  # Pass 1 is 0-50%
                progress_callback(progress)

            # Save checkpoint
            if self.output_dir and self.basename:
                remaining_lines = [line for batch in batches[i - start_batch:] for line in batch]
                ckpt = Checkpoint(
                    pass_num=1,
                    completed_batches=i,
                    total_batches=total,
                    partial_results=results,
                    remaining_lines=remaining_lines,
                )
                save_checkpoint(ckpt, self.output_dir, self.basename)

            # Rate limiting
            if i < total:
                time.sleep(1)

        return self._merge_results(results)

    def process_pass2(
        self,
        text: str,
        checkpoint: Optional[Checkpoint] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Pass 2: Polish into fluent article."""
        logger.info("Pass 2: Polishing into fluent article")

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Resume from checkpoint or create new batches
        if checkpoint and checkpoint.pass_num == 2:
            batches = self._create_batches(checkpoint.remaining_lines, self.batch_size_pass2)
            results = checkpoint.partial_results.copy()
            start_batch = checkpoint.completed_batches
            logger.info(f"Resuming from checkpoint: {start_batch} batches completed")
        else:
            batches = self._create_batches(lines, self.batch_size_pass2)
            results = []
            start_batch = 0
            self.batch_times = []  # Reset timing

        total = len(batches) + start_batch

        for i, batch in enumerate(batches, start_batch + 1):
            batch_text = "\n".join(batch)
            prompt = USER_PROMPT_PASS2.format(text=batch_text)

            eta = self._estimate_remaining_time(i - 1, total)
            logger.info(f"[Pass 2] Batch {i}/{total} (ETA: {eta})")

            start_time = time.time()
            result = self.client.generate(prompt, SYSTEM_PROMPT_PASS2)
            elapsed = time.time() - start_time
            self.batch_times.append(elapsed)

            if result:
                results.append(clean_response(result))
                logger.info(f"[Pass 2] Batch {i} complete ({elapsed:.1f}s)")
            else:
                results.append(batch_text)
                logger.warning(f"[Pass 2] Batch {i} failed, using original")

            # Update progress
            if progress_callback:
                progress = 50 + int(50 * i / total)  # Pass 2 is 50-100%
                progress_callback(progress)

            # Save checkpoint
            if self.output_dir and self.basename:
                remaining_lines = [line for batch in batches[i - start_batch:] for line in batch]
                ckpt = Checkpoint(
                    pass_num=2,
                    completed_batches=i,
                    total_batches=total,
                    partial_results=results,
                    remaining_lines=remaining_lines,
                )
                save_checkpoint(ckpt, self.output_dir, self.basename)

            if i < total:
                time.sleep(1)

        return self._merge_results(results)

    def process(
        self,
        raw_text: str,
        single_pass: bool = False,
        checkpoint: Optional[Checkpoint] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """
        Run full two-pass processing.

        Args:
            raw_text: Raw transcript text
            single_pass: Only run Pass 1 (closer to verbatim transcript)
            checkpoint: Resume from checkpoint
            progress_callback: Progress callback (0-100)

        Returns:
            Processed text
        """
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        logger.info(f"Processing {len(lines)} lines")

        # Determine starting point
        if checkpoint:
            if checkpoint.pass_num == 1:
                pass1_text = self.process_pass1(lines, checkpoint, progress_callback)
            else:
                # Checkpoint is at Pass 2
                pass1_text = "\n\n".join(checkpoint.partial_results)
                pass1_text = self.process_pass2(pass1_text, checkpoint, progress_callback)
                return pass1_text
        else:
            pass1_text = self.process_pass1(lines, progress_callback=progress_callback)

        if single_pass:
            logger.info("Single-pass mode complete")
            if self.output_dir and self.basename:
                clear_checkpoint(self.output_dir, self.basename)
            return pass1_text

        final_text = self.process_pass2(pass1_text, progress_callback=progress_callback)
        logger.info("Two-pass processing complete")

        # Clear checkpoint
        if self.output_dir and self.basename:
            clear_checkpoint(self.output_dir, self.basename)

        return final_text


# ========== High-level API ==========

class TextPolisher:
    """
    High-level text polisher using Ollama.

    Compatible with existing pipeline interface.
    """

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if LLM polishing is available."""
        if self._available is None:
            if not settings.LLM_ENABLED:
                self._available = False
            else:
                self._available = check_ollama_available()
                if not self._available:
                    logger.warning("Ollama not available - LLM polishing disabled")
        return self._available

    def polish_text(
        self,
        text: str,
        show_name: Optional[str] = None,
        single_pass: bool = False,
        output_dir: Optional[Path] = None,
        basename: Optional[str] = None,
        resume: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """
        Polish transcript text using two-pass LLM processing.

        Args:
            text: Raw transcript text
            show_name: Podcast show name for keyword corrections
            single_pass: Only run Pass 1
            output_dir: Directory for checkpoint files
            basename: Base name for checkpoint files
            resume: Try to resume from checkpoint
            progress_callback: Progress callback (0-100)

        Returns:
            Polished text
        """
        if not self.is_available():
            logger.info("LLM polishing unavailable, returning original text")
            return text

        # Check for checkpoint
        checkpoint = None
        if resume and output_dir and basename:
            checkpoint = load_checkpoint(output_dir, basename)
            if checkpoint:
                logger.info(
                    f"Found checkpoint: Pass {checkpoint.pass_num}, "
                    f"{checkpoint.completed_batches}/{checkpoint.total_batches} batches"
                )

        processor = BatchProcessor(
            output_dir=output_dir,
            basename=basename,
            show_name=show_name,
        )

        return processor.process(
            text,
            single_pass=single_pass,
            checkpoint=checkpoint,
            progress_callback=progress_callback,
        )

    def polish_segments(
        self,
        segments: list[dict],
        show_name: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[dict]:
        """
        Polish transcript segments (backward compatible with old API).

        Note: This method combines segments, polishes, then returns.
        For better results, use polish_text() with the full text.
        """
        if not self.is_available():
            return segments

        # Combine segment texts
        combined = "\n".join(seg.get('text', '').strip() for seg in segments if seg.get('text'))

        # Polish the combined text
        polished = self.polish_text(combined, show_name=show_name, progress_callback=progress_callback)

        # Return as single segment (timing info is lost in polishing)
        if polished:
            return [{'text': polished, 'start': 0, 'end': 0}]
        return segments


# Global instance
_text_polisher: Optional[TextPolisher] = None


def get_text_polisher() -> TextPolisher:
    """Get the global text polisher instance (singleton)."""
    global _text_polisher
    if _text_polisher is None:
        _text_polisher = TextPolisher()
    return _text_polisher


def release_polisher_gpu():
    """Compatibility function - no longer needed with Ollama."""
    pass  # Ollama manages its own GPU memory
