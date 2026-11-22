"""Token-window chunking with overlap and page tracking.

A "token" is whatever the supplied ``span_fn`` says it is:
  * default: whitespace-delimited words (no dependencies), or
  * the embedding model's own sub-word tokenizer (see embeddings.hf_span_fn), which is what
    you want when you quote chunk sizes "in tokens".
Chunks are cut from the original text by character offsets, so text is never re-joined or altered.
"""
from __future__ import annotations

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass

SpanFn = Callable[[str], list[tuple[int, int]]]
_WORD = re.compile(r"\S+")


def whitespace_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _WORD.finditer(text)]


@dataclass(frozen=True)
class Chunk:
    file: str
    chunk_id: int
    page_start: int | None
    page_end: int | None
    text: str
    n_tokens: int

    @property
    def page_label(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


def chunk_pages(
    file: str,
    pages: list[tuple[int | None, str]],
    chunk_size: int,
    overlap: int,
    span_fn: SpanFn = whitespace_spans,
) -> list[Chunk]:
    """Chunk a document given as ``[(page_number | None, page_text), ...]``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    parts: list[str] = []
    page_starts: list[int] = []  # char offset where each page begins in the joined text
    page_nums: list[int | None] = []
    pos = 0
    for num, text in pages:
        text = text.strip()
        if not text:
            continue
        page_starts.append(pos)
        page_nums.append(num)
        parts.append(text)
        pos += len(text) + 2  # "\n\n" separator
    full = "\n\n".join(parts)
    if not full:
        return []

    spans = span_fn(full)
    if not spans:
        return []

    def page_at(char_off: int) -> int | None:
        i = bisect.bisect_right(page_starts, char_off) - 1
        return page_nums[max(i, 0)]

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(spans):
        end = min(start + chunk_size, len(spans))
        c0, c1 = spans[start][0], spans[end - 1][1]
        chunks.append(
            Chunk(
                file=file,
                chunk_id=len(chunks),
                page_start=page_at(c0),
                page_end=page_at(c1 - 1),
                text=full[c0:c1],
                n_tokens=end - start,
            )
        )
        if end == len(spans):
            break
        start += step
    return chunks
