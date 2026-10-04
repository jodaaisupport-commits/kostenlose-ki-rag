"""Text-Chunking: zerlegt lange Dokumente in überlappende Abschnitte."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(
    text: str,
    source: str = "dokument",
    chunk_size: int = 220,
    overlap: int = 40,
) -> List[Chunk]:
    """Teilt Text in überlappende Wort-Chunks auf.

    chunk_size / overlap sind in Wörtern angegeben, damit die
    Chunk-Größe unabhängig von der Sprache halbwegs konsistent bleibt.
    """
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    idx = 0
    pos = 0
    n = len(words)
    while pos < n:
        window = words[pos : pos + chunk_size]
        chunk_str = " ".join(window).strip()
        if chunk_str:
            chunks.append(Chunk(text=chunk_str, source=source, chunk_index=idx))
            idx += 1
        if pos + chunk_size >= n:
            break
        pos += step

    return chunks
