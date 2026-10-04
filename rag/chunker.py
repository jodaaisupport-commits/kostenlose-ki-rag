"""Text-Chunking: zerlegt lange Dokumente in überlappende Abschnitte."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    page: Optional[int] = None  # 1-basierte Seitenzahl, falls bekannt (z. B. aus PDFs)


# Seitenmarker wie "[Seite 12]" (von rag.loader._extract_pdf_text erzeugt).
# Wichtig: das Leerzeichen zwischen "Seite" und der Zahl bedeutet, dass ein
# naives text.split() den Marker in zwei Tokens ("[Seite", "12]") zerreißt -
# daher wird der Marker HIER, auf dem Gesamttext, per Regex entfernt/erkannt,
# bevor überhaupt in Wörter gesplittet wird.
_PAGE_MARKER_RE = re.compile(r"\[Seite\s+(\d+)\]")


def _split_words_with_pages(text: str):
    """Zerlegt `text` in (Wort, Seite)-Paare. `Seite` ist die zuletzt vor dem
    jeweiligen Wort gesehene Seitenzahl aus einem "[Seite N]"-Marker, oder
    None, falls der Text keine Seitenmarker enthält.
    """
    words: List[str] = []
    word_pages: List[Optional[int]] = []
    current_page: Optional[int] = None
    pos = 0
    for match in _PAGE_MARKER_RE.finditer(text):
        segment = text[pos : match.start()]
        for w in segment.split():
            words.append(w)
            word_pages.append(current_page)
        current_page = int(match.group(1))
        pos = match.end()
    # Rest nach dem letzten Marker (oder gesamter Text, falls kein Marker)
    for w in text[pos:].split():
        words.append(w)
        word_pages.append(current_page)
    return words, word_pages


def chunk_text(
    text: str,
    source: str = "dokument",
    chunk_size: int = 220,
    overlap: int = 40,
) -> List[Chunk]:
    """Teilt Text in überlappende Wort-Chunks auf.

    chunk_size / overlap sind in Wörtern angegeben, damit die
    Chunk-Größe unabhängig von der Sprache halbwegs konsistent bleibt.

    Enthält der Text Seitenmarker der Form "[Seite N]" (wie sie
    rag.loader._extract_pdf_text erzeugt), wird jedem Chunk die Seite
    zugeordnet, auf der sein erstes Wort steht – für genauere
    Quellenangaben im UI.
    """
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    words, word_pages = _split_words_with_pages(text)
    if not words:
        return []

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    idx = 0
    pos = 0
    n = len(words)
    while pos < n:
        window = words[pos : pos + chunk_size]
        chunk_str = " ".join(window).strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    text=chunk_str,
                    source=source,
                    chunk_index=idx,
                    page=word_pages[pos],
                )
            )
            idx += 1
        if pos + chunk_size >= n:
            break
        pos += step

    return chunks
