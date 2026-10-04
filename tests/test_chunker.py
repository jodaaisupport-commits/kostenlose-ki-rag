"""Tests für rag.chunker: Wort-Chunking mit Overlap und Seitenzuordnung."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", source="x.txt") == []
    assert chunk_text("   \n  ", source="x.txt") == []


def test_short_text_single_chunk():
    text = "Dies ist ein kurzer Testsatz mit wenigen Wörtern."
    chunks = chunk_text(text, source="kurz.txt", chunk_size=220, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].source == "kurz.txt"
    assert chunks[0].chunk_index == 0


def test_long_text_produces_multiple_overlapping_chunks():
    words = [f"wort{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, source="lang.txt", chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # Aufeinanderfolgende Chunks sollten sich überlappen: das Ende von
    # Chunk i erscheint am Anfang von Chunk i+1.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    overlap_region = first_words[-20:]
    assert overlap_region == second_words[:20]

    # chunk_index muss fortlaufend sein
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_all_words_are_covered_without_gaps():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_text(text, source="x.txt", chunk_size=10, overlap=3)

    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(words)


def test_invalid_overlap_is_auto_corrected():
    # overlap >= chunk_size darf nicht zu einer Endlosschleife führen
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_text(text, source="x.txt", chunk_size=10, overlap=50)
    assert len(chunks) > 0


def test_non_positive_chunk_size_is_clamped():
    text = "ein zwei drei vier fünf"
    # chunk_size <= 0 darf nicht crashen, sondern wird auf mind. 1 geklemmt
    chunks = chunk_text(text, source="x.txt", chunk_size=0, overlap=0)
    assert len(chunks) > 0


def test_page_markers_are_extracted_and_not_counted_as_words():
    text = "[Seite 1]\nSatz auf Seite eins. [Seite 2]\nSatz auf Seite zwei."
    chunks = chunk_text(text, source="pdf.pdf", chunk_size=4, overlap=0)

    # Seitenmarker dürfen nicht als Wörter im Chunk-Text auftauchen
    for c in chunks:
        assert "[Seite" not in c.text

    pages = [c.page for c in chunks]
    assert 1 in pages
    assert 2 in pages


def test_page_is_none_when_no_markers_present():
    text = "Ganz normaler Text ohne Seitenmarker."
    chunks = chunk_text(text, source="x.txt", chunk_size=10, overlap=0)
    assert all(c.page is None for c in chunks)
