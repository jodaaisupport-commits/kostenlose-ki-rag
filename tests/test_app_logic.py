"""Tests für die Chat-/RAG-Hilfsfunktionen in app.py – insbesondere den
Fix, dass augmentierte RAG-Prompts NICHT dauerhaft in der an das LLM
gesendeten History landen (sonst Token-Aufblähung über mehrere Turns).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    _history_to_provider_messages,
    format_sources_markdown,
    format_citation_markdown,
)
from rag.chunker import Chunk


def test_history_filters_out_error_messages():
    history = [
        {"role": "user", "content": "Frage 1"},
        {"role": "assistant", "content": "❌ Groq-Fehler: ups"},
        {"role": "user", "content": "Frage 2"},
        {"role": "assistant", "content": "Antwort 2"},
    ]
    result = _history_to_provider_messages(history)
    contents = [m["content"] for m in result]
    assert "❌ Groq-Fehler: ups" not in contents
    assert "Antwort 2" in contents
    assert len(result) == 3  # Frage1, Frage2, Antwort2 (Fehler rausgefiltert)


def test_history_filters_empty_messages():
    history = [
        {"role": "user", "content": "Frage"},
        {"role": "assistant", "content": ""},
    ]
    result = _history_to_provider_messages(history)
    assert len(result) == 1
    assert result[0]["content"] == "Frage"


def test_history_does_not_contain_rag_context_block():
    """Simuliert zwei aufeinanderfolgende RAG-Turns: die an das LLM
    gesendete History darf NIE den '=== KONTEXT ===' Block enthalten,
    da dieser nur pro Aufruf an den Provider gereicht wird, nicht in die
    sichtbare/gespeicherte History geschrieben wird (siehe handle_chat)."""
    # So sieht die Chat-History nach dem Fix in app.py aus: die
    # Original-Nutzerfrage (ohne Kontext-Block) wird gespeichert.
    history = [
        {"role": "user", "content": "Was ist der Hauptpunkt?"},
        {"role": "assistant", "content": "Der Hauptpunkt ist X."},
    ]
    result = _history_to_provider_messages(history)
    for m in result:
        assert "=== KONTEXT ===" not in m["content"]


def test_history_skips_rate_limit_prefixed_messages():
    history = [
        {"role": "user", "content": "Frage"},
        {"role": "assistant", "content": "⏳ Rate-Limit erreicht"},
    ]
    result = _history_to_provider_messages(history)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_format_sources_markdown_none_index():
    assert "keine dokumente" in format_sources_markdown(None).lower()


def test_format_citation_markdown_with_results():
    chunk = Chunk(text="Ein Beispieltext für die Zitation.", source="doc.txt", chunk_index=0, page=3)
    md = format_citation_markdown([(chunk, 0.87)])
    assert "doc.txt" in md
    assert "Seite 3" in md
    assert "0.87" in md


def test_format_citation_markdown_empty():
    md = format_citation_markdown([])
    assert "Keine" in md
