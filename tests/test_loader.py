"""Tests für rag.loader: Textextraktion, Encoding-Fallback, Größenlimits."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loader import DocumentTooLargeError, extract_text


def test_extract_plain_txt(tmp_path):
    path = os.path.join(str(tmp_path), "a.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Hallo Welt mit Umlauten: äöüß")
    assert extract_text(path) == "Hallo Welt mit Umlauten: äöüß"


def test_extract_markdown(tmp_path):
    path = os.path.join(str(tmp_path), "a.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Überschrift\n\nEin Absatz.")
    content = extract_text(path)
    assert "Überschrift" in content


def test_extract_latin1_fallback(tmp_path):
    path = os.path.join(str(tmp_path), "a.txt")
    with open(path, "w", encoding="latin-1") as f:
        f.write("Café à la carte")
    content = extract_text(path)
    assert "Caf" in content


def test_unsupported_extension_raises(tmp_path):
    path = os.path.join(str(tmp_path), "a.docx")
    with open(path, "w") as f:
        f.write("egal")
    with pytest.raises(ValueError):
        extract_text(path)


def test_file_too_large_raises(tmp_path, monkeypatch):
    path = os.path.join(str(tmp_path), "big.txt")
    with open(path, "w") as f:
        f.write("x")

    import rag.loader as loader_mod

    monkeypatch.setattr(loader_mod, "MAX_FILE_SIZE_MB", 0)  # jede Datei gilt jetzt als zu groß
    with pytest.raises(DocumentTooLargeError):
        extract_text(path)
