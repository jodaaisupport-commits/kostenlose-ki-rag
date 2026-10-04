"""Tests für rag.index_store: Hashing, Caching, Limits und TF-IDF-Suche."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.index_store import (
    IndexLimitError,
    MAX_FILES_PER_INDEX,
    _check_limits,
    _hash_files,
    build_or_load_index,
)


def _write_tmp_file(tmp_path, name: str, content: str) -> str:
    path = os.path.join(tmp_path, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_hash_changes_with_chunk_params(tmp_path):
    f = _write_tmp_file(str(tmp_path), "a.txt", "Hallo Welt, dies ist ein Test.")
    h1 = _hash_files([f], "tfidf", 220, 40)
    h2 = _hash_files([f], "tfidf", 100, 20)
    h3 = _hash_files([f], "gemini", 220, 40)
    assert h1 != h2, "Unterschiedliche Chunk-Parameter müssen unterschiedliche Cache-Keys erzeugen"
    assert h1 != h3, "Unterschiedlicher Embedding-Modus muss unterschiedlichen Cache-Key erzeugen"


def test_hash_stable_for_same_inputs(tmp_path):
    f = _write_tmp_file(str(tmp_path), "a.txt", "Immer derselbe Inhalt.")
    h1 = _hash_files([f], "tfidf", 220, 40)
    h2 = _hash_files([f], "tfidf", 220, 40)
    assert h1 == h2


def test_hash_changes_with_file_content(tmp_path):
    f1 = _write_tmp_file(str(tmp_path), "a.txt", "Version eins.")
    h1 = _hash_files([f1], "tfidf", 220, 40)
    f2 = _write_tmp_file(str(tmp_path), "a.txt", "Version zwei (geändert).")
    h2 = _hash_files([f2], "tfidf", 220, 40)
    assert h1 != h2


def test_check_limits_raises_on_too_many_files(tmp_path):
    files = []
    for i in range(MAX_FILES_PER_INDEX + 1):
        files.append(_write_tmp_file(str(tmp_path), f"f{i}.txt", "x"))
    with pytest.raises(IndexLimitError):
        _check_limits(files)


def test_check_limits_ok_for_few_small_files(tmp_path):
    files = [_write_tmp_file(str(tmp_path), "a.txt", "kleine Datei")]
    _check_limits(files)  # darf nicht raisen


def test_build_index_tfidf_and_search(tmp_path):
    f1 = _write_tmp_file(
        str(tmp_path), "katzen.txt", "Katzen sind unabhängige und neugierige Haustiere."
    )
    f2 = _write_tmp_file(
        str(tmp_path), "autos.txt", "Autos benötigen Benzin oder Strom als Antrieb."
    )

    index = build_or_load_index(
        file_paths=[f1, f2], embedding_mode="tfidf", chunk_size=50, overlap=5
    )
    assert index.is_ready
    assert set(index.source_names) == {"katzen.txt", "autos.txt"}

    results = index.search("Was sind Katzen?", top_k=1)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert top_chunk.source == "katzen.txt"
    assert score > 0


def test_build_index_uses_cache_on_second_call(tmp_path):
    f1 = _write_tmp_file(str(tmp_path), "doc.txt", "Ein Testdokument für den Cache-Mechanismus.")

    index1 = build_or_load_index(file_paths=[f1], embedding_mode="tfidf", chunk_size=50, overlap=5)
    index2 = build_or_load_index(file_paths=[f1], embedding_mode="tfidf", chunk_size=50, overlap=5)

    # Gleicher Cache-Key -> dasselbe (aus Cache geladene) Objekt inhaltlich
    assert index1.cache_key == index2.cache_key
    assert [c.text for c in index1.chunks] == [c.text for c in index2.chunks]


def test_build_index_raises_on_empty_file_list():
    with pytest.raises(ValueError):
        build_or_load_index(file_paths=[], embedding_mode="tfidf")
