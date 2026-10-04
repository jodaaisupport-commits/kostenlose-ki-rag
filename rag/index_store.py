"""Persistenter RAG-Index: TF-IDF (Standard, kostenlos & lokal) oder
Gemini-Embeddings (optional, benötigt GOOGLE_API_KEY).

Der Index wird pro Session im Arbeitsspeicher gehalten und zusätzlich
als Cache-Datei (joblib) persistiert, damit ein erneutes Hochladen
derselben Dokumente mit denselben Einstellungen nicht erneut verarbeitet
werden muss.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk, chunk_text
from .loader import extract_text

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Sicherheitsgrenzen für einen einzelnen Indizierungs-Vorgang.
MAX_FILES_PER_INDEX = 20
MAX_TOTAL_SIZE_MB = 100


class IndexLimitError(ValueError):
    """Wird ausgelöst, wenn zu viele/zu große Dateien auf einmal indiziert werden sollen."""


def _hash_files(file_paths: List[str], embedding_mode: str, chunk_size: int, overlap: int) -> str:
    """Erzeugt einen stabilen Hash über Dateiinhalte + Namen + Indizierungs-
    Parameter für den Cache-Key. Ändert sich chunk_size/overlap/Modus, wird
    automatisch neu indiziert statt fälschlich ein alter Cache-Eintrag
    zurückgegeben.
    """
    hasher = hashlib.sha256()
    hasher.update(f"mode={embedding_mode}|chunk={chunk_size}|overlap={overlap}".encode("utf-8"))
    for path in sorted(file_paths):
        hasher.update(os.path.basename(path).encode("utf-8"))
        try:
            with open(path, "rb") as f:
                hasher.update(f.read())
        except OSError:
            pass
    return hasher.hexdigest()[:24]


def _check_limits(file_paths: List[str]) -> None:
    if len(file_paths) > MAX_FILES_PER_INDEX:
        raise IndexLimitError(
            f"Zu viele Dateien auf einmal ({len(file_paths)}). "
            f"Maximal {MAX_FILES_PER_INDEX} Dateien pro Indizierung erlaubt."
        )
    total_mb = 0.0
    for p in file_paths:
        try:
            total_mb += os.path.getsize(p) / (1024 * 1024)
        except OSError:
            continue
    if total_mb > MAX_TOTAL_SIZE_MB:
        raise IndexLimitError(
            f"Gesamtgröße der Uploads ({total_mb:.1f} MB) überschreitet das "
            f"Limit von {MAX_TOTAL_SIZE_MB} MB."
        )


@dataclass
class RagIndex:
    chunks: List[Chunk] = field(default_factory=list)
    embedding_mode: str = "tfidf"  # "tfidf" oder "gemini"
    vectorizer: Optional[TfidfVectorizer] = None
    matrix: Optional[np.ndarray] = None  # TF-IDF Matrix oder Gemini-Embeddings
    source_names: List[str] = field(default_factory=list)
    cache_key: str = ""
    created_at: float = 0.0

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.matrix is not None

    def search(self, query: str, top_k: int = 4, embed_fn=None) -> List[Tuple[Chunk, float]]:
        if not self.is_ready:
            return []

        if self.embedding_mode == "tfidf":
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.matrix)[0]
        else:
            if embed_fn is None:
                raise ValueError("embed_fn wird für Gemini-Embedding-Suche benötigt")
            q_emb = np.array(embed_fn([query])[0], dtype=np.float32).reshape(1, -1)
            sims = cosine_similarity(q_emb, self.matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_indices:
            score = float(sims[i])
            if score <= 0:
                continue
            results.append((self.chunks[i], score))
        return results


def _cache_path(cache_key: str, embedding_mode: str) -> str:
    return os.path.join(CACHE_DIR, f"index_{embedding_mode}_{cache_key}.joblib")


def build_or_load_index(
    file_paths: List[str],
    embedding_mode: str = "tfidf",
    chunk_size: int = 220,
    overlap: int = 40,
    embed_fn=None,
    force_rebuild: bool = False,
    progress_cb=None,
) -> RagIndex:
    """Baut einen RAG-Index aus den gegebenen Dateien oder lädt ihn aus dem Cache."""
    if not file_paths:
        raise ValueError("Keine Dateien zum Indizieren übergeben.")

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, int(overlap))

    _check_limits(file_paths)

    cache_key = _hash_files(file_paths, embedding_mode, chunk_size, overlap)
    cache_file = _cache_path(cache_key, embedding_mode)

    def _notify(msg: str, frac: Optional[float] = None):
        if progress_cb:
            progress_cb(msg, frac)

    if not force_rebuild and os.path.exists(cache_file):
        try:
            _notify("📦 Lade zwischengespeicherten Index aus Cache ...", 0.5)
            index: RagIndex = joblib.load(cache_file)
            if index.is_ready:
                _notify("✅ Index aus Cache geladen.", 1.0)
                return index
        except Exception:
            pass  # Cache defekt -> neu bauen

    _notify("📄 Extrahiere Text aus Dokumenten ...", 0.0)
    all_chunks: List[Chunk] = []
    source_names: List[str] = []
    for file_idx, path in enumerate(file_paths):
        name = os.path.basename(path)
        source_names.append(name)
        try:
            text = extract_text(path)
        except Exception as e:
            _notify(f"⚠️ Fehler beim Lesen von {name}: {e}")
            continue
        if not text.strip():
            _notify(f"⚠️ Keine Textinhalte gefunden in {name}")
            continue
        chunks = chunk_text(text, source=name, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        _notify(
            f"📄 {name} verarbeitet ({file_idx + 1}/{len(file_paths)}) ...",
            0.1 + 0.2 * ((file_idx + 1) / max(1, len(file_paths))),
        )

    if not all_chunks:
        raise ValueError("Es konnte kein verwertbarer Text aus den Dokumenten extrahiert werden.")

    _notify(f"✂️ {len(all_chunks)} Text-Abschnitte erzeugt ...", 0.3)

    texts = [c.text for c in all_chunks]

    if embedding_mode == "gemini":
        if embed_fn is None:
            raise ValueError("embed_fn wird für Gemini-Embeddings benötigt (GOOGLE_API_KEY fehlt?)")
        _notify("🧠 Erzeuge Gemini-Embeddings (kann etwas dauern) ...", 0.3)
        batch_size = 32
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(embed_fn(batch))
            done = min(i + batch_size, len(texts))
            _notify(
                f"🧠 Embeddings: {done}/{len(texts)}",
                0.3 + 0.6 * (done / len(texts)),
            )
        matrix = np.array(vectors, dtype=np.float32)
        vectorizer = None
    else:
        _notify("🔢 Berechne TF-IDF-Vektoren (lokal, kostenlos) ...", 0.5)
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        _notify("🔢 TF-IDF-Vektoren berechnet.", 0.9)

    index = RagIndex(
        chunks=all_chunks,
        embedding_mode=embedding_mode,
        vectorizer=vectorizer,
        matrix=matrix,
        source_names=source_names,
        cache_key=cache_key,
        created_at=time.time(),
    )

    try:
        joblib.dump(index, cache_file)
        _notify("💾 Index im Cache gespeichert.", 1.0)
    except Exception as e:
        _notify(f"⚠️ Index konnte nicht gecacht werden: {e}", 1.0)

    return index
