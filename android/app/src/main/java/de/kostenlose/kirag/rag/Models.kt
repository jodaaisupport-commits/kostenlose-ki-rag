package de.kostenlose.kirag.rag

import kotlinx.serialization.Serializable

/**
 * Entspricht rag/chunker.py::Chunk aus der Python-Referenzimplementierung:
 * ein überlappender Text-Abschnitt eines Dokuments, optional mit Seitenzahl.
 */
@Serializable
data class DocChunk(
    val text: String,
    val source: String,
    val chunkIndex: Int,
    val page: Int? = null,
)

/** Ein einzelnes indiziertes Dokument (Name + Anzahl erzeugter Chunks). */
@Serializable
data class IndexedSource(
    val name: String,
    val chunkCount: Int,
)

/** Zustand des aktuell geladenen RAG-Index (persistiert via Room/DataStore-Snapshot). */
@Serializable
data class RagIndexSnapshot(
    val chunks: List<DocChunk>,
    val sources: List<IndexedSource>,
    val vocabulary: List<String>,
    val idf: List<Double>,
    val tfidfVectors: List<Map<Int, Double>>, // sparse: term-index -> tfidf-weight pro Chunk
    val chunkSize: Int,
    val overlap: Int,
    val createdAt: Long,
) {
    val isReady: Boolean get() = chunks.isNotEmpty() && tfidfVectors.isNotEmpty()
}

/** Ein Suchtreffer: Chunk + Kosinus-Ähnlichkeits-Score. */
data class SearchResult(
    val chunk: DocChunk,
    val score: Double,
)
