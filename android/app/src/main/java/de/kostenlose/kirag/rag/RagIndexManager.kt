package de.kostenlose.kirag.rag

import android.content.ContentResolver
import android.content.Context
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.security.MessageDigest

/** Wird geworfen, wenn zu viele/zu große Dateien auf einmal indiziert werden sollen. */
class IndexLimitException(message: String) : Exception(message)

/**
 * Orchestriert das Indizieren von Dokumenten und die persistente
 * Zwischenspeicherung (Cache) des Index auf dem Gerät.
 * Kotlin-Entsprechung von rag/index_store.py::build_or_load_index.
 *
 * Wichtig (siehe PR-Historie der Python-App): der Cache-Key enthält
 * chunkSize/overlap, damit geänderte Chunk-Parameter nicht fälschlich
 * einen alten Cache-Treffer zurückliefern.
 */
class RagIndexManager(private val context: Context) {

    companion object {
        const val MAX_FILES_PER_INDEX = 20
        const val MAX_TOTAL_SIZE_MB = 100
    }

    private val json = Json { ignoreUnknownKeys = true }
    private val cacheDir: File
        get() = File(context.cacheDir, "rag_index").apply { mkdirs() }

    fun checkLimits(files: List<DocumentLoader.LoadedFile>) {
        if (files.size > MAX_FILES_PER_INDEX) {
            throw IndexLimitException(
                "Zu viele Dateien auf einmal (${files.size}). " +
                    "Maximal $MAX_FILES_PER_INDEX Dateien pro Indizierung erlaubt."
            )
        }
        val totalMb = files.sumOf { it.sizeBytes } / (1024.0 * 1024.0)
        if (totalMb > MAX_TOTAL_SIZE_MB) {
            throw IndexLimitException(
                "Gesamtgröße der Uploads (%.1f MB) überschreitet das Limit von $MAX_TOTAL_SIZE_MB MB."
                    .format(totalMb)
            )
        }
    }

    private fun hashFiles(files: List<DocumentLoader.LoadedFile>, chunkSize: Int, overlap: Int): String {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("chunk=$chunkSize|overlap=$overlap".toByteArray())
        for (f in files.sortedBy { it.name }) {
            digest.update(f.name.toByteArray())
            digest.update(f.sizeBytes.toString().toByteArray())
        }
        return digest.digest().joinToString("") { "%02x".format(it) }.take(24)
    }

    private fun cacheFile(key: String): File = File(cacheDir, "index_$key.json")

    /**
     * Baut einen RAG-Index aus den gegebenen Dateien oder lädt ihn aus dem
     * lokalen Cache (falls identische Dateien + Chunk-Parameter bereits
     * zuvor indiziert wurden).
     */
    suspend fun buildOrLoadIndex(
        resolver: ContentResolver,
        files: List<DocumentLoader.LoadedFile>,
        chunkSize: Int = 220,
        overlap: Int = 40,
        forceRebuild: Boolean = false,
        onProgress: (String, Float?) -> Unit = { _, _ -> },
    ): RagIndexSnapshot {
        if (files.isEmpty()) throw IllegalArgumentException("Keine Dateien zum Indizieren übergeben.")
        checkLimits(files)

        val safeChunkSize = maxOf(1, chunkSize)
        val safeOverlap = maxOf(0, overlap)
        val key = hashFiles(files, safeChunkSize, safeOverlap)
        val cf = cacheFile(key)

        if (!forceRebuild && cf.exists()) {
            try {
                onProgress("📦 Lade zwischengespeicherten Index aus Cache ...", 0.5f)
                val cached = json.decodeFromString(RagIndexSnapshot.serializer(), cf.readText())
                if (cached.isReady) {
                    onProgress("✅ Index aus Cache geladen.", 1.0f)
                    return cached
                }
            } catch (e: Exception) {
                // Cache defekt -> neu bauen
            }
        }

        onProgress("📄 Extrahiere Text aus Dokumenten ...", 0.0f)
        val allChunks = mutableListOf<DocChunk>()
        val sources = mutableListOf<IndexedSource>()

        files.forEachIndexed { fileIdx, file ->
            val text = try {
                DocumentLoader.extractText(resolver, file)
            } catch (e: Exception) {
                onProgress("⚠️ Fehler beim Lesen von ${file.name}: ${e.message}", null)
                return@forEachIndexed
            }
            if (text.isBlank()) {
                onProgress("⚠️ Keine Textinhalte gefunden in ${file.name}", null)
                return@forEachIndexed
            }
            val chunks = Chunker.chunkText(text, file.name, safeChunkSize, safeOverlap)
            allChunks.addAll(chunks)
            sources.add(IndexedSource(file.name, chunks.size))
            onProgress(
                "📄 ${file.name} verarbeitet (${fileIdx + 1}/${files.size}) ...",
                0.1f + 0.3f * ((fileIdx + 1).toFloat() / files.size),
            )
        }

        if (allChunks.isEmpty()) {
            throw IllegalStateException("Es konnte kein verwertbarer Text aus den Dokumenten extrahiert werden.")
        }

        onProgress("✂️ ${allChunks.size} Text-Abschnitte erzeugt ...", 0.4f)
        onProgress("🔢 Berechne TF-IDF-Vektoren (lokal, kostenlos) ...", 0.5f)

        val (vocabulary, idf, vectors) = TfIdfIndex.build(allChunks.map { it.text })

        onProgress("🔢 TF-IDF-Vektoren berechnet.", 0.9f)

        val sparseVectorsAsMaps = vectors // bereits Map<Int, Double>

        val snapshot = RagIndexSnapshot(
            chunks = allChunks,
            sources = sources,
            vocabulary = vocabulary,
            idf = idf,
            tfidfVectors = sparseVectorsAsMaps,
            chunkSize = safeChunkSize,
            overlap = safeOverlap,
            createdAt = System.currentTimeMillis(),
        )

        try {
            cf.writeText(json.encodeToString(snapshot))
            onProgress("💾 Index im Cache gespeichert.", 1.0f)
        } catch (e: Exception) {
            onProgress("⚠️ Index konnte nicht gecacht werden: ${e.message}", 1.0f)
        }

        return snapshot
    }

    fun clearCache() {
        cacheDir.listFiles()?.forEach { it.delete() }
    }

    /** Durchsucht den Index nach den [topK] relevantesten Chunks zu [query]. */
    fun search(index: RagIndexSnapshot, query: String, topK: Int = 4): List<SearchResult> {
        if (!index.isReady) return emptyList()
        val queryVec = TfIdfIndex.vectorizeQuery(query, index.vocabulary, index.idf)
        val hits = TfIdfIndex.search(queryVec, index.tfidfVectors, topK)
        return hits.map { (i, score) -> SearchResult(index.chunks[i], score) }
    }
}
