package de.kostenlose.kirag.rag

/**
 * Text-Chunking: zerlegt lange Dokumente in überlappende Wort-Abschnitte.
 * Kotlin-Portierung von rag/chunker.py (inkl. des dort behobenen
 * Seitenmarker-Parsing-Bugs: "[Seite N]" wird auf dem Gesamttext per Regex
 * erkannt, BEVOR in Wörter gesplittet wird, da ein naives split() den
 * Marker sonst in zwei Tokens zerreißen würde).
 */
object Chunker {

    private val PAGE_MARKER_RE = Regex("""\[Seite\s+(\d+)]""")

    private fun splitWordsWithPages(text: String): Pair<List<String>, List<Int?>> {
        val words = mutableListOf<String>()
        val wordPages = mutableListOf<Int?>()
        var currentPage: Int? = null
        var pos = 0

        for (match in PAGE_MARKER_RE.findAll(text)) {
            val segment = text.substring(pos, match.range.first)
            for (w in segment.split(Regex("\\s+")).filter { it.isNotBlank() }) {
                words.add(w)
                wordPages.add(currentPage)
            }
            currentPage = match.groupValues[1].toIntOrNull()
            pos = match.range.last + 1
        }
        val rest = text.substring(pos)
        for (w in rest.split(Regex("\\s+")).filter { it.isNotBlank() }) {
            words.add(w)
            wordPages.add(currentPage)
        }
        return words to wordPages
    }

    /**
     * Teilt [text] in überlappende Wort-Chunks auf. chunkSize/overlap sind
     * in Wörtern angegeben (konsistent mit der Python-Referenz).
     */
    fun chunkText(
        text: String,
        source: String,
        chunkSize: Int = 220,
        overlap: Int = 40,
    ): List<DocChunk> {
        val safeChunkSize = maxOf(1, chunkSize)
        var safeOverlap = maxOf(0, overlap)
        if (safeOverlap >= safeChunkSize) {
            safeOverlap = maxOf(0, safeChunkSize / 4)
        }

        val (words, wordPages) = splitWordsWithPages(text)
        if (words.isEmpty()) return emptyList()

        val chunks = mutableListOf<DocChunk>()
        val step = safeChunkSize - safeOverlap
        var idx = 0
        var pos = 0
        val n = words.size

        while (pos < n) {
            val end = minOf(pos + safeChunkSize, n)
            val chunkStr = words.subList(pos, end).joinToString(" ").trim()
            if (chunkStr.isNotEmpty()) {
                chunks.add(
                    DocChunk(
                        text = chunkStr,
                        source = source,
                        chunkIndex = idx,
                        page = wordPages[pos],
                    )
                )
                idx++
            }
            if (pos + safeChunkSize >= n) break
            pos += step
        }
        return chunks
    }
}
