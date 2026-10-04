package de.kostenlose.kirag.rag

import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Reine Kotlin-Implementierung einer TF-IDF-Suche mit Unigram+Bigram-
 * Vokabular und Kosinus-Ähnlichkeit – funktional äquivalent zu
 * sklearn.feature_extraction.text.TfidfVectorizer in rag/index_store.py,
 * aber ohne externe ML-Bibliothek (läuft komplett on-device, kostenlos).
 *
 * Vereinfachungen gegenüber scikit-learn (bewusst, für Mobile-Performance):
 * - einfache Tokenisierung (lowercase + Wortgrenzen-Regex) statt vollem
 *   Unicode-Tokenizer
 * - sublinear TF (1 + log(tf)) wie sublinear_tf=True im Original
 * - kein max_features-Cutoff nötig, da Dokumentenmengen auf Mobilgeräten
 *   deutlich kleiner sind als im Server-Anwendungsfall
 */
object TfIdfIndex {

    private val TOKEN_RE = Regex("[\\p{L}\\p{N}]+")

    private fun tokenize(text: String): List<String> {
        val words = TOKEN_RE.findAll(text.lowercase()).map { it.value }.toList()
        if (words.isEmpty()) return emptyList()
        // Unigramme + Bigramme, analog ngram_range=(1,2) im Original
        val tokens = ArrayList<String>(words.size * 2)
        tokens.addAll(words)
        for (i in 0 until words.size - 1) {
            tokens.add("${words[i]}_${words[i + 1]}")
        }
        return tokens
    }

    /**
     * Baut Vokabular, IDF-Gewichte und TF-IDF-Vektoren (sparse, als
     * term-index -> Gewicht Maps) für die gegebenen Chunk-Texte.
     */
    fun build(texts: List<String>): Triple<List<String>, List<Double>, List<Map<Int, Double>>> {
        val docTokenCounts = texts.map { text ->
            val counts = HashMap<String, Int>()
            for (tok in tokenize(text)) {
                counts[tok] = (counts[tok] ?: 0) + 1
            }
            counts
        }

        // Vokabular = alle Terme, die in mindestens einem Dokument vorkommen
        val vocabulary = LinkedHashSet<String>()
        for (counts in docTokenCounts) vocabulary.addAll(counts.keys)
        val vocabList = vocabulary.toList()
        val termIndex = vocabList.withIndex().associate { (i, t) -> t to i }

        val n = texts.size
        val docFreq = IntArray(vocabList.size)
        for (counts in docTokenCounts) {
            for (term in counts.keys) {
                docFreq[termIndex.getValue(term)]++
            }
        }
        // Smooth-IDF wie sklearn-Default: idf = ln((1+n)/(1+df)) + 1
        val idf = DoubleArray(vocabList.size) { i ->
            ln((1.0 + n) / (1.0 + docFreq[i])) + 1.0
        }

        val vectors = docTokenCounts.map { counts ->
            val raw = HashMap<Int, Double>(counts.size)
            for ((term, tf) in counts) {
                val idx = termIndex.getValue(term)
                // sublinear_tf: 1 + log(tf)
                val weight = (1.0 + ln(tf.toDouble())) * idf[idx]
                raw[idx] = weight
            }
            l2Normalize(raw)
        }

        return Triple(vocabList, idf.toList(), vectors)
    }

    /** Vektorisiert eine Suchanfrage mit dem bestehenden Vokabular/IDF. */
    fun vectorizeQuery(query: String, vocabulary: List<String>, idf: List<Double>): Map<Int, Double> {
        val termIndex = vocabulary.withIndex().associate { (i, t) -> t to i }
        val counts = HashMap<String, Int>()
        for (tok in tokenize(query)) {
            counts[tok] = (counts[tok] ?: 0) + 1
        }
        val raw = HashMap<Int, Double>()
        for ((term, tf) in counts) {
            val idx = termIndex[term] ?: continue
            val weight = (1.0 + ln(tf.toDouble())) * idf[idx]
            raw[idx] = weight
        }
        return l2Normalize(raw)
    }

    private fun l2Normalize(vec: Map<Int, Double>): Map<Int, Double> {
        if (vec.isEmpty()) return vec
        var sumSq = 0.0
        for (v in vec.values) sumSq += v * v
        val norm = sqrt(sumSq)
        if (norm == 0.0) return vec
        return vec.mapValues { it.value / norm }
    }

    /** Kosinus-Ähnlichkeit zwischen zwei bereits L2-normalisierten sparse Vektoren. */
    fun cosineSimilarity(a: Map<Int, Double>, b: Map<Int, Double>): Double {
        val (smaller, larger) = if (a.size <= b.size) a to b else b to a
        var dot = 0.0
        for ((idx, va) in smaller) {
            val vb = larger[idx] ?: continue
            dot += va * vb
        }
        return dot
    }

    /** Durchsucht alle [vectors] nach den topK ähnlichsten zu [queryVec]. */
    fun search(
        queryVec: Map<Int, Double>,
        vectors: List<Map<Int, Double>>,
        topK: Int,
    ): List<Pair<Int, Double>> {
        val scored = vectors.mapIndexed { i, v -> i to cosineSimilarity(queryVec, v) }
        return scored
            .filter { it.second > 0.0 }
            .sortedByDescending { it.second }
            .take(topK)
    }
}
