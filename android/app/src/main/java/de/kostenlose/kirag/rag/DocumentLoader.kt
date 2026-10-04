package de.kostenlose.kirag.rag

import android.content.ContentResolver
import android.net.Uri
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import java.io.InputStream
import java.nio.charset.Charset

/** Wird geworfen, wenn eine Datei die konfigurierten Sicherheitslimits überschreitet. */
class DocumentTooLargeException(message: String) : Exception(message)

/** Wird geworfen, wenn das Dateiformat nicht unterstützt wird. */
class UnsupportedDocumentException(message: String) : Exception(message)

/**
 * Dokumenten-Loader: extrahiert reinen Text aus PDF-, TXT- und MD-Dateien.
 * Kotlin-Entsprechung von rag/loader.py, inkl. derselben Sicherheitslimits.
 */
object DocumentLoader {

    const val MAX_FILE_SIZE_MB = 25
    const val MAX_PDF_PAGES = 1000
    val SUPPORTED_EXTENSIONS = setOf("pdf", "txt", "md")

    data class LoadedFile(
        val name: String,
        val uri: Uri,
        val sizeBytes: Long,
    )

    fun extractText(resolver: ContentResolver, file: LoadedFile): String {
        checkFileSize(file)
        val ext = file.name.substringAfterLast('.', "").lowercase()
        return when (ext) {
            "pdf" -> extractPdfText(resolver, file.uri)
            "txt", "md" -> extractPlainText(resolver, file.uri)
            else -> throw UnsupportedDocumentException(
                "Nicht unterstütztes Dateiformat: .$ext. " +
                    "Unterstützt werden: ${SUPPORTED_EXTENSIONS.joinToString(", ")}"
            )
        }
    }

    private fun checkFileSize(file: LoadedFile) {
        val sizeMb = file.sizeBytes / (1024.0 * 1024.0)
        if (sizeMb > MAX_FILE_SIZE_MB) {
            throw DocumentTooLargeException(
                "Datei '${file.name}' ist %.1f MB groß und überschreitet das Limit von $MAX_FILE_SIZE_MB MB."
                    .format(sizeMb)
            )
        }
    }

    private fun extractPdfText(resolver: ContentResolver, uri: Uri): String {
        val inputStream: InputStream = resolver.openInputStream(uri)
            ?: throw UnsupportedDocumentException("Datei konnte nicht geöffnet werden.")
        inputStream.use { stream ->
            PDDocument.load(stream).use { document ->
                val numPages = document.numberOfPages
                if (numPages > MAX_PDF_PAGES) {
                    throw DocumentTooLargeException(
                        "PDF hat $numPages Seiten und überschreitet das Limit von $MAX_PDF_PAGES Seiten."
                    )
                }
                val pagesText = mutableListOf<String>()
                val stripper = PDFTextStripper()
                for (i in 1..numPages) {
                    stripper.startPage = i
                    stripper.endPage = i
                    val text = try {
                        stripper.getText(document)
                    } catch (e: Exception) {
                        ""
                    }
                    if (text.isNotBlank()) {
                        pagesText.add("[Seite $i]\n$text")
                    }
                }
                return pagesText.joinToString("\n\n")
            }
        }
    }

    private fun extractPlainText(resolver: ContentResolver, uri: Uri): String {
        val bytes = resolver.openInputStream(uri)?.use { it.readBytes() }
            ?: throw UnsupportedDocumentException("Datei konnte nicht geöffnet werden.")
        return try {
            bytes.toString(Charsets.UTF_8)
        } catch (e: Exception) {
            bytes.toString(Charset.forName("ISO-8859-1"))
        }
    }
}
