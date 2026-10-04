package de.kostenlose.kirag

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import de.kostenlose.kirag.data.SecurePrefs
import de.kostenlose.kirag.network.ChatMessage
import de.kostenlose.kirag.network.ChatRole
import de.kostenlose.kirag.network.DEFAULT_SYSTEM_PROMPT
import de.kostenlose.kirag.network.GEMINI_MODELS
import de.kostenlose.kirag.network.GROQ_MODELS
import de.kostenlose.kirag.network.GeminiClient
import de.kostenlose.kirag.network.GroqClient
import de.kostenlose.kirag.network.LlmProvider
import de.kostenlose.kirag.network.ProviderError
import de.kostenlose.kirag.network.RateLimitError
import de.kostenlose.kirag.rag.DocumentLoader
import de.kostenlose.kirag.rag.DocumentTooLargeException
import de.kostenlose.kirag.rag.IndexLimitException
import de.kostenlose.kirag.rag.RagIndexManager
import de.kostenlose.kirag.rag.RagIndexSnapshot
import de.kostenlose.kirag.rag.SearchResult
import de.kostenlose.kirag.rag.UnsupportedDocumentException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Gesamter UI-Zustand des Chat-Bildschirms (ein einziger StateFlow, analog zu Gradio-State). */
data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val isGenerating: Boolean = false,
    val provider: LlmProvider = LlmProvider.GROQ,
    val groqModel: String = GROQ_MODELS.first(),
    val geminiModel: String = GEMINI_MODELS.first(),
    val temperature: Double = 0.3,
    val systemPrompt: String = DEFAULT_SYSTEM_PROMPT,
    val useRag: Boolean = true,
    val topK: Int = 4,
    val groqApiKey: String = "",
    val googleApiKey: String = "",
    val chunkSize: Int = 220,
    val overlap: Int = 40,
    val ragIndex: RagIndexSnapshot? = null,
    val indexStatus: String = "",
    val indexInProgress: Boolean = false,
    val indexProgress: Float = 0f,
    val errorBanner: String? = null,
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val securePrefs = SecurePrefs(application)
    private val ragManager = RagIndexManager(application)
    private val groqClient = GroqClient()
    private val geminiClient = GeminiClient()

    private val _uiState = MutableStateFlow(
        ChatUiState(
            groqApiKey = securePrefs.groqApiKey,
            googleApiKey = securePrefs.googleApiKey,
        )
    )
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null

    // ---------------------------------------------------------------
    // Einstellungen
    // ---------------------------------------------------------------

    fun updateGroqApiKey(key: String) {
        securePrefs.groqApiKey = key
        _uiState.update { it.copy(groqApiKey = key) }
    }

    fun updateGoogleApiKey(key: String) {
        securePrefs.googleApiKey = key
        _uiState.update { it.copy(googleApiKey = key) }
    }

    fun updateProvider(provider: LlmProvider) = _uiState.update { it.copy(provider = provider) }
    fun updateGroqModel(model: String) = _uiState.update { it.copy(groqModel = model) }
    fun updateGeminiModel(model: String) = _uiState.update { it.copy(geminiModel = model) }
    fun updateTemperature(value: Double) = _uiState.update { it.copy(temperature = value) }
    fun updateSystemPrompt(value: String) = _uiState.update { it.copy(systemPrompt = value) }
    fun updateUseRag(value: Boolean) = _uiState.update { it.copy(useRag = value) }
    fun updateTopK(value: Int) = _uiState.update { it.copy(topK = value) }
    fun updateChunkSize(value: Int) = _uiState.update { it.copy(chunkSize = value) }
    fun updateOverlap(value: Int) = _uiState.update { it.copy(overlap = value) }
    fun dismissError() = _uiState.update { it.copy(errorBanner = null) }

    // ---------------------------------------------------------------
    // Dokumenten-Indizierung
    // ---------------------------------------------------------------

    fun indexDocuments(uris: List<Uri>) {
        val state = _uiState.value
        if (uris.isEmpty()) {
            _uiState.update { it.copy(indexStatus = "⚠️ Bitte zuerst mindestens eine Datei auswählen.") }
            return
        }

        _uiState.update { it.copy(indexInProgress = true, indexProgress = 0f, indexStatus = "") }

        viewModelScope.launch {
            val resolver = getApplication<Application>().contentResolver
            val logLines = mutableListOf<String>()

            try {
                val loadedFiles = uris.map { uri ->
                    val name = queryDisplayName(uri) ?: uri.lastPathSegment ?: "dokument"
                    val size = querySize(uri)
                    DocumentLoader.LoadedFile(name, uri, size)
                }

                val index = ragManager.buildOrLoadIndex(
                    resolver = resolver,
                    files = loadedFiles,
                    chunkSize = state.chunkSize,
                    overlap = state.overlap,
                    onProgress = { msg, frac ->
                        logLines.add(msg)
                        _uiState.update {
                            it.copy(
                                indexStatus = logLines.joinToString("\n"),
                                indexProgress = frac ?: it.indexProgress,
                            )
                        }
                    },
                )

                logLines.add("✅ Index erfolgreich erstellt / geladen.")
                _uiState.update {
                    it.copy(
                        ragIndex = index,
                        indexStatus = logLines.joinToString("\n"),
                        indexInProgress = false,
                        indexProgress = 1f,
                    )
                }
            } catch (e: IndexLimitException) {
                logLines.add("🚫 ${e.message}")
                _uiState.update { it.copy(indexStatus = logLines.joinToString("\n"), indexInProgress = false) }
            } catch (e: DocumentTooLargeException) {
                logLines.add("🚫 ${e.message}")
                _uiState.update { it.copy(indexStatus = logLines.joinToString("\n"), indexInProgress = false) }
            } catch (e: UnsupportedDocumentException) {
                logLines.add("🚫 ${e.message}")
                _uiState.update { it.copy(indexStatus = logLines.joinToString("\n"), indexInProgress = false) }
            } catch (e: Exception) {
                logLines.add("❌ Fehler beim Indizieren: ${e.message}")
                _uiState.update { it.copy(indexStatus = logLines.joinToString("\n"), indexInProgress = false) }
            }
        }
    }

    fun clearIndex() {
        ragManager.clearCache()
        _uiState.update {
            it.copy(ragIndex = null, indexStatus = "🗑️ Index zurückgesetzt.", indexProgress = 0f)
        }
    }

    private fun queryDisplayName(uri: Uri): String? {
        val resolver = getApplication<Application>().contentResolver
        return resolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIdx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (nameIdx >= 0 && cursor.moveToFirst()) cursor.getString(nameIdx) else null
        }
    }

    private fun querySize(uri: Uri): Long {
        val resolver = getApplication<Application>().contentResolver
        return resolver.query(uri, null, null, null, null)?.use { cursor ->
            val sizeIdx = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE)
            if (sizeIdx >= 0 && cursor.moveToFirst()) cursor.getLong(sizeIdx) else 0L
        } ?: 0L
    }

    // ---------------------------------------------------------------
    // Chat
    // ---------------------------------------------------------------

    /**
     * Sendet [userText] an den gewählten LLM-Provider.
     *
     * WICHTIG (Lehre aus der Python-Referenz-App, siehe PR-Historie dort):
     * Der RAG-Kontext-Block wird NUR für diesen einzelnen API-Aufruf
     * gebaut und niemals in [ChatUiState.messages] gespeichert. Die dort
     * sichtbare History enthält immer nur die Original-Nutzerfrage, damit
     * der Kontext nicht bei jedem weiteren Chat-Turn erneut mitgeschickt
     * wird (sonst Token-/Rate-Limit-Explosion über die Zeit).
     */
    fun sendMessage(userText: String) {
        val text = userText.trim()
        if (text.isEmpty()) return

        val state = _uiState.value
        val userMsg = ChatMessage(ChatRole.USER, text)
        val placeholder = ChatMessage(ChatRole.ASSISTANT, "⏳ Denke nach ...")

        _uiState.update { it.copy(messages = it.messages + userMsg + placeholder, isGenerating = true) }

        // Saubere History für den LLM-Call: ohne die 2 gerade angehängten
        // Einträge, ohne Fehlermeldungen (❌/🚫/⏳-Präfix) aus früheren Turns.
        val historyForLlm = state.messages.filterCleanForProvider()

        generationJob = viewModelScope.launch {
            var augmentedMessage = text
            var citations: List<SearchResult> = emptyList()

            if (state.useRag && state.ragIndex?.isReady == true) {
                try {
                    citations = ragManager.search(state.ragIndex, text, state.topK)
                    if (citations.isNotEmpty()) {
                        val contextBlock = citations.joinToString("\n\n---\n\n") { (chunk, score) ->
                            val pageInfo = if (chunk.page != null) " | Seite ${chunk.page}" else ""
                            "[Quelle: ${chunk.source}$pageInfo | Relevanz: ${"%.2f".format(score)}]\n${chunk.text}"
                        }
                        augmentedMessage =
                            "Nutze den folgenden Kontext aus hochgeladenen Dokumenten, um die Frage zu beantworten.\n\n" +
                                "=== KONTEXT ===\n$contextBlock\n=== ENDE KONTEXT ===\n\n" +
                                "Frage: $text"
                    }
                } catch (e: Exception) {
                    // Bei Fehler ohne Kontext fortfahren (nicht kritisch für den Chat)
                }
            }

            updateLastAssistantMessage("")

            val flow = when (state.provider) {
                LlmProvider.GROQ -> groqClient.streamCompletion(
                    apiKey = state.groqApiKey,
                    model = state.groqModel,
                    systemPrompt = state.systemPrompt,
                    history = historyForLlm,
                    userMessage = augmentedMessage,
                    temperature = state.temperature,
                )
                LlmProvider.GEMINI -> geminiClient.streamCompletion(
                    apiKey = state.googleApiKey,
                    model = state.geminiModel,
                    systemPrompt = state.systemPrompt,
                    history = historyForLlm,
                    userMessage = augmentedMessage,
                    temperature = state.temperature,
                )
            }

            val fullResponse = StringBuilder()
            try {
                flow.catch { e -> throw e }
                    .collect { delta ->
                        fullResponse.append(delta)
                        updateLastAssistantMessage(fullResponse.toString())
                    }
                updateLastAssistantMessage(fullResponse.toString(), citations)
            } catch (e: RateLimitError) {
                updateLastAssistantMessage("⏳ ${e.message}")
            } catch (e: ProviderError) {
                updateLastAssistantMessage("❌ ${e.message}")
            } catch (e: Exception) {
                updateLastAssistantMessage("❌ Unerwarteter Fehler: ${e.message}")
            } finally {
                _uiState.update { it.copy(isGenerating = false) }
            }
        }
    }

    fun stopGeneration() {
        generationJob?.cancel()
        _uiState.update { it.copy(isGenerating = false) }
    }

    fun clearChat() {
        stopGeneration()
        _uiState.update { it.copy(messages = emptyList()) }
    }

    private fun updateLastAssistantMessage(content: String, citations: List<SearchResult> = emptyList()) {
        _uiState.update { state ->
            val msgs = state.messages.toMutableList()
            if (msgs.isNotEmpty() && msgs.last().role == ChatRole.ASSISTANT) {
                msgs[msgs.lastIndex] = msgs.last().copy(content = content, citations = citations)
            }
            state.copy(messages = msgs)
        }
    }

    /**
     * Filtert Fehlermeldungen (❌/🚫/⏳) und leere Nachrichten aus der
     * History heraus, bevor sie an den Provider geschickt wird - Portierung
     * von app.py::_history_to_provider_messages.
     */
    private fun List<ChatMessage>.filterCleanForProvider(): List<ChatMessage> =
        filter { msg ->
            msg.content.isNotBlank() &&
                !msg.content.trimStart().let { it.startsWith("❌") || it.startsWith("🚫") || it.startsWith("⏳") }
        }

    // ---------------------------------------------------------------
    // Chat-Export
    // ---------------------------------------------------------------

    fun exportChatToFile(): File? {
        val messages = _uiState.value.messages
        if (messages.isEmpty()) return null

        val dir = File(getApplication<Application>().cacheDir, "chat_exports").apply { mkdirs() }
        val timestamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.GERMANY).format(Date())
        val file = File(dir, "chat_export_$timestamp.md")

        val sb = StringBuilder("# Chat-Export – $timestamp\n\n")
        for (m in messages) {
            val role = if (m.role == ChatRole.USER) "🧑 Nutzer" else "🤖 Assistent"
            sb.append("### $role\n\n${m.content}\n\n")
        }
        file.writeText(sb.toString())
        return file
    }
}
