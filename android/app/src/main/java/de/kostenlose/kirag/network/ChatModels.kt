package de.kostenlose.kirag.network

import kotlinx.serialization.Serializable

enum class ChatRole { USER, ASSISTANT }

/** Eine einzelne Chat-Nachricht in der UI-History. */
data class ChatMessage(
    val role: ChatRole,
    val content: String,
    val citations: List<de.kostenlose.kirag.rag.SearchResult> = emptyList(),
)

enum class LlmProvider(val displayName: String) {
    GROQ("Groq (empfohlen, sehr schnell)"),
    GEMINI("Google Gemini"),
}

val GROQ_MODELS = listOf(
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
)

val GEMINI_MODELS = listOf(
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
)

data class SystemPromptPreset(val label: String, val prompt: String)

val SYSTEM_PROMPT_PRESETS = listOf(
    SystemPromptPreset(
        "Standard (ausgewogen)",
        "Du bist ein hilfreicher, präziser Assistent. " +
            "Wenn Kontext aus Dokumenten bereitgestellt wird, nutze ihn vorrangig, " +
            "um die Frage zu beantworten. Wenn die Antwort nicht im Kontext steht, " +
            "sage das ehrlich, anstatt zu raten. Antworte auf Deutsch, außer der " +
            "Nutzer schreibt in einer anderen Sprache.",
    ),
    SystemPromptPreset(
        "Strikt (nur aus Dokument antworten)",
        "Du bist ein strenger Dokumenten-Assistent. Beantworte Fragen " +
            "AUSSCHLIESSLICH auf Basis des bereitgestellten Kontexts. " +
            "Steht die Antwort nicht eindeutig im Kontext, antworte wortwörtlich: " +
            "'Das steht nicht in den hochgeladenen Dokumenten.' Erfinde niemals " +
            "Informationen. Antworte auf Deutsch.",
    ),
    SystemPromptPreset(
        "Zusammenfassen",
        "Du bist ein Assistent, der Inhalte prägnant zusammenfasst. " +
            "Fasse den bereitgestellten Kontext bzw. die Antwort in klaren, " +
            "kurzen Stichpunkten zusammen. Antworte auf Deutsch.",
    ),
    SystemPromptPreset(
        "Kreativ",
        "Du bist ein kreativer, lockerer Assistent mit Humor. Nutze " +
            "bereitgestellten Dokumentenkontext als Inspiration, aber antworte " +
            "frei und unterhaltsam. Antworte auf Deutsch.",
    ),
)

val DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_PRESETS.first().prompt
