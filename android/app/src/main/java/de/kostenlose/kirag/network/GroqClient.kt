package de.kostenlose.kirag.network

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.BufferedReader
import java.util.concurrent.TimeUnit

/**
 * Groq-Chat-Completions-Client (OpenAI-kompatible API, SSE-Streaming).
 * Kotlin-Entsprechung von llm/providers.py::stream_groq_completion, inkl.
 * derselben verständlichen deutschen Fehlermeldungen für Rate-Limits/Auth.
 */
class GroqClient(private val httpClient: OkHttpClient = defaultHttpClient()) {

    companion object {
        private const val BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

        fun defaultHttpClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }

    fun streamCompletion(
        apiKey: String,
        model: String,
        systemPrompt: String,
        history: List<ChatMessage>,
        userMessage: String,
        temperature: Double = 0.3,
    ): Flow<String> = flow {
        if (apiKey.isBlank()) {
            throw ProviderError(
                "Kein Groq API-Key gefunden. Bitte in den Einstellungen eingeben " +
                    "(kostenlos erhältlich unter https://console.groq.com/keys)."
            )
        }

        val messages = buildJsonArray {
            add(buildJsonObject {
                put("role", "system")
                put("content", systemPrompt)
            })
            for (m in history) {
                add(buildJsonObject {
                    put("role", if (m.role == ChatRole.USER) "user" else "assistant")
                    put("content", m.content)
                })
            }
            add(buildJsonObject {
                put("role", "user")
                put("content", userMessage)
            })
        }

        val bodyJson = buildJsonObject {
            put("model", model)
            put("messages", messages)
            put("temperature", temperature)
            put("stream", true)
        }

        val request = Request.Builder()
            .url(BASE_URL)
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .post(bodyJson.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = try {
            httpClient.newCall(request).execute()
        } catch (e: Exception) {
            throw ProviderError("🌐 Verbindung zu Groq fehlgeschlagen. Bitte Internetverbindung prüfen.")
        }

        response.use { resp ->
            if (!resp.isSuccessful) {
                throw mapHttpError(resp.code, resp.body?.string(), "Groq")
            }

            val reader: BufferedReader = resp.body?.charStream()?.buffered()
                ?: throw ProviderError("Groq-Fehler: Leere Antwort erhalten.")

            // Bewusst eine manuelle Zeilen-Schleife statt reader.forEachLine {}:
            // der flow{}-Builder ist ein suspend-Lambda und emit() darf nur aus
            // suspend-Kontext aufgerufen werden. forEachLine nimmt aber ein
            // nicht-suspend Lambda entgegen, sodass wir dort nicht direkt
            // emit() aufrufen könnten (ein runBlocking{} als Workaround würde
            // den aufrufenden Dispatcher blockieren und Deadlock-Risiken
            // bergen) - daher hier readLine() in einer regulären while-Schleife.
            while (true) {
                val line = reader.readLine() ?: break
                if (!line.startsWith("data:")) continue
                val data = line.removePrefix("data:").trim()
                if (data == "[DONE]" || data.isEmpty()) continue
                try {
                    val obj = Json.parseToJsonElement(data).jsonObject
                    val choices = obj["choices"]?.jsonArray ?: continue
                    val delta = choices.firstOrNull()?.jsonObject?.get("delta")?.jsonObject
                    val content = delta?.get("content")?.jsonPrimitive?.contentOrNull()
                    if (!content.isNullOrEmpty()) {
                        emit(content)
                    }
                } catch (e: Exception) {
                    // Unparsebare Zeile ignorieren (z. B. Keep-Alive-Kommentare)
                }
            }
        }
    }

    private fun mapHttpError(code: Int, body: String?, provider: String): ProviderError {
        return when (code) {
            429 -> RateLimitError(
                "⏳ $provider-Rate-Limit erreicht (Free-Tier). Bitte kurz warten " +
                    "(meist 20-60 Sekunden) und erneut versuchen, oder ein kleineres Modell wählen."
            )
            401, 403 -> ProviderError(
                "🔑 $provider-API-Key ungültig oder abgelaufen. Bitte in den Einstellungen prüfen."
            )
            else -> ProviderError("$provider-API-Fehler (Status $code): ${body?.take(200) ?: "unbekannt"}")
        }
    }
}

private fun JsonPrimitive?.contentOrNull(): String? = try {
    this?.content
} catch (e: Exception) {
    null
}
