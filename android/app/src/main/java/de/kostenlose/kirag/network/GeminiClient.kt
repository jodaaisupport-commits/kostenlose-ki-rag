package de.kostenlose.kirag.network

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.Json
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
 * Google-Gemini-Client (Generative Language REST-API, SSE-Streaming via
 * `streamGenerateContent?alt=sse`). Kotlin-Entsprechung von
 * llm/providers.py::stream_gemini_completion.
 */
class GeminiClient(private val httpClient: OkHttpClient = GroqClient.defaultHttpClient()) {

    companion object {
        private const val BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
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
                "Kein Google API-Key gefunden. Bitte in den Einstellungen eingeben " +
                    "(kostenlos erhältlich unter https://aistudio.google.com/apikey)."
            )
        }

        val contents = buildJsonArray {
            for (m in history) {
                add(buildJsonObject {
                    put("role", if (m.role == ChatRole.USER) "user" else "model")
                    put("parts", buildJsonArray {
                        add(buildJsonObject { put("text", m.content) })
                    })
                })
            }
            add(buildJsonObject {
                put("role", "user")
                put("parts", buildJsonArray {
                    add(buildJsonObject { put("text", userMessage) })
                })
            })
        }

        val bodyJson = buildJsonObject {
            put("system_instruction", buildJsonObject {
                put("parts", buildJsonArray {
                    add(buildJsonObject { put("text", systemPrompt) })
                })
            })
            put("contents", contents)
            put("generationConfig", buildJsonObject {
                put("temperature", temperature)
            })
        }

        val url = "$BASE_URL/$model:streamGenerateContent?alt=sse&key=$apiKey"
        val request = Request.Builder()
            .url(url)
            .header("Content-Type", "application/json")
            .post(bodyJson.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = try {
            httpClient.newCall(request).execute()
        } catch (e: Exception) {
            throw ProviderError("🌐 Verbindung zu Gemini fehlgeschlagen. Bitte Internetverbindung prüfen.")
        }

        response.use { resp ->
            if (!resp.isSuccessful) {
                throw mapHttpError(resp.code, resp.body?.string())
            }

            val reader: BufferedReader = resp.body?.charStream()?.buffered()
                ?: throw ProviderError("Gemini-Fehler: Leere Antwort erhalten.")

            while (true) {
                val line = reader.readLine() ?: break
                if (!line.startsWith("data:")) continue
                val data = line.removePrefix("data:").trim()
                if (data.isEmpty()) continue
                try {
                    val obj = Json.parseToJsonElement(data).jsonObject
                    val candidates = obj["candidates"]?.jsonArray ?: continue
                    val parts = candidates.firstOrNull()?.jsonObject
                        ?.get("content")?.jsonObject
                        ?.get("parts")?.jsonArray ?: continue
                    for (part in parts) {
                        val text = part.jsonObject["text"]?.jsonPrimitive?.contentOrNull()
                        if (!text.isNullOrEmpty()) emit(text)
                    }
                } catch (e: Exception) {
                    // Unparsebare Zeile ignorieren
                }
            }
        }
    }

    private fun mapHttpError(code: Int, body: String?): ProviderError {
        return when (code) {
            429 -> RateLimitError(
                "⏳ Gemini-Rate-Limit erreicht (Free-Tier). Bitte kurz warten " +
                    "(meist 20-60 Sekunden) und erneut versuchen, oder ein anderes Modell wählen."
            )
            401, 403 -> ProviderError(
                "🔑 Google-API-Key ungültig oder ohne Berechtigung. Bitte in den Einstellungen prüfen " +
                    "(https://aistudio.google.com/apikey)."
            )
            else -> ProviderError("Gemini-API-Fehler (Status $code): ${body?.take(200) ?: "unbekannt"}")
        }
    }
}

private fun kotlinx.serialization.json.JsonPrimitive?.contentOrNull(): String? = try {
    this?.content
} catch (e: Exception) {
    null
}
