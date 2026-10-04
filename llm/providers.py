"""LLM-Provider-Abstraktion für Groq und Google Gemini (beide Free-Tier-fähig)."""
from __future__ import annotations

import os
from typing import Generator, List, Optional

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# Gemini erlaubt pro batch_embed_contents-Aufruf maximal 100 Inhalte.
GEMINI_EMBEDDING_BATCH_SIZE = 100


class ProviderError(RuntimeError):
    """Allgemeiner, dem Nutzer anzuzeigender Fehler eines LLM-Providers."""


class RateLimitError(ProviderError):
    """Spezifischer Fehler für erreichte Free-Tier-Rate-Limits."""


def get_groq_client(api_key: Optional[str] = None):
    from groq import Groq

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ProviderError(
            "Kein Groq API-Key gefunden. Bitte GROQ_API_KEY setzen oder im UI eingeben "
            "(kostenlos erhältlich unter https://console.groq.com/keys)."
        )
    return Groq(api_key=key)


def get_gemini_module(api_key: Optional[str] = None):
    import google.generativeai as genai

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ProviderError(
            "Kein Google API-Key gefunden. Bitte GOOGLE_API_KEY setzen oder im UI eingeben "
            "(kostenlos erhältlich unter https://aistudio.google.com/apikey)."
        )
    genai.configure(api_key=key)
    return genai


def _friendly_groq_error(e: Exception) -> ProviderError:
    """Wandelt Groq-SDK-Fehler in verständliche, deutschsprachige Meldungen um."""
    try:
        import groq as groq_sdk

        if isinstance(e, groq_sdk.RateLimitError):
            return RateLimitError(
                "⏳ Groq-Rate-Limit erreicht (Free-Tier). Bitte kurz warten "
                "(meist 20-60 Sekunden) und erneut versuchen, oder ein kleineres "
                "Modell wählen."
            )
        if isinstance(e, groq_sdk.AuthenticationError):
            return ProviderError(
                "🔑 Groq-API-Key ungültig oder abgelaufen. Bitte im UI-Feld prüfen "
                "(https://console.groq.com/keys)."
            )
        if isinstance(e, groq_sdk.APIConnectionError):
            return ProviderError("🌐 Verbindung zu Groq fehlgeschlagen. Bitte Internetverbindung prüfen.")
        if isinstance(e, groq_sdk.APIStatusError):
            return ProviderError(f"Groq-API-Fehler (Status {e.status_code}): {e.message}")
    except ImportError:
        pass
    return ProviderError(f"Groq-Fehler: {e}")


def _friendly_gemini_error(e: Exception) -> ProviderError:
    """Wandelt Gemini-/google-api_core-Fehler in verständliche Meldungen um."""
    try:
        from google.api_core import exceptions as gexc

        if isinstance(e, (gexc.ResourceExhausted, gexc.TooManyRequests)):
            return RateLimitError(
                "⏳ Gemini-Rate-Limit erreicht (Free-Tier). Bitte kurz warten "
                "(meist 20-60 Sekunden) und erneut versuchen, oder ein anderes "
                "Modell wählen."
            )
        if isinstance(e, gexc.Unauthenticated) or isinstance(e, gexc.PermissionDenied):
            return ProviderError(
                "🔑 Google-API-Key ungültig oder ohne Berechtigung. Bitte im UI-Feld prüfen "
                "(https://aistudio.google.com/apikey)."
            )
        if isinstance(e, gexc.DeadlineExceeded):
            return ProviderError("⏱️ Zeitüberschreitung bei Gemini. Bitte erneut versuchen.")
        if isinstance(e, gexc.GoogleAPICallError):
            return ProviderError(f"Gemini-API-Fehler: {e.message if hasattr(e, 'message') else e}")
    except ImportError:
        pass
    return ProviderError(f"Gemini-Fehler: {e}")


def stream_groq_completion(
    client,
    model: str,
    system_prompt: str,
    history: List[dict],
    user_message: str,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except ProviderError:
        raise
    except Exception as e:
        raise _friendly_groq_error(e)


def stream_gemini_completion(
    genai_module,
    model: str,
    system_prompt: str,
    history: List[dict],
    user_message: str,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    try:
        gmodel = genai_module.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config={"temperature": temperature},
        )
        # Konvertiere OpenAI-Style-History in Gemini-Chat-Format
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = gmodel.start_chat(history=gemini_history)
        response = chat.send_message(user_message, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except ProviderError:
        raise
    except Exception as e:
        raise _friendly_gemini_error(e)


def make_gemini_embed_fn(genai_module):
    """Erzeugt eine Embedding-Funktion, die Gemini's native Batch-API nutzt
    (bis zu GEMINI_EMBEDDING_BATCH_SIZE Texte pro Request) statt pro Text
    einen einzelnen API-Call abzusetzen.
    """

    def embed_fn(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        try:
            for i in range(0, len(texts), GEMINI_EMBEDDING_BATCH_SIZE):
                batch = texts[i : i + GEMINI_EMBEDDING_BATCH_SIZE]
                resp = genai_module.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document",
                )
                results.extend(resp["embedding"])
        except ProviderError:
            raise
        except Exception as e:
            raise _friendly_gemini_error(e)
        return results

    return embed_fn
