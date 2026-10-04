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


class ProviderError(RuntimeError):
    pass


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
    except Exception as e:
        raise ProviderError(f"Groq-Fehler: {e}")


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
    except Exception as e:
        raise ProviderError(f"Gemini-Fehler: {e}")


def make_gemini_embed_fn(genai_module):
    def embed_fn(texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            resp = genai_module.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=t,
                task_type="retrieval_document",
            )
            results.append(resp["embedding"])
        return results

    return embed_fn
