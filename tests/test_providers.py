"""Tests für llm.providers: Fehler-Mapping (Rate-Limits, Auth) und
Embedding-Batching, jeweils ohne echte Netzwerk-/API-Aufrufe (gemockt).
"""
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.providers import (
    GEMINI_EMBEDDING_BATCH_SIZE,
    ProviderError,
    RateLimitError,
    get_gemini_module,
    get_groq_client,
    make_gemini_embed_fn,
    stream_groq_completion,
)


def test_get_groq_client_without_key_raises_provider_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        get_groq_client(api_key=None)


def test_get_gemini_module_without_key_raises_provider_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        get_gemini_module(api_key=None)


def test_groq_rate_limit_is_mapped_to_rate_limit_error(monkeypatch):
    import groq as groq_sdk

    class FakeCompletions:
        def create(self, **kwargs):
            resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://x"))
            raise groq_sdk.RateLimitError("rate limited", response=resp, body=None)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    with pytest.raises(RateLimitError):
        list(
            stream_groq_completion(
                client=FakeClient(),
                model="llama-3.3-70b-versatile",
                system_prompt="sys",
                history=[],
                user_message="hallo",
            )
        )


def test_groq_auth_error_is_mapped_to_provider_error(monkeypatch):
    import groq as groq_sdk

    class FakeCompletions:
        def create(self, **kwargs):
            resp = httpx.Response(status_code=401, request=httpx.Request("POST", "http://x"))
            raise groq_sdk.AuthenticationError("bad key", response=resp, body=None)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    with pytest.raises(ProviderError) as exc_info:
        list(
            stream_groq_completion(
                client=FakeClient(),
                model="llama-3.3-70b-versatile",
                system_prompt="sys",
                history=[],
                user_message="hallo",
            )
        )
    assert "Groq-API-Key" in str(exc_info.value)
    # Auth-Fehler ist kein Rate-Limit-Fehler
    assert not isinstance(exc_info.value, RateLimitError)


def test_gemini_resource_exhausted_is_mapped_to_rate_limit_error():
    from google.api_core import exceptions as gexc

    class FakeModel:
        def start_chat(self, history):
            return self

        def send_message(self, msg, stream=True):
            raise gexc.ResourceExhausted("quota exceeded")

    class FakeGenAI:
        def GenerativeModel(self, **kwargs):
            return FakeModel()

    from llm.providers import stream_gemini_completion

    with pytest.raises(RateLimitError):
        list(
            stream_gemini_completion(
                genai_module=FakeGenAI(),
                model="gemini-1.5-flash",
                system_prompt="sys",
                history=[],
                user_message="hallo",
            )
        )


def test_gemini_embed_fn_batches_requests():
    """Stellt sicher, dass embed_fn Texte in Batches sendet statt pro Text
    einen eigenen API-Call abzusetzen (Performance-/Rate-Limit-Fix)."""
    call_log = []

    class FakeGenAI:
        def embed_content(self, model, content, task_type):
            # content ist bei Batch-Aufrufen eine Liste
            assert isinstance(content, list)
            call_log.append(len(content))
            return {"embedding": [[0.1, 0.2, 0.3] for _ in content]}

    embed_fn = make_gemini_embed_fn(FakeGenAI())
    texts = [f"text {i}" for i in range(GEMINI_EMBEDDING_BATCH_SIZE + 10)]
    vectors = embed_fn(texts)

    assert len(vectors) == len(texts)
    # Es sollten genau 2 Batches gesendet worden sein (100 + 10)
    assert call_log == [GEMINI_EMBEDDING_BATCH_SIZE, 10]


def test_gemini_embed_fn_empty_list_returns_empty():
    embed_fn = make_gemini_embed_fn(genai_module=None)
    assert embed_fn([]) == []
