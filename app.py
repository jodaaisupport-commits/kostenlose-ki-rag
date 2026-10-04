"""
Kostenlose KI-Chat-App mit Dokumenten-RAG
===========================================
Gradio-UI · Groq / Gemini Free-Tier-APIs · PDF/TXT-Upload
TF-IDF- oder Gemini-Embedding-Suche · persistenter Index-Cache

Start: python app.py
"""
from __future__ import annotations

import os
import traceback
from typing import List, Optional

import gradio as gr
from dotenv import load_dotenv

from rag.index_store import RagIndex, build_or_load_index
from llm.providers import (
    GEMINI_MODELS,
    GROQ_MODELS,
    ProviderError,
    get_gemini_module,
    get_groq_client,
    make_gemini_embed_fn,
    stream_gemini_completion,
    stream_groq_completion,
)

load_dotenv()

APP_TITLE = "🆓 Kostenlose KI-RAG-Chat-App"
DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher, präziser Assistent. "
    "Wenn Kontext aus Dokumenten bereitgestellt wird, nutze ihn vorrangig, "
    "um die Frage zu beantworten. Wenn die Antwort nicht im Kontext steht, "
    "sage das ehrlich, anstatt zu raten. Antworte auf Deutsch, außer der "
    "Nutzer schreibt in einer anderen Sprache."
)

CUSTOM_CSS = """
#title-row {text-align: center; margin-bottom: 0.5rem;}
#title-row h1 {margin-bottom: 0.2rem;}
.status-box textarea {font-family: monospace; font-size: 0.85rem;}
footer {display: none !important;}
"""


# --------------------------------------------------------------------------
# App-State-Helfer
# --------------------------------------------------------------------------

def build_context_snippet(index: Optional[RagIndex], query: str, embed_fn=None, top_k: int = 4) -> str:
    if index is None or not index.is_ready:
        return ""
    try:
        results = index.search(query, top_k=top_k, embed_fn=embed_fn)
    except Exception:
        return ""
    if not results:
        return ""

    parts = []
    for chunk, score in results:
        parts.append(f"[Quelle: {chunk.source} | Relevanz: {score:.2f}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def format_sources_markdown(index: Optional[RagIndex]) -> str:
    if index is None or not index.is_ready:
        return "_Noch keine Dokumente indiziert._"
    unique_sources = sorted(set(index.source_names))
    lines = [f"- 📄 **{s}**" for s in unique_sources]
    lines.append(f"\n**{len(index.chunks)}** Text-Abschnitte indiziert · Modus: `{index.embedding_mode}`")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Callback: Dokumente indizieren
# --------------------------------------------------------------------------

def handle_index_build(
    files,
    embedding_mode: str,
    chunk_size: int,
    overlap: int,
    google_api_key: str,
    state_index,
    progress=gr.Progress(track_tqdm=False),
):
    if not files:
        return state_index, "⚠️ Bitte zuerst mindestens eine Datei hochladen.", format_sources_markdown(state_index)

    file_paths = [f.name if hasattr(f, "name") else f for f in files]

    log_lines: List[str] = []

    def progress_cb(msg: str):
        log_lines.append(msg)
        try:
            progress(0, desc=msg)
        except Exception:
            pass

    embed_fn = None
    try:
        if embedding_mode == "gemini":
            genai_module = get_gemini_module(google_api_key or None)
            embed_fn = make_gemini_embed_fn(genai_module)

        index = build_or_load_index(
            file_paths=file_paths,
            embedding_mode=embedding_mode,
            chunk_size=int(chunk_size),
            overlap=int(overlap),
            embed_fn=embed_fn,
            progress_cb=progress_cb,
        )
        log_lines.append("✅ Index erfolgreich erstellt / geladen.")
        return index, "\n".join(log_lines), format_sources_markdown(index)
    except ProviderError as e:
        log_lines.append(f"❌ {e}")
        return state_index, "\n".join(log_lines), format_sources_markdown(state_index)
    except Exception as e:
        log_lines.append(f"❌ Fehler beim Indizieren: {e}")
        log_lines.append(traceback.format_exc(limit=2))
        return state_index, "\n".join(log_lines), format_sources_markdown(state_index)


def handle_clear_index():
    return None, "🗑️ Index zurückgesetzt.", format_sources_markdown(None)


# --------------------------------------------------------------------------
# Callback: Chat
# --------------------------------------------------------------------------

def handle_chat(
    message: str,
    chat_history: List[dict],
    provider: str,
    groq_model: str,
    gemini_model: str,
    system_prompt: str,
    temperature: float,
    use_rag: bool,
    top_k: int,
    groq_api_key: str,
    google_api_key: str,
    state_index,
):
    if not message or not message.strip():
        yield chat_history, ""
        return

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    yield chat_history, ""

    # RAG-Kontext ggf. einbinden
    augmented_message = message
    context_note = ""
    if use_rag and state_index is not None and state_index.is_ready:
        embed_fn = None
        try:
            if state_index.embedding_mode == "gemini":
                genai_module = get_gemini_module(google_api_key or None)
                embed_fn = make_gemini_embed_fn(genai_module)
            context = build_context_snippet(state_index, message, embed_fn=embed_fn, top_k=int(top_k))
            if context:
                augmented_message = (
                    f"Nutze den folgenden Kontext aus hochgeladenen Dokumenten, um die Frage zu beantworten.\n\n"
                    f"=== KONTEXT ===\n{context}\n=== ENDE KONTEXT ===\n\n"
                    f"Frage: {message}"
                )
                context_note = f"\n\n_(📚 {min(int(top_k), len(state_index.chunks))} Kontext-Abschnitte verwendet)_"
        except ProviderError as e:
            chat_history[-1]["content"] = f"❌ {e}"
            yield chat_history, ""
            return
        except Exception:
            pass  # bei Fehler ohne Kontext fortfahren

    history_for_llm = chat_history[:-2]  # ohne aktuelle leere Antwort & aktuelle Frage

    try:
        if provider == "Groq (empfohlen, sehr schnell)":
            client = get_groq_client(groq_api_key or None)
            stream = stream_groq_completion(
                client=client,
                model=groq_model,
                system_prompt=system_prompt,
                history=history_for_llm,
                user_message=augmented_message,
                temperature=float(temperature),
            )
        else:
            genai_module = get_gemini_module(google_api_key or None)
            stream = stream_gemini_completion(
                genai_module=genai_module,
                model=gemini_model,
                system_prompt=system_prompt,
                history=history_for_llm,
                user_message=augmented_message,
                temperature=float(temperature),
            )

        full_response = ""
        for delta in stream:
            full_response += delta
            chat_history[-1]["content"] = full_response
            yield chat_history, ""

        if context_note:
            chat_history[-1]["content"] = full_response + context_note
            yield chat_history, ""

    except ProviderError as e:
        chat_history[-1]["content"] = f"❌ {e}"
        yield chat_history, ""
    except Exception as e:
        chat_history[-1]["content"] = f"❌ Unerwarteter Fehler: {e}"
        yield chat_history, ""


def handle_retry_clear():
    return []


# --------------------------------------------------------------------------
# UI-Aufbau
# --------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        index_state = gr.State(None)

        with gr.Row(elem_id="title-row"):
            gr.Markdown(
                f"# {APP_TITLE}\n"
                "Chatte kostenlos mit **Groq** oder **Google Gemini** (Free-Tier) "
                "und stelle Fragen zu eigenen **PDF/TXT-Dokumenten** (RAG)."
            )

        with gr.Row():
            # ---------------- Linke Spalte: Einstellungen ----------------
            with gr.Column(scale=1, min_width=340):
                with gr.Accordion("🔑 API-Schlüssel (kostenlos erhältlich)", open=True):
                    gr.Markdown(
                        "- **Groq**: [console.groq.com/keys](https://console.groq.com/keys)\n"
                        "- **Google Gemini**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)\n\n"
                        "_Alternativ als Umgebungsvariable `GROQ_API_KEY` / `GOOGLE_API_KEY` setzen._"
                    )
                    groq_api_key = gr.Textbox(
                        label="Groq API-Key",
                        type="password",
                        placeholder="gsk_...",
                        value=os.environ.get("GROQ_API_KEY", ""),
                    )
                    google_api_key = gr.Textbox(
                        label="Google (Gemini) API-Key",
                        type="password",
                        placeholder="AIza...",
                        value=os.environ.get("GOOGLE_API_KEY", ""),
                    )

                with gr.Accordion("🤖 Modell-Einstellungen", open=True):
                    provider = gr.Radio(
                        choices=["Groq (empfohlen, sehr schnell)", "Google Gemini"],
                        value="Groq (empfohlen, sehr schnell)",
                        label="LLM-Anbieter",
                    )
                    groq_model = gr.Dropdown(
                        choices=GROQ_MODELS, value=GROQ_MODELS[0], label="Groq-Modell"
                    )
                    gemini_model = gr.Dropdown(
                        choices=GEMINI_MODELS, value=GEMINI_MODELS[0], label="Gemini-Modell"
                    )
                    temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="Temperature")
                    system_prompt = gr.Textbox(
                        label="System-Prompt",
                        value=DEFAULT_SYSTEM_PROMPT,
                        lines=4,
                    )

                with gr.Accordion("📚 Dokumenten-RAG", open=True):
                    use_rag = gr.Checkbox(label="RAG-Kontext im Chat verwenden", value=True)
                    files = gr.File(
                        label="PDF / TXT / MD hochladen",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md"],
                    )
                    embedding_mode = gr.Radio(
                        choices=[("TF-IDF (lokal, kostenlos, schnell)", "tfidf"),
                                 ("Gemini-Embeddings (benötigt Google-Key)", "gemini")],
                        value="tfidf",
                        label="Such-/Embedding-Modus",
                    )
                    with gr.Row():
                        chunk_size = gr.Number(value=220, label="Chunk-Größe (Wörter)", precision=0)
                        overlap = gr.Number(value=40, label="Überlappung (Wörter)", precision=0)
                    top_k = gr.Slider(1, 10, value=4, step=1, label="Anzahl Kontext-Abschnitte (Top-K)")

                    with gr.Row():
                        build_btn = gr.Button("📥 Dokumente indizieren", variant="primary")
                        clear_index_btn = gr.Button("🗑️ Index löschen")

                    index_status = gr.Textbox(
                        label="Status", lines=4, interactive=False, elem_classes=["status-box"]
                    )
                    sources_md = gr.Markdown(format_sources_markdown(None))

            # ---------------- Rechte Spalte: Chat ----------------
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=560,
                    avatar_images=(None, None),
                    buttons=["copy"],
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Stelle eine Frage ... (Enter zum Senden)",
                        scale=5,
                        show_label=False,
                        container=False,
                    )
                    send_btn = gr.Button("Senden", variant="primary", scale=1)
                with gr.Row():
                    clear_chat_btn = gr.Button("🧹 Chat leeren")

        # ---------------- Events ----------------
        build_btn.click(
            fn=handle_index_build,
            inputs=[files, embedding_mode, chunk_size, overlap, google_api_key, index_state],
            outputs=[index_state, index_status, sources_md],
        )
        clear_index_btn.click(
            fn=handle_clear_index,
            inputs=[],
            outputs=[index_state, index_status, sources_md],
        )

        chat_inputs = [
            msg_box,
            chatbot,
            provider,
            groq_model,
            gemini_model,
            system_prompt,
            temperature,
            use_rag,
            top_k,
            groq_api_key,
            google_api_key,
            index_state,
        ]
        chat_outputs = [chatbot, msg_box]

        msg_box.submit(fn=handle_chat, inputs=chat_inputs, outputs=chat_outputs)
        send_btn.click(fn=handle_chat, inputs=chat_inputs, outputs=chat_outputs)
        clear_chat_btn.click(fn=handle_retry_clear, inputs=[], outputs=[chatbot])

        gr.Markdown(
            "---\n"
            "💡 **Tipp:** TF-IDF funktioniert komplett lokal ohne zusätzlichen API-Key. "
            "Für semantischere Suche kannst du auf Gemini-Embeddings umschalten (Google-Key nötig)."
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CUSTOM_CSS,
    )
