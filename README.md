# 🆓 Kostenlose KI-Chat-App mit Dokumenten-RAG

Eine vollständig kostenlose KI-Chat-Anwendung mit **Gradio-UI**, die über die
**Free-Tier-APIs von Groq und Google Gemini** läuft und eigene Dokumente
(PDF/TXT/MD) per **RAG (Retrieval-Augmented Generation)** durchsuchbar macht.

## ✨ Features

- 💬 **Chat-Interface** mit Streaming-Antworten (Gradio Chatbot)
- 🤖 **Zwei kostenlose LLM-Anbieter**: [Groq](https://console.groq.com/keys) (Llama 3.3, sehr schnell) und [Google Gemini](https://aistudio.google.com/apikey)
- 📄 **Dokumenten-Upload**: PDF, TXT und Markdown
- 🔍 **Zwei Such-Modi**:
  - **TF-IDF** – läuft komplett lokal, kein zusätzlicher API-Key nötig
  - **Gemini-Embeddings** – semantischere Suche (benötigt Google-Key)
- 💾 **Persistenter Index-Cache**: einmal indizierte Dokumente werden lokal
  gecacht (joblib) und beim nächsten Start sofort wiederverwendet
- 🔑 API-Keys können im UI eingegeben oder als Umgebungsvariable gesetzt werden

## 🚀 Schnellstart

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# (Optional) API-Keys als Umgebungsvariablen setzen
cp .env.example .env
# .env bearbeiten und Keys eintragen

# App starten
python app.py
```

Die App läuft anschließend standardmäßig auf **http://0.0.0.0:7860**.

## 🔑 Kostenlose API-Keys

| Anbieter | Link | Hinweis |
|---|---|---|
| Groq | https://console.groq.com/keys | Sehr schnelle Inferenz, großzügiges Free-Tier |
| Google Gemini | https://aistudio.google.com/apikey | Auch für Embeddings nutzbar |

## 📁 Projektstruktur

```
app.py                  # Gradio-UI & App-Logik
rag/
  loader.py             # PDF/TXT-Textextraktion
  chunker.py            # Text-Chunking mit Overlap
  index_store.py        # TF-IDF/Gemini-Index + persistenter Cache
llm/
  providers.py          # Groq- & Gemini-Chat-/Embedding-Anbindung
cache/                  # Persistierte Indizes (joblib, gitignored)
uploads/                # Temporäre Uploads
```

## 🛠️ Nutzung

1. API-Key(s) im linken Panel eingeben (oder per `.env`)
2. Dokumente hochladen und auf **„Dokumente indizieren"** klicken
3. Such-Modus (TF-IDF oder Gemini-Embeddings) wählen
4. Im Chat Fragen stellen – relevanter Dokumenten-Kontext wird automatisch
   eingebunden, wenn **„RAG-Kontext verwenden"** aktiviert ist

## Lizenz

Siehe [LICENSE](LICENSE).
