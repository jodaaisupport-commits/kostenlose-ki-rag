---
title: Kostenlose KI-Chat-App mit Dokumenten-RAG
emoji: 🆓
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
---

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
  gecacht (joblib, inkl. Chunk-Parameter im Cache-Key) und beim nächsten Start
  sofort wiederverwendet
- 🔑 API-Keys können im UI eingegeben oder als Umgebungsvariable gesetzt werden
- 📌 **Transparente Quellenangaben**: jede RAG-Antwort zeigt aufklappbar, welche
  Dokumenten-Abschnitte (inkl. Seitenzahl bei PDFs) und Relevanz-Scores
  tatsächlich verwendet wurden
- ⏹️ **Stopp-Button** zum Abbrechen einer laufenden Antwort
- 💾 **Chat-Export** als Markdown-Datei
- 🎭 **System-Prompt-Vorlagen** (Standard, Strikt/nur-aus-Dokument, Zusammenfassen, Kreativ)
- 🛡️ **Robustheit**: Datei-/Größen-/Seitenlimits gegen versehentliche Riesen-Uploads,
  verständliche deutsche Fehlermeldungen bei Rate-Limits (Groq/Gemini Free-Tier)
- ✅ **35 automatisierte Tests** (pytest) für Chunking, Indexing, Provider-Fehlerbehandlung

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

## ☁️ Deployment auf Hugging Face Spaces

1. Auf [Hugging Face Spaces](https://huggingface.co/new-space) einen neuen Space erstellen und **Gradio** als SDK wählen.
2. Den Inhalt dieses Repositorys in den Space übertragen. Die Space-Konfiguration ist bereits in diesem README hinterlegt; `requirements.txt` installiert die Abhängigkeiten.
3. Optional API-Keys unter **Settings → Variables and secrets** als `GROQ_API_KEY` und/oder `GOOGLE_API_KEY` hinterlegen. Alternativ können die Schlüssel direkt in der App eingegeben werden. Schlüssel niemals in Repository-Dateien eintragen.

Die App startet auf dem von Spaces bereitgestellten Port. Hochgeladene Dokumente und der Index-Cache liegen im Space-Dateisystem und sind ohne aktivierten persistenten Speicher nicht dauerhaft gespeichert.

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
tests/                  # Pytest-Testsuite (rag/, llm/, app.py)
```

## 🛠️ Nutzung

1. API-Key(s) im linken Panel eingeben (oder per `.env`)
2. Dokumente hochladen und auf **„Dokumente indizieren"** klicken
3. Such-Modus (TF-IDF oder Gemini-Embeddings) wählen
4. Im Chat Fragen stellen – relevanter Dokumenten-Kontext wird automatisch
   eingebunden, wenn **„RAG-Kontext verwenden"** aktiviert ist; die verwendeten
   Quellen lassen sich unter jeder Antwort aufklappen

## 🧪 Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Lizenz

Siehe [LICENSE](LICENSE).
