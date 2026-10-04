# Kostenlose KI-RAG — Android App

Native Android-Portierung der Python/Gradio-App "Kostenlose KI-RAG" (kostenloser
KI-Chat mit Dokumenten-RAG). Gebaut mit Kotlin + Jetpack Compose (Material 3),
MVVM (`ViewModel` + `StateFlow`).

## Funktionsumfang

- **Chat** mit Groq (OpenAI-kompatible Chat-Completions, SSE-Streaming) oder
  Google Gemini (`streamGenerateContent?alt=sse`), beide über kostenlose
  Free-Tier-APIs.
- **Dokumenten-RAG**: PDF/TXT hochladen (Android `OpenMultipleDocuments`),
  Text-Extraktion on-device via PDFBox-Android, Chunking (Wort-basiert mit
  Seitenmarker-Tracking wie im Python-Original).
- **Suche**: Reine Kotlin-Implementierung von TF-IDF (Unigramm+Bigramm,
  sublineare TF, smooth IDF, L2-normalisierte Sparse-Vektoren,
  Cosinus-Similarity) — läuft komplett offline auf dem Gerät, keine
  Cloud-Embedding-Abhängigkeit.
- **Index-Cache**: JSON-Cache (kotlinx.serialization) im App-Cache-Verzeichnis,
  Cache-Key inkl. Dateihashes + Chunk-Size/Overlap (identische Logik wie
  `rag/index_store.py`).
- **System-Prompt-Vorlagen**: Standard / Strikt / Zusammenfassen / Kreativ
  (identisch zur Python-Version übernommen).
- **Sichere API-Key-Ablage**: `EncryptedSharedPreferences`
  (AES-256-GCM via Android Keystore).
- **History-Bloat-Fix**: Wie im Python-Original wird der RAG-Kontext nur für
  den aktuellen API-Call injiziert, nicht dauerhaft in den Chatverlauf
  geschrieben; Fehler-/Status-Nachrichten werden vor dem Senden an den
  Provider aus dem Verlauf gefiltert.

## Bewusste Vereinfachung ggü. der Python-Version

Der **Gemini-Embedding-Suchmodus** wurde für die Android-App bewusst
weggelassen. Die App nutzt ausschließlich die On-Device-TF-IDF-Suche:

- funktioniert vollständig offline (keine zusätzliche API-Abhängigkeit für
  die Suche selbst),
- vermeidet einen zweiten, separaten API-Call-Pfad nur für Embeddings auf
  dem Mobilgerät,
- spart API-Kontingent/Kosten beim Indexieren größerer Dokumentenmengen.

Alle anderen Kernfunktionen (Chat-Provider, Chunking, Caching, Prompts,
History-Fix) sind 1:1 aus der Python-Version übernommen.

## Build

Voraussetzungen: JDK 17+, Android SDK (API 34, Build-Tools 34.0.0).

```bash
cd android
export ANDROID_HOME=/pfad/zum/android-sdk
./gradlew assembleDebug
```

Die Debug-APK liegt danach unter:
`android/app/build/outputs/apk/debug/app-debug.apk`

> Hinweis für ressourcenbeschränkte Build-Umgebungen (≤2 GB RAM): In
> `gradle.properties` ist die Gradle-/Kotlin-Daemon-Nutzung bewusst
> deaktiviert (`org.gradle.daemon=false`, `org.gradle.parallel=false`,
> `org.gradle.workers.max=1`, reduziertes `-Xmx`), um OOM-Abbrüche beim
> Dexing/R8-Schritt zu vermeiden. Bei ausreichend RAM können diese Werte
> für schnellere Builds wieder erhöht werden.

## Installation auf dem Gerät

1. APK auf das Android-Gerät übertragen (min. Android 8.0 / API 26).
2. "Installation aus unbekannten Quellen" erlauben.
3. APK installieren und öffnen.
4. In den Einstellungen den Groq- und/oder Google-API-Key eintragen
   (kostenlose Keys z. B. über console.groq.com bzw. aistudio.google.com).

## Projektstruktur

```
android/
  app/src/main/java/de/kostenlose/kirag/
    ChatViewModel.kt         # UI-State, Chat-Logik, History-Fix
    MainActivity.kt
    data/SecurePrefs.kt      # verschlüsselte API-Key-Ablage
    network/                 # GroqClient, GeminiClient, ChatModels, Fehler
    rag/                     # Chunker, DocumentLoader, TfIdfIndex, RagIndexManager, Models
    ui/                      # ChatScreen, DocumentsScreen, SettingsScreen, Theme
```
