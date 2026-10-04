"""Dokumenten-Loader: extrahiert reinen Text aus PDF- und Textdateien."""
from __future__ import annotations

import os
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Sicherheitsgrenzen gegen versehentliche Riesen-Uploads, die die App
# (Speicher, Rechenzeit) blockieren oder zum Absturz bringen könnten.
MAX_FILE_SIZE_MB = 25
MAX_PDF_PAGES = 1000


class DocumentTooLargeError(ValueError):
    """Wird ausgelöst, wenn eine Datei die konfigurierten Limits überschreitet."""


def extract_text(file_path: str) -> str:
    """Extrahiert den Text-Inhalt einer Datei (PDF, TXT, MD)."""
    _check_file_size(file_path)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_pdf_text(file_path)
    elif ext in (".txt", ".md"):
        return _extract_plain_text(file_path)
    else:
        raise ValueError(
            f"Nicht unterstütztes Dateiformat: {ext}. "
            f"Unterstützt werden: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _check_file_size(file_path: str) -> None:
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return
    if size_mb > MAX_FILE_SIZE_MB:
        raise DocumentTooLargeError(
            f"Datei ist {size_mb:.1f} MB groß und überschreitet das Limit von "
            f"{MAX_FILE_SIZE_MB} MB."
        )


def _extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    if num_pages > MAX_PDF_PAGES:
        raise DocumentTooLargeError(
            f"PDF hat {num_pages} Seiten und überschreitet das Limit von "
            f"{MAX_PDF_PAGES} Seiten."
        )
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages_text.append(f"[Seite {i + 1}]\n{text}")
    return "\n\n".join(pages_text)


def _extract_plain_text(file_path: str) -> str:
    # Robustes Einlesen mit Fallback-Encoding
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Letzter Fallback: Fehler ignorieren
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
