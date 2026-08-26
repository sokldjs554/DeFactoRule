"""Native-PDF vs scanned-image intake with an OCR fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pymupdf

from app.document_ai.models import DocumentText, PageText
from app.document_ai.ocr import TesseractOCR
from app.extraction.casebook import MIN_TEXT_HEALTH, text_health

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})


def detect_mode(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "ocr"
    if suffix != ".pdf":
        raise ValueError(f"unsupported document type: {path.suffix}")
    doc = pymupdf.open(path)
    text = "".join(page.get_text() for page in doc)
    if len("".join(text.split())) < 20 or text_health(text) < MIN_TEXT_HEALTH:
        return "ocr"
    return "native"


def _render_page(page: pymupdf.Page, dpi: int) -> bytes:
    return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")


def read_document(
    path: Path,
    ocr: Optional[TesseractOCR] = None,
    dpi: int = 220,
    max_pages: Optional[int] = None,
) -> DocumentText:
    """Read a PDF/image, invoking OCR only when native text is unavailable/unhealthy."""
    mode = detect_mode(path)
    if mode == "native":
        doc = pymupdf.open(path)
        limit = min(doc.page_count, max_pages) if max_pages else doc.page_count
        pages = [
            PageText(page=i + 1, text=doc[i].get_text().strip(), source="native")
            for i in range(limit)
        ]
        text = "\n".join(p.text for p in pages)
        return DocumentText(
            mode="native",
            engine="pymupdf-native-text",
            pages=pages,
            text_health=text_health(text),
        )

    engine = ocr or TesseractOCR()
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        output = engine.recognize(path.read_bytes())
        pages = [PageText(page=1, text=output.text, source="ocr")]
        return DocumentText(
            mode="ocr",
            engine=output.engine,
            pages=pages,
            text_health=text_health(output.text),
        )

    doc = pymupdf.open(path)
    limit = min(doc.page_count, max_pages) if max_pages else doc.page_count
    pages = []
    engine_name = ""
    for i in range(limit):
        output = engine.recognize(_render_page(doc[i], dpi=dpi))
        engine_name = output.engine
        pages.append(PageText(page=i + 1, text=output.text, source="ocr"))
    text = "\n".join(p.text for p in pages)
    return DocumentText(
        mode="ocr",
        engine=engine_name,
        pages=pages,
        text_health=text_health(text),
    )
