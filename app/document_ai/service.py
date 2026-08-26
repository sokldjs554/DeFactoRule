"""End-to-end Document AI intake service, separate from the frozen decision profile."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.document_ai.extraction import extract_fields, extract_fields_llm
from app.document_ai.intake import read_document
from app.document_ai.models import DocumentAIResult
from app.document_ai.ocr import TesseractOCR
from app.document_ai.validation import validate_extraction


def process_document(
    path: Path,
    ocr: Optional[TesseractOCR] = None,
    client=None,
    use_llm: bool = False,
    dpi: int = 220,
) -> DocumentAIResult:
    document = read_document(path, ocr=ocr, dpi=dpi)
    fields = extract_fields_llm(document.text, client) if use_llm else extract_fields(document.text)
    report = validate_extraction(document.text, fields)
    return DocumentAIResult(
        document=document,
        fields=fields,
        validation=report,
        extractor="llm-structured" if use_llm else "deterministic-form-baseline",
        input_name=path.name,
    )
