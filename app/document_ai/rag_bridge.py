"""Bridge validated Document AI output into the optional Evidence RAG service."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.document_ai.models import DocumentAIResult
from app.document_ai.ocr import TesseractOCR
from app.document_ai.service import process_document
from app.rag.schemas import RAGRequest, RAGResponse
from app.rag.service import run_rag


@dataclass(frozen=True)
class DocumentRAGResult:
    document_ai: DocumentAIResult
    rag: RAGResponse | None
    forwarded: bool


def process_document_with_rag(
    path: Path,
    *,
    ocr: TesseractOCR | None = None,
    client=None,
    use_llm_extraction: bool = False,
    generate_memo: bool = False,
    top_k: int = 5,
    dpi: int = 220,
) -> DocumentRAGResult:
    """Forward only validated structured requests into temporal Evidence RAG."""
    document = process_document(
        path,
        ocr=ocr,
        client=client,
        use_llm=use_llm_extraction,
        dpi=dpi,
    )
    if not document.validation.valid or not document.fields.request:
        return DocumentRAGResult(document_ai=document, rag=None, forwarded=False)

    response = run_rag(
        RAGRequest(
            request_text=document.fields.request,
            request_serial=document.fields.serial,
            top_k=top_k,
            temporal_policy="serial",
            generate_memo=generate_memo,
        ),
        client=client,
    )
    return DocumentRAGResult(document_ai=document, rag=response, forwarded=True)
