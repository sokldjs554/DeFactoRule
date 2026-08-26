"""OCR-aware document intake and structured extraction."""

from app.document_ai.rag_bridge import process_document_with_rag
from app.document_ai.service import process_document

__all__ = ["process_document", "process_document_with_rag"]
