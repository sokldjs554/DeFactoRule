"""Retrieval-augmented evidence layer.

This package is intentionally optional: it does not change the frozen Router/S5
operating profile.  It retrieves traceable precedent evidence and can ask an LLM
to write an evidence memo whose citations are validated deterministically.
"""

from app.rag.evidence import EvidenceHit, EvidenceRetriever
from app.rag.memo import MemoValidation, validate_memo

__all__ = ["EvidenceHit", "EvidenceRetriever", "MemoValidation", "validate_memo"]
