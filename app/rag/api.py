"""FastAPI router for the optional Evidence RAG layer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.rag.schemas import RAGRequest, RAGResponse
from app.rag.service import RAGUnavailable, run_rag

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/evidence",
    response_model=RAGResponse,
    summary="Temporal hybrid retrieval + grounded evidence memo",
)
def evidence_rag(req: RAGRequest) -> RAGResponse:
    """Retrieve traceable precedent evidence and optionally generate a memo.

    `generate_memo=false` is deterministic/API-free.  Generation never returns a
    trusted memo unless every citation ID exists and every quoted span is grounded
    in its cited precedent request.
    """
    try:
        return run_rag(req)
    except RAGUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
