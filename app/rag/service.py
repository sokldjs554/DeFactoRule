"""Optional Evidence RAG service.

Retrieval is deterministic and free.  Memo generation is opt-in and uses the
shared Anthropic structured-output boundary.  An invalid citation/quote never
becomes a trusted memo: the response is marked abstained instead.
"""

from __future__ import annotations

from functools import lru_cache

from app.infrastructure.anthropic_client import FatalApiError, call_structured
from app.rag.evidence import EvidenceRetriever, render_context
from app.rag.memo import MEMO_SCHEMA, MEMO_SYSTEM, build_prompt, validate_memo
from app.rag.schemas import RAGMemo, RAGRequest, RAGResponse, RAGValidation


class RAGUnavailable(RuntimeError):
    """The optional generation path cannot be used."""


@lru_cache(maxsize=1)
def _evidence_retriever() -> EvidenceRetriever:
    return EvidenceRetriever()


def _client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise RAGUnavailable("anthropic 패키지가 없습니다: pip install anthropic") from exc
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover
        raise RAGUnavailable(f"Anthropic 자격증명을 찾을 수 없습니다: {exc}") from exc


def run_rag(req: RAGRequest, client=None) -> RAGResponse:
    retriever = _evidence_retriever()
    hits = retriever.retrieve(
        req.request_text,
        request_serial=req.request_serial,
        k=req.top_k,
        temporal_policy=req.temporal_policy,
    )
    evidence = [hit.as_dict() for hit in hits]

    base = {
        "retriever": retriever.retriever.name,
        "temporal_policy": req.temporal_policy,
        "evidence_count": len(hits),
        "evidence": evidence,
    }
    if not hits:
        reason = (
            "temporal_policy=serial requires a parseable request_serial and at least "
            "one past precedent"
            if req.temporal_policy == "serial"
            else "no precedent evidence retrieved"
        )
        return RAGResponse(**base, abstained=True, abstain_reason=reason)

    if not req.generate_memo:
        return RAGResponse(**base)

    context = render_context(hits)
    prompt = build_prompt(req.request_text, context)
    try:
        result = call_structured(
            client or _client(),
            MEMO_SYSTEM,
            prompt,
            MEMO_SCHEMA,
            max_tokens=1400,
            effort="low",
        )
    except FatalApiError as exc:
        raise RAGUnavailable(f"계정 수준 오류: {exc}") from exc

    if "error" in result:
        return RAGResponse(
            **base,
            abstained=True,
            abstain_reason=str(result.get("error")),
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
        )

    data = result["data"]
    validation = validate_memo(data, hits)
    memo = RAGMemo.model_validate(data)
    validation_payload = RAGValidation(
        valid=validation.valid,
        invalid_citations=list(validation.invalid_citations),
        ungrounded_quotes=list(validation.ungrounded_quotes),
        reason=validation.reason,
    )
    return RAGResponse(
        **base,
        memo=memo,
        validation=validation_payload,
        abstained=not validation.valid,
        abstain_reason=None if validation.valid else validation.reason,
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
    )
