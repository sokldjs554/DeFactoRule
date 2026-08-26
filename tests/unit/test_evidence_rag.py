from __future__ import annotations

from app.rag.evidence import EvidenceRetriever, evidence_id, render_context
from app.rag.memo import validate_memo
from app.rag.schemas import RAGRequest
from app.rag.service import run_rag


def _rows() -> list[dict]:
    return [
        {
            "source": "a.pdf",
            "serial": "220010",
            "page": 1,
            "pair_index": 1,
            "sector": "보험",
            "request": "보험회사가 내부 업무 시스템을 이용하는 경우",
            "label": "비조치",
        },
        {
            "source": "b.pdf",
            "serial": "230010",
            "page": 2,
            "pair_index": 1,
            "sector": "보험",
            "request": "보험회사가 외부 클라우드 서비스를 이용하는 경우",
            "label": "기타",
        },
        {
            "source": "c.pdf",
            "serial": "250010",
            "page": 3,
            "pair_index": 1,
            "sector": "보험",
            "request": "보험회사가 외부 클라우드 서비스를 신규 이용하는 경우",
            "label": "조치",
        },
    ]


def test_evidence_id_is_stable() -> None:
    assert evidence_id(_rows()[0]) == "P-220010-1"


def test_temporal_filter_happens_before_ranking() -> None:
    retriever = EvidenceRetriever(_rows())
    hits = retriever.retrieve(
        "보험회사가 외부 클라우드 서비스를 신규 이용하는 경우",
        request_serial="240001",
        k=3,
    )
    assert hits
    assert all(hit.serial < "240001" for hit in hits)
    assert "250010" not in {hit.serial for hit in hits}


def test_serial_policy_without_serial_fails_closed() -> None:
    retriever = EvidenceRetriever(_rows())
    assert retriever.retrieve("보험회사 클라우드", request_serial=None) == []


def test_render_context_preserves_provenance() -> None:
    hit = EvidenceRetriever(_rows()).retrieve(
        "보험회사가 외부 클라우드 서비스를 이용하는 경우",
        request_serial="240001",
        k=1,
    )[0]
    context = render_context([hit])
    assert f"[{hit.evidence_id}]" in context
    assert "historical_outcome:" in context
    assert hit.request in context


def test_memo_validation_accepts_grounded_quote() -> None:
    hit = EvidenceRetriever(_rows()).retrieve(
        "보험회사가 외부 클라우드 서비스를 이용하는 경우",
        request_serial="240001",
        k=1,
    )[0]
    quote = hit.request[:8]
    result = validate_memo(
        {
            "claims": [
                {"claim": "관련 선례가 있다", "evidence_id": hit.evidence_id, "quote": quote}
            ]
        },
        [hit],
    )
    assert result.valid


def test_memo_validation_rejects_unknown_citation_and_quote() -> None:
    hit = EvidenceRetriever(_rows()).retrieve(
        "보험회사가 외부 클라우드 서비스를 이용하는 경우",
        request_serial="240001",
        k=1,
    )[0]
    bad_id = validate_memo(
        {"claims": [{"claim": "x", "evidence_id": "P-999999-1", "quote": "x"}]},
        [hit],
    )
    assert not bad_id.valid
    assert bad_id.invalid_citations == ("P-999999-1",)

    bad_quote = validate_memo(
        {"claims": [{"claim": "x", "evidence_id": hit.evidence_id, "quote": "없는 문장"}]},
        [hit],
    )
    assert not bad_quote.valid
    assert bad_quote.ungrounded_quotes


def test_retrieval_only_service_never_calls_llm(monkeypatch) -> None:
    retriever = EvidenceRetriever(_rows())
    monkeypatch.setattr("app.rag.service._evidence_retriever", lambda: retriever)
    monkeypatch.setattr(
        "app.rag.service._client",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    response = run_rag(
        RAGRequest(
            request_text="보험회사가 외부 클라우드 서비스를 이용하는 경우",
            request_serial="240001",
            generate_memo=False,
        )
    )
    assert response.evidence_count > 0
    assert response.memo is None
    assert not response.abstained
