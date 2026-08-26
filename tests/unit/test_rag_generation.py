from __future__ import annotations

import json

from app.rag.evidence import EvidenceRetriever
from app.rag.schemas import RAGRequest
from app.rag.service import run_rag


class _Usage:
    input_tokens = 100
    output_tokens = 40


class _Block:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Response:
    stop_reason = "end_turn"
    usage = _Usage()

    def __init__(self, payload: dict) -> None:
        self.content = [_Block(json.dumps(payload, ensure_ascii=False))]


class _Messages:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        return _Response(self.payload)


class _Client:
    def __init__(self, payload: dict) -> None:
        self.messages = _Messages(payload)


def _retriever() -> EvidenceRetriever:
    return EvidenceRetriever(
        [
            {
                "source": "precedent.pdf",
                "serial": "230010",
                "page": 7,
                "pair_index": 1,
                "sector": "전자금융",
                "request": "금융회사가 외부 클라우드 서비스를 이용하는 경우",
                "label": "기타",
            }
        ]
    )


def test_generated_memo_with_grounded_citation_is_trusted(monkeypatch) -> None:
    retriever = _retriever()
    monkeypatch.setattr("app.rag.service._evidence_retriever", lambda: retriever)
    payload = {
        "summary": "관련 선례가 있다.",
        "claims": [
            {
                "claim": "외부 클라우드 이용 선례가 존재한다.",
                "evidence_id": "P-230010-1",
                "quote": "외부 클라우드 서비스를 이용하는 경우",
            }
        ],
        "uncertainty": "신규 요청의 세부 조건은 별도 검토가 필요하다.",
        "handoff_recommended": True,
    }
    client = _Client(payload)
    result = run_rag(
        RAGRequest(
            request_text="금융회사 클라우드 이용 문의",
            request_serial="240001",
            generate_memo=True,
        ),
        client=client,
    )
    assert client.messages.calls == 1
    assert result.memo is not None
    assert result.validation is not None and result.validation.valid
    assert not result.abstained
    assert result.input_tokens == 100
    assert result.output_tokens == 40


def test_generated_memo_with_hallucinated_quote_fails_closed(monkeypatch) -> None:
    retriever = _retriever()
    monkeypatch.setattr("app.rag.service._evidence_retriever", lambda: retriever)
    payload = {
        "summary": "관련 선례가 있다.",
        "claims": [
            {
                "claim": "존재하지 않는 조건을 주장한다.",
                "evidence_id": "P-230010-1",
                "quote": "원문에 존재하지 않는 인용",
            }
        ],
        "uncertainty": "",
        "handoff_recommended": False,
    }
    result = run_rag(
        RAGRequest(
            request_text="금융회사 클라우드 이용 문의",
            request_serial="240001",
            generate_memo=True,
        ),
        client=_Client(payload),
    )
    assert result.validation is not None and not result.validation.valid
    assert result.abstained
    assert result.abstain_reason == "citation_or_quote_validation_failed"
