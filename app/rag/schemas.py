"""Pydantic contracts for the optional Evidence RAG API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    request_text: str = Field(..., min_length=1, max_length=20000)
    request_serial: str | None = Field(
        None,
        description=(
            "YYNNN chronology proxy. temporal_policy=serial이면 필수이며, "
            "없으면 안전하게 evidence 0건으로 처리한다."
        ),
    )
    top_k: int = Field(5, ge=1, le=10)
    temporal_policy: Literal["serial", "none"] = "serial"
    generate_memo: bool = Field(
        False,
        description="false면 retrieval만 수행하여 API 비용이 0이다.",
    )


class RAGEvidence(BaseModel):
    evidence_id: str
    source: str
    page: int | None = None
    serial: str
    pair_index: int | None = None
    sector: str | None = None
    request: str
    outcome: str | None = None
    score: float
    shared_quote: str | None = None


class RAGClaim(BaseModel):
    claim: str
    evidence_id: str
    quote: str


class RAGMemo(BaseModel):
    summary: str
    claims: list[RAGClaim]
    uncertainty: str
    handoff_recommended: bool


class RAGValidation(BaseModel):
    valid: bool
    invalid_citations: list[str] = []
    ungrounded_quotes: list[str] = []
    reason: str | None = None


class RAGResponse(BaseModel):
    retriever: str
    temporal_policy: str
    evidence_count: int
    evidence: list[RAGEvidence]
    memo: RAGMemo | None = None
    validation: RAGValidation | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
