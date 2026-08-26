"""Pydantic contracts for the optional Evidence RAG API."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    request_text: str = Field(..., min_length=1, max_length=20000)
    request_serial: Optional[str] = Field(
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
    page: Optional[int] = None
    serial: str
    pair_index: Optional[int] = None
    sector: Optional[str] = None
    request: str
    outcome: Optional[str] = None
    score: float
    shared_quote: Optional[str] = None


class RAGClaim(BaseModel):
    claim: str
    evidence_id: str
    quote: str


class RAGMemo(BaseModel):
    summary: str
    claims: List[RAGClaim]
    uncertainty: str
    handoff_recommended: bool


class RAGValidation(BaseModel):
    valid: bool
    invalid_citations: List[str] = Field(default_factory=list)
    ungrounded_quotes: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


class RAGResponse(BaseModel):
    retriever: str
    temporal_policy: str
    evidence_count: int
    evidence: List[RAGEvidence]
    memo: Optional[RAGMemo] = None
    validation: Optional[RAGValidation] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
