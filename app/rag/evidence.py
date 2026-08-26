"""Evidence retrieval for the optional RAG service layer.

The frozen decision Router is not modified here. RAG gets its own precedent
retrieval service with provenance-preserving evidence IDs, temporal filtering
before ranking, and the same calibrated similarity floor used by the decision
system to avoid filling context with arbitrarily weak top-k matches.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.io import load_jsonl
from app.core.paths import EVAL
from app.domain.similarity import SIMILARITY_FLOOR
from app.domain.temporal import eligible_indices
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import LexicalRetriever, shared_span

CLEAN_DEV = EVAL / "nonaction_dev_clean.jsonl"
DEFAULT_K = 5


@dataclass(frozen=True)
class EvidenceHit:
    evidence_id: str
    source: str
    page: int | None
    serial: str
    pair_index: int | None
    sector: str | None
    request: str
    outcome: str | None
    score: float
    shared_quote: str | None

    def as_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "page": self.page,
            "serial": self.serial,
            "pair_index": self.pair_index,
            "sector": self.sector,
            "request": self.request,
            "outcome": self.outcome,
            "score": self.score,
            "shared_quote": self.shared_quote,
        }


def evidence_id(row: dict) -> str:
    serial = str(row.get("serial") or "unknown")
    pair = row.get("pair_index")
    suffix = str(pair) if pair is not None else "0"
    return f"P-{serial}-{suffix}"


class EvidenceRetriever:
    """Hybrid precedent retriever with T-serial eligibility and a relevance floor.

    The corpus is the clean dev precedent pool. It is deliberately separate from
    the frozen evaluation Router so adding RAG cannot silently change published
    aggregate metrics. Results below the project's calibrated similarity floor
    are omitted rather than supplied to the LLM as weak evidence.
    """

    def __init__(self, precedents: list[dict] | None = None) -> None:
        self.precedents = precedents if precedents is not None else load_jsonl(CLEAN_DEV)
        corpus = [str(row.get("request") or "") for row in self.precedents]
        self.retriever = HybridRetriever(LexicalRetriever(), DenseRetriever())
        self.retriever.fit(self.precedents, corpus)

    def retrieve(
        self,
        request_text: str,
        *,
        request_serial: str | None = None,
        k: int = DEFAULT_K,
        temporal_policy: str = "serial",
        min_score: float = SIMILARITY_FLOOR,
    ) -> list[EvidenceHit]:
        if not request_text.strip() or k <= 0:
            return []

        if temporal_policy == "serial":
            if not request_serial:
                return []
            candidate_indices = eligible_indices(
                self.precedents,
                {"serial": request_serial},
                policy="serial",
            )
        elif temporal_policy == "none":
            candidate_indices = None
        else:
            raise ValueError(f"unsupported temporal policy: {temporal_policy}")

        ranked = self.retriever.search(
            request_text,
            k=k,
            candidate_indices=candidate_indices,
        )
        hits: list[EvidenceHit] = []
        for index, score in ranked:
            if score < min_score:
                continue
            row = self.precedents[index]
            precedent_request = str(row.get("request") or "")
            hits.append(
                EvidenceHit(
                    evidence_id=evidence_id(row),
                    source=str(row.get("source") or ""),
                    page=row.get("page"),
                    serial=str(row.get("serial") or ""),
                    pair_index=row.get("pair_index"),
                    sector=row.get("sector"),
                    request=precedent_request,
                    outcome=row.get("label"),
                    score=float(score),
                    shared_quote=shared_span(request_text, precedent_request),
                )
            )
        return hits


def render_context(hits: list[EvidenceHit]) -> str:
    """Render evidence without losing provenance IDs.

    Labels are historical outcomes, not instructions to copy the decision.
    """
    blocks = []
    for hit in hits:
        blocks.append(
            "\n".join(
                [
                    f"[{hit.evidence_id}]",
                    f"source: {hit.source}",
                    f"page: {hit.page}",
                    f"serial: {hit.serial}",
                    f"sector: {hit.sector or ''}",
                    f"historical_outcome: {hit.outcome or ''}",
                    f"retrieval_score: {hit.score:.6f}",
                    "precedent_request:",
                    hit.request,
                ]
            )
        )
    return "\n\n".join(blocks)
