"""Offline evaluation for the optional Evidence RAG retrieval layer.

There are no human relevance judgments for precedent retrieval in this corpus, so
this module deliberately does *not* call label agreement Recall@K or nDCG.
It reports evidence availability/provenance/temporal integrity plus outcome
agreement as a diagnostic only.
"""

from __future__ import annotations

from statistics import mean, median

from app.core.io import load_jsonl
from app.core.paths import EVAL
from app.domain.temporal import precedent_is_eligible
from app.rag.evidence import EvidenceRetriever

CLEAN_TEST = EVAL / "nonaction_test_clean.jsonl"


def evaluate_retrieval(k: int = 5) -> dict:
    test = load_jsonl(CLEAN_TEST)
    retriever = EvidenceRetriever()

    top1_scores: list[float] = []
    top1_agreement: list[bool] = []
    topk_same_outcome: list[bool] = []
    evidence_counts: list[int] = []
    ids: list[str] = []
    temporal_violations = 0

    for row in test:
        hits = retriever.retrieve(
            str(row.get("request") or ""),
            request_serial=str(row.get("serial") or "") or None,
            k=k,
            temporal_policy="serial",
        )
        evidence_counts.append(len(hits))
        ids.extend(hit.evidence_id for hit in hits)

        for hit in hits:
            if not precedent_is_eligible(
                {"serial": hit.serial},
                {"serial": row.get("serial")},
                policy="serial",
            ):
                temporal_violations += 1

        if not hits:
            continue
        top1_scores.append(hits[0].score)
        gold = row.get("label")
        top1_agreement.append(hits[0].outcome == gold)
        topk_same_outcome.append(any(hit.outcome == gold for hit in hits))

    n = len(test)
    with_evidence = sum(count > 0 for count in evidence_counts)
    return {
        "schema_version": 1,
        "split": "clean_test",
        "n": n,
        "k": k,
        "retriever": retriever.retriever.name,
        "temporal_policy": "serial",
        "evidence_available": with_evidence,
        "evidence_coverage": with_evidence / n if n else 0.0,
        "zero_evidence": n - with_evidence,
        "mean_evidence_count": mean(evidence_counts) if evidence_counts else 0.0,
        "median_top1_score": median(top1_scores) if top1_scores else None,
        "temporal_violations": temporal_violations,
        "evidence_ids_unique_within_run": len(ids) == len(set(ids)),
        "top1_outcome_agreement": (
            sum(top1_agreement) / len(top1_agreement) if top1_agreement else None
        ),
        "topk_contains_same_outcome": (
            sum(topk_same_outcome) / len(topk_same_outcome) if topk_same_outcome else None
        ),
        "diagnostic_note": (
            "outcome agreement is not a relevance metric; the corpus has no human "
            "precedent-relevance judgments"
        ),
    }
