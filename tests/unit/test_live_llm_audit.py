"""live audit이 **세는 방식**을 고정한다. API 없이 검증 가능한 부분만.

표본 선정이 흔들리면 "cherry-picking 없음"이라는 주장 자체가 무너지고,
집계의 분모가 흔들리면 100%라는 숫자가 무의미해진다. 그 둘을 고정한다.
"""

from app.evaluation.live_llm_audit import (
    FROZEN_PROFILE,
    _rag_record,
    aggregate,
    stride_pick,
)
from app.rag.schemas import RAGClaim, RAGMemo, RAGResponse, RAGValidation


class TestStridePick:
    def test_it_starts_at_zero_and_is_deterministic(self):
        items = list(range(88))
        picked = stride_pick(items, 9)
        assert picked == [0, 9, 18, 27, 36, 45, 54, 63, 72, 81]
        assert stride_pick(items, 9) == picked

    def test_a_short_list_still_returns_its_head(self):
        assert stride_pick([1, 2, 3], 9) == [1]


class TestFrozenProfile:
    def test_the_guarded_numbers_are_the_published_ones(self):
        assert FROZEN_PROFILE == {"n": 168, "answered": 76, "abstained": 92,
                                  "correct": 63, "wrong": 13}


def rag_response(*, valid: bool, claims: int = 2, bad_quote: bool = False):
    memo = RAGMemo(
        summary="s",
        claims=[RAGClaim(claim=f"c{i}", evidence_id="P-1-1", quote="q")
                for i in range(claims)],
        uncertainty="", handoff_recommended=False,
    )
    validation = RAGValidation(
        valid=valid,
        ungrounded_quotes=[] if not bad_quote else ["P-1-1:q"],
        reason=None if valid else "citation_or_quote_validation_failed",
    )
    return RAGResponse(
        retriever="hybrid", temporal_policy="serial", evidence_count=1,
        evidence=[], memo=memo, validation=validation,
        abstained=not valid,
        abstain_reason=None if valid else "citation_or_quote_validation_failed",
        input_tokens=100, output_tokens=50,
    )


class FakeHit:
    evidence_id = "P-1-1"


def rag_item():
    return {"arm": "rag", "row": {"serial": "1", "source": "s", "page": 1,
                                  "pair_index": 1, "request": "x"},
            "hits": [FakeHit()]}


class TestRagRecord:
    def test_a_validated_memo_is_usable_downstream(self):
        record = _rag_record(rag_item(), rag_response(valid=True), 1.0)
        assert record["api_success"] and record["schema_valid"]
        assert record["validator_blocked"] is False
        assert record["downstream"] == "usable memo"

    def test_a_failed_validation_is_fail_closed_not_dropped(self):
        record = _rag_record(rag_item(), rag_response(valid=False, bad_quote=True), 1.0)
        assert record["schema_valid"] is True          # 스키마는 통과했다
        assert record["validator_blocked"] is True     # 그러나 검증이 막았다
        assert record["downstream"] == "abstain (fail-closed)"
        assert record["ungrounded_quotes"]

    def test_an_api_failure_is_not_counted_as_a_validator_block(self):
        response = RAGResponse(
            retriever="hybrid", temporal_policy="serial", evidence_count=1,
            evidence=[], abstained=True, abstain_reason="refusal",
        )
        record = _rag_record(rag_item(), response, 1.0)
        assert record["api_success"] is False
        assert record["validator_blocked"] is False
        assert record["failure_reason"] == "refusal"


class TestAggregate:
    def test_denominators_are_explicit_and_rates_match_by_hand(self):
        records = [
            {"arm": "s5", "api_success": True, "schema_valid": True,
             "n_factors": 4, "n_factors_grounded": 3, "n_factors_rejected": 1},
            {"arm": "s5", "api_success": False, "schema_valid": False,
             "n_factors": None, "n_factors_grounded": None,
             "n_factors_rejected": None},
            {"arm": "rag", "api_success": True, "schema_valid": True,
             "n_claims": 2, "invalid_citations": [], "ungrounded_quotes": ["x"],
             "validator_blocked": True},
        ]
        agg = aggregate(records)
        assert agg["api_calls_attempted"] == 3
        assert agg["api_success"] == {"n": 2, "denominator": 3, "rate": 0.6667}
        assert agg["s5_factor_grounding"]["grounded_rate"] == 0.75
        assert agg["rag_exact_quotes"]["ungrounded"] == 1
        # 검증에 걸린 표본: rejected factor가 있는 s5 1건 + blocked rag 1건
        assert agg["validator_rejections"]["samples_with_any_rejection"] == 2
        assert agg["unsupported_output_found"]["total"] == 2

    def test_zero_denominators_yield_none_not_a_fake_100(self):
        agg = aggregate([])
        assert agg["api_success"]["rate"] is None
        assert agg["s5_factor_grounding"]["grounded_rate"] is None
