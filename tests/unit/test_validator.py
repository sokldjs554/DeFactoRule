"""Validator 여섯 검사 — **각각을 일부러 깨뜨려 발화를 확인한다.**

## 왜 이렇게 검사하는가

실제 파이프라인을 돌려 보면 V1·V2·V3·V5 는 한 번도 발화하지 않는다. 그것은
검사가 쓸모없다는 뜻이 아니라 **지금 배선에서는 발화할 수 없다**는 뜻이다.

    V1 schema             결정은 `decide()` 가 근거의 라벨에서 뽑으므로 항상 유효
    V2 evidence existence 인용 목록이 검색 결과에서 나오므로 항상 실재
    V3 source consistency 인용을 선례 원문에서 잘라 내므로 항상 대조된다
    V5 unsupported claim  결정이 근거에서 나오므로 항상 받쳐진다

넷 다 **구성상 참**이다. 그러면 왜 두는가 — 모델이 결정이나 인용을 내놓는
경로(V3+ LLM 적용가능성 검증, 선택 항목)가 붙는 순간 셋 다 발화 가능해지기
때문이다. 그때 검사가 없으면 지어낸 인용과 근거 없는 결론이 그대로 나간다.

그러므로 여기서는 **합성 상태로 각 검사를 일부러 실패시켜** 살아 있음을
확인한다. 실행 중에 안 걸린다고 검사가 죽은 것은 아니지만, 깨뜨려도 안 걸리면
그때는 정말 죽은 것이다.
"""

from __future__ import annotations

from app.agents.state import (
    AbstentionReason,
    AgentState,
    Evidence,
    EvidenceKind,
    Path,
)
from app.agents.validator import BLOCKING, validate
from app.core.audit import Discards


def _state(decision="비조치", **kwargs) -> AgentState:
    state = AgentState(request="망분리 대체통제를 적용한 클라우드 연계 질의",
                       request_key=("2024년 사례집.pdf", 1, "1", 1))
    state.retrieved_evidence = [Evidence(
        id="prec:2024년 사례집.pdf#7", kind=EvidenceKind.PRECEDENT, label="비조치",
        score=0.8, rank=0, source="2024년 사례집.pdf", serial="7",
        quote=kwargs.get("quote"))]
    state.route, state.route_reason = Path.PRECEDENT, "R8"
    state.decision = decision
    state.confidence = "high"
    state.evidence_used = kwargs.get("used", ["prec:2024년 사례집.pdf#7"])
    return state


def _result(state, check):
    return next(r for r in state.validation if r.check == check)


def test_v1_catches_an_unknown_label():
    state = _state(decision="아무말")
    validate(state)
    assert not _result(state, "V1").passed
    assert state.abstained and state.abstention_reason == AbstentionReason.VALIDATION_FAILED


def test_v2_catches_a_citation_that_does_not_exist():
    state = _state(used=["prec:없는집#999"])
    validate(state)
    assert not _result(state, "V2").passed
    assert state.abstained


def test_v3_catches_a_quote_that_is_not_in_the_source():
    """지어낸 인용을 잡는가. LLM 경로가 붙으면 여기가 첫 방어선이다."""
    state = _state(quote="원문 어디에도 없는 문장입니다")
    sources = {"prec:2024년 사례집.pdf#7": "망분리 대체통제를 적용한 클라우드 연계"}
    validate(state, sources)
    result = _result(state, "V3")
    assert not result.passed, "지어낸 인용을 통과시켰습니다"
    assert result.dropped_evidence == ["prec:2024년 사례집.pdf#7"]


def test_v3_accepts_a_verbatim_quote():
    origin = "망분리 대체통제를 적용한 클라우드 연계에 관한 질의"
    state = _state(quote="클라우드 연계")
    validate(state, {"prec:2024년 사례집.pdf#7": origin})
    assert _result(state, "V3").passed


def test_v4_catches_a_decision_against_a_high_precision_rule():
    state = _state(decision="비조치")
    state.rule_evidence = [Evidence(id="rule:e6#1", kind=EvidenceKind.RULE,
                                    label="조치", score=0.99, rank=1)]
    validate(state)
    assert not _result(state, "V4").passed


def test_v4_ignores_a_low_precision_rule():
    """정밀도가 낮은 규칙은 결정을 뒤집지 못한다 — 증거의 위계."""
    state = _state(decision="비조치")
    state.rule_evidence = [Evidence(id="rule:e6#1", kind=EvidenceKind.RULE,
                                    label="조치", score=0.70, rank=1)]
    validate(state)
    assert _result(state, "V4").passed


def test_v5_catches_a_decision_no_evidence_supports():
    state = _state(decision="조치")     # 근거는 전부 비조치를 가리킨다
    validate(state)
    assert not _result(state, "V5").passed
    assert state.abstained


def test_v6_flags_conflict_without_blocking():
    """충돌은 막지 않고 신뢰도만 낮춘다 — 막으면 커버리지가 무너진다."""
    state = _state(decision="비조치")
    state.confidence = "medium"
    state.rule_evidence = [Evidence(id="rule:e6#2", kind=EvidenceKind.RULE,
                                    label="조치", score=0.80, rank=2)]
    validate(state)
    assert not _result(state, "V6").passed
    assert not state.abstained, "V6 이 결정을 막았습니다"
    assert state.confidence == "low", "충돌인데 신뢰도를 낮추지 않았습니다"


def test_every_check_runs_even_when_it_passes():
    """통과한 검사도 결과를 남기는가 — 세어 보지 않으면 죽은 검사를 못 찾는다."""
    state = _state()
    validate(state)
    assert {r.check for r in state.validation} == {"V1", "V2", "V3", "V4", "V5", "V6"}


def test_blocking_checks_are_the_ones_that_abstain():
    """막는 검사와 막지 않는 검사가 문서대로인가."""
    assert BLOCKING == {"V1", "V2", "V5"}
    for check in ("V3", "V4", "V6"):
        assert check not in BLOCKING


def test_dropped_evidence_is_recorded_with_a_reason():
    """폐기된 근거가 이유와 함께 남는가 (IN-02·EV-16 에서 배운 것)."""
    state = _state(quote="원문에 없는 문장")
    discards: Discards = validate(
        state, {"prec:2024년 사례집.pdf#7": "전혀 다른 원문"})
    assert len(discards) == 1
    assert discards.records()[0]["rejected_for"] == ["인용이 원문에 없다"]
