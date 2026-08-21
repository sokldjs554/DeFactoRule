"""워크플로 F9 — **실제 `조치` 사례에서 선례를 따르지 않는가.**

이것이 Phase 3 의 존재 이유이자 완료 기준 1번이다.

E5 실측: test 의 `조치` 14건 중 dev 에 닮은 선례가 있는 것은 **1건(7.1%)** 이다.
그러므로 Agent 가 `조치` 사례에서 선례 경로(A)를 자주 고른다면, 그것은 없는
근거를 있다고 착각하는 것이다. 합성으로는 이 성질을 흉내 낼 수 없어 실제
데이터를 쓴다.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from app.agents.fixtures import action_cases
from app.agents.state import Path as RoutePath
from app.agents.workflow import Workflow
from app.core.io import load_jsonl
from app.core.paths import EVAL, PROCESSED, RESULTS
from app.retrieval.lexical import LexicalRetriever


@pytest.fixture(scope="module")
def workflow_states():
    for path in (EVAL / "nonaction_dev.jsonl", EVAL / "nonaction_test.jsonl",
                 PROCESSED / "cases_nonaction.jsonl", RESULTS / "e6_rules.json",
                 RESULTS / "trap_risk.json"):
        if not path.exists():
            pytest.skip(f"{path.name} 이 없습니다")

    dev = [r for r in load_jsonl(EVAL / "nonaction_dev.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
              for c in cases]
    rules = json.loads((RESULTS / "e6_rules.json").read_text(encoding="utf-8"))["rules"]
    risk = json.loads((RESULTS / "trap_risk.json").read_text(encoding="utf-8"))

    flow = Workflow(LexicalRetriever().fit(dev, [t for t in corpus if t]),
                    dev, rules, risk)
    return test, [flow.run(row) for row in test]


def test_action_cases_do_not_follow_precedents(workflow_states):
    """`조치` 사례에서 선례 경로를 고르지 않는가 — 완료 기준 1번."""
    test, states = workflow_states
    actions = action_cases(test)
    assert actions, "test 에 `조치` 사례가 없습니다"

    routes = Counter(state.route for row, state in zip(test, states)
                     if row["label"] == "조치")
    followed = routes[RoutePath.PRECEDENT]
    assert followed <= 1, (
        f"`조치` {len(actions)}건 중 {followed}건에서 선례를 따랐습니다. "
        f"E5 실측으로 닮은 선례가 있는 것은 1건(7.1%)뿐입니다 — 없는 근거를 "
        f"있다고 본 것입니다. 경로 분포: {dict(routes)}"
    )


def test_the_agent_abstains_rather_than_guessing_on_actions(workflow_states):
    """`조치` 사례의 대부분에서 판단을 보류하는가.

    커버리지가 떨어지는 것은 실패가 아니다. **근거 없이 답하는 것이 실패다.**
    """
    test, states = workflow_states
    abstained = sum(1 for row, state in zip(test, states)
                    if row["label"] == "조치" and state.abstained)
    total = sum(1 for row in test if row["label"] == "조치")
    assert abstained >= total * 0.5, (
        f"`조치` {total}건 중 기권은 {abstained}건뿐입니다. 근거가 없는데 "
        f"답하고 있습니다."
    )


def test_every_answer_carries_the_evidence_it_used(workflow_states):
    """답한 건은 전부 근거를 추적할 수 있는가 — 완료 기준 5번."""
    _test, states = workflow_states
    for state in states:
        if state.abstained:
            assert state.abstention_reason is not None, "이유 없이 기권했습니다"
            continue
        assert state.evidence_used, f"근거 없이 답했습니다: {state.request_key}"
        known = set(state.evidence_by_id())
        assert set(state.evidence_used) <= known, "실재하지 않는 근거를 인용했습니다"


def test_execution_trace_is_complete(workflow_states):
    """모든 단계가 흔적을 남기는가 — 재현 가능성의 최소 조건."""
    _test, states = workflow_states
    for state in states:
        nodes = [step.node for step in state.execution_trace]
        assert nodes[:3] == ["retrieve", "rules", "route"], nodes
        assert state.route_reason, "어느 줄이 발화했는지 남지 않았습니다"


def test_both_evidence_paths_are_used(workflow_states):
    """선례 경로와 규칙 경로가 둘 다 쓰이는가 — 완료 기준 2번.

    한쪽만 쓰인다면 경로를 나눈 의미가 없다.
    """
    _test, states = workflow_states
    routes = Counter(state.route for state in states)
    for path in (RoutePath.PRECEDENT, RoutePath.RULE, RoutePath.ABSTAIN):
        assert routes[path] > 0, f"{path.value} 경로가 한 번도 쓰이지 않았습니다: {dict(routes)}"


def test_abstained_rows_still_carry_a_provisional_label(workflow_states):
    """기권한 건도 평가용 레코드에는 라벨을 담는가 (EV-24).

    `predicted: null` 로 내보내면 위험-커버리지 하네스가 기권을 **오답**으로
    센다. 기권을 보여주려고 만든 지표가 기권을 벌하는 꼴이고, 실제로 AURC
    비교의 판정이 뒤집혔다(-0.270 유의 -> -0.021 판정 보류).

    서비스가 쓰는 `decision` 은 여전히 비어 있어야 한다 — 평가와 서비스는
    다른 것을 본다.
    """
    _test, states = workflow_states
    abstained = [s for s in states if s.abstained]
    assert abstained, "기권한 건이 하나도 없습니다"
    for state in abstained:
        assert state.decision is None, "기권했는데 서비스가 답을 내놓습니다"
        record = state.to_prediction()
        assert record["predicted"], (
            f"기권 레코드에 라벨이 없습니다 — 곡선이 이것을 오답으로 셉니다: "
            f"{state.request_key}"
        )
        assert record["abstained"] is True
        assert record["confidence"] == "low"


def test_a_precedent_below_the_floor_never_recovers_an_abstention():
    """E11b dry-run 이 잡은 결함 — **문턱 아래 선례로 기권이 거둬졌다.**

    `apply_verdict` 는 1등 근거를 `rank == 0` 으로만 집었고 점수를 보지
    않았다. 그래서 유사도 0.0559 짜리 선례로도 답을 내놓았다. dev 보정에서
    `DOUBT`(0.15) 아래 구간의 오류율은 0.500 이다 — 동전 던지기다.

    지금은 `targets()` 가 사정거리를 22건으로 막아 실제로는 도달하지 않는다.
    그러나 그러면 안전이 **호출자의 예의**에 걸린다. 사정거리 밖 기권 56건은
    전부 1등 선례를 문턱 아래에 갖고 있으므로, 실행기가 `targets()` 를 한 번
    빼먹으면 56건이 동전 던지기로 답한다.

    0.0559 는 지어낸 값이 아니라 test #0 에서 관측한 값이다.
    """
    from app.agents.applicability import apply_verdict
    from app.agents.state import (
        AbstentionReason,
        AgentState,
        Evidence,
        EvidenceKind,
    )
    from app.domain.similarity import DOUBT

    def below_floor_abstention(score: float):
        state = AgentState(request="요청", request_key=("2021년.pdf", 2, "5", 1))
        state.retrieved_evidence = [
            Evidence(id="prec:2021년.pdf#5", kind=EvidenceKind.PRECEDENT,
                     label="비조치", score=score, rank=0,
                     source="2021년.pdf", serial="5"),
        ]
        state.precedent_score = score
        state.route, state.route_reason = RoutePath.ABSTAIN, "R2"
        state.abstain(AbstentionReason.NO_EVIDENCE, provisional="비조치")
        return state

    observed = 0.0559
    assert observed < DOUBT, "이 시험의 전제가 무너졌습니다"

    state = below_floor_abstention(observed)
    recovered, _ = apply_verdict(state, "applies")

    assert not recovered, f"유사도 {observed} 선례로 기권을 거뒀습니다"
    assert state.abstained, "기권이 풀렸습니다"
    assert state.abstention_reason is AbstentionReason.NO_EVIDENCE
    assert state.decision is None, "문턱 아래 선례로 답을 내놓았습니다"
    assert state.route_reason == "R2", "원래 기권 경로가 덮어써졌습니다"

    # 막은 것이 **문턱**임을 보인다. 점수만 문턱 위로 올리면 회수된다 —
    # 그렇지 않으면 이 시험은 다른 이유로 통과하고 있는 것이다.
    above = below_floor_abstention(DOUBT)
    recovered, _ = apply_verdict(above, "applies")
    assert recovered and above.decision == "비조치"
