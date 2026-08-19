"""Validator — 결정을 내보내기 전에 **별도 계층에서** 검증한다.

## 왜 판정과 분리하는가

Router 가 경로를 고르고 근거가 결론을 가리켰다고 해서 그 결론을 내보낼 수 있는
것은 아니다. 인용이 원문에 없을 수도 있고, 고른 근거가 결론을 받치지 못할 수도
있고, 고정밀 규칙과 정면으로 어긋날 수도 있다.

검증이 결정 안에 섞여 있으면 "무엇이 막았는가" 를 셀 수 없다. 그래서 여섯 개를
따로 두고 **각각의 발화 횟수를 남긴다.** 아무것도 막지 않는 검사는 검사가
아니고, 세어 보지 않으면 그것을 알 수 없다(설계서 §예상 failure P6).

## 다섯은 공짜고 하나만 모델을 쓴다

    V1 schema             형식
    V2 evidence existence 인용한 근거가 실재하는가
    V3 source consistency 인용이 원문에 글자 그대로 있는가
    V4 rule consistency   고정밀 규칙과 정면으로 어긋나는가
    V5 unsupported claim  결론을 가리키는 근거가 하나라도 있는가
    V6 conflict detection 근거들이 서로 다른 결론을 가리키는가

V3+ (선례가 이 요청에 **실제로 적용되는가**) 만 모델이 필요하고, 그것은 선택
항목이다. 결정론 다섯으로 먼저 결과를 낸다.

폐기된 근거는 조용히 사라지지 않는다 — 이유와 함께 `Discards` 에 남는다.
"""

from __future__ import annotations

from collections import Counter

from app.agents.state import (
    AbstentionReason,
    AgentState,
    Evidence,
    EvidenceKind,
    ValidationResult,
)
from app.core.audit import Discards
from app.domain.labels import NON_ACTIONS

# 이 정밀도 이상인 규칙과 정면으로 어긋나면 결정을 통과시키지 않는다
HIGH_PRECISION = 0.95


def v1_schema(state: AgentState) -> ValidationResult:
    """결정이 라벨 체계 안에 있는가."""
    if state.decision is None:
        return ValidationResult(check="V1", passed=True, detail="기권 — 검사 대상 아님")
    ok = state.decision in NON_ACTIONS
    return ValidationResult(check="V1", passed=ok,
                            detail="" if ok else f"알 수 없는 라벨: {state.decision}")


def v2_evidence_exists(state: AgentState) -> ValidationResult:
    """인용한 근거가 실제로 검색 결과 안에 있는가."""
    known = set(state.evidence_by_id())
    missing = [eid for eid in state.evidence_used if eid not in known]
    return ValidationResult(check="V2", passed=not missing,
                            detail=f"실재하지 않는 근거 {missing}" if missing else "")


def v3_source_consistency(state: AgentState, sources: dict[str, str],
                          discards: Discards) -> ValidationResult:
    """인용 구절이 원문에 글자 그대로 있는가.

    `app.agents.criteria.quote_is_grounded` 를 그대로 쓴다 — 공백과 조판 잔재는
    무시하고 글자는 건드리지 않는다. 다시 구현하면 미묘하게 다른 검사가 된다.
    """
    from app.agents.criteria import quote_is_grounded

    dropped = []
    for item in state.all_evidence():
        if not item.quote:
            continue
        origin = sources.get(item.id)
        if origin is None:
            continue
        if not quote_is_grounded(item.quote, origin):
            dropped.append(item.id)
            discards.drop({"id": item.id, "quote": item.quote[:80]},
                          ["인용이 원문에 없다"])
    if dropped:
        state.retrieved_evidence = [e for e in state.retrieved_evidence
                                    if e.id not in dropped]
        state.rule_evidence = [e for e in state.rule_evidence if e.id not in dropped]
        state.evidence_used = [e for e in state.evidence_used if e not in dropped]
    return ValidationResult(check="V3", passed=not dropped,
                            detail=f"인용 미대조 {len(dropped)}건" if dropped else "",
                            dropped_evidence=dropped)


def v4_rule_consistency(state: AgentState) -> ValidationResult:
    """고정밀 규칙과 정면으로 어긋나는 결정인가."""
    if state.decision is None:
        return ValidationResult(check="V4", passed=True)
    against = [e.id for e in state.rule_evidence
               if e.score >= HIGH_PRECISION and e.label != state.decision]
    return ValidationResult(
        check="V4", passed=not against,
        detail=f"고정밀 규칙과 어긋남 {against}" if against else "")


def v5_unsupported_claim(state: AgentState) -> ValidationResult:
    """결론을 가리키는 근거가 하나라도 있는가."""
    if state.decision is None:
        return ValidationResult(check="V5", passed=True)
    supporting = [e.id for e in state.all_evidence() if e.label == state.decision]
    return ValidationResult(check="V5", passed=bool(supporting),
                            detail="" if supporting else "결론을 받치는 근거가 없다")


def v6_conflict(state: AgentState) -> ValidationResult:
    """쓸 만한 근거들이 서로 다른 결론을 가리키는가.

    막지는 않는다 — 신뢰도를 낮춘다. 충돌 자체는 흔하고, 그것만으로 기권하면
    커버리지가 무너진다. Router 의 R6·R9 가 심한 경우를 이미 잡는다.
    """
    labels = Counter(e.label for e in state.all_evidence())
    conflicted = len(labels) > 1
    return ValidationResult(
        check="V6", passed=not conflicted,
        detail=f"근거가 가리키는 결론 {dict(labels)}" if conflicted else "")


# 통과하지 못하면 결정을 내보내지 않는 검사들
BLOCKING = {"V1", "V2", "V5"}


def validate(state: AgentState, sources: dict[str, str] | None = None) -> Discards:
    """여섯 검사를 돌리고, 막아야 하면 기권시킨다.

    통과한 검사도 결과를 남긴다 — 아무것도 안 막는 검사를 찾으려면 세어야 한다.
    """
    discards = Discards("validator")
    state.validation = [
        v1_schema(state),
        v2_evidence_exists(state),
        v3_source_consistency(state, sources or {}, discards),
        v4_rule_consistency(state),
        v5_unsupported_claim(state),
        v6_conflict(state),
    ]
    state.step("validate", "검사 6종",
               failed=[r.check for r in state.validation if not r.passed])

    blocked = [r for r in state.validation if not r.passed and r.check in BLOCKING]
    if blocked:
        state.abstain(AbstentionReason.VALIDATION_FAILED,
                      "검증 실패: " + ", ".join(r.check for r in blocked))
        return discards

    if any(not r.passed for r in state.validation):
        state.confidence = "low" if state.confidence == "medium" else state.confidence
    return discards


def evidence_sources(evidence: list[Evidence], precedents: list[dict]) -> dict[str, str]:
    """근거 id → 대조할 원문. 선례는 그 선례의 요청문이다."""
    by_serial = {f"prec:{p.get('source')}#{p.get('serial')}": p.get("request", "")
                 for p in precedents}
    return {e.id: by_serial.get(e.id, "")
            for e in evidence if e.kind == EvidenceKind.PRECEDENT and e.id in by_serial}
