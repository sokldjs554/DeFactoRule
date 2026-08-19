"""Router — **어떤 근거로 판단할지** 고른다. 전부 결정론.

## 이 프로젝트에서 Agent 의 능력은 답을 만드는 것이 아니다

E5 실측: 최근접 선례를 따르는 전략은 순응 구간 정확도 1.000, 함정 구간 0.000.
그리고 `조치` 14건 중 닮은 선례가 있는 것은 1건(7.1%)뿐이다. 검색은 가장 비싼
판단에서 눈이 먼다.

그러므로 물어야 할 것은 "무엇이 답인가" 가 아니라 **"지금 이 근거를 믿어도
되는가"** 다. 이 파일이 그것만 한다.

## LLM 은 여기 없다

경로 선택은 수치 비교와 논리 조건이다 — 명세 §9 가 결정론적 코드에 맡기라고
적은 그것이다. 모델이 경로까지 고르면 왜 그 경로를 골랐는지 되짚을 수 없고,
`route_reason` 은 장식이 된다.

## 문턱

다섯 개 중 넷은 dev 실측에서 나온다(`scripts/calibrate.py`).

    TRUST 0.60 / DOUBT 0.15   구간별 오류율 0.062 vs 0.500, 신뢰구간 비중첩
    RISK_CEILING 0.20         믿음 구간 오류율의 95% 상한(0.201)에 맞췄다
    MIN_AGREEMENT 0.60        상위 근거의 과반이 같은 결론을 가리켜야 한다

`MAX_YEAR_GAP` 만 데이터에서 나오지 않았다. 코퍼스가 2021~2025 이므로 5년을
넘는 간극은 선례가 현재 코퍼스 세대 이전이라는 뜻으로 잡았다. **이것 하나는
근거가 약하고, 그 사실을 여기 적어 둔다.**

## 지금 보정에서 R5 와 R8 은 사실상 같은 것을 본다

`trap_risk` 는 유사도 구간의 함수이고, 현재 dev 보정에서는 믿음 구간만
`RISK_CEILING` 아래다. 그래서 두 줄이 유사도 하나로 갈린다. 그래도 원시
유사도가 아니라 **보정된 위험**을 쓰는 이유는, 코퍼스가 늘거나 검색기가
바뀌면 그 대응이 달라지기 때문이다. 지금 같아 보인다는 사실은 숨기지 않는다.
"""

from __future__ import annotations

from collections import Counter

from app.agents.state import AbstentionReason, Evidence, Path, RouterSignals
from app.domain.similarity import DOUBT, TRUST

RISK_CEILING = 0.20     # 이보다 위험하면 선례를 쓰지 않는다
MIN_AGREEMENT = 0.60    # 상위 근거의 결론 일치도 하한
MIN_MARGIN = 0.02       # 1등과 2등이 이보다 붙으면 어느 쪽인지 정할 수 없다
MAX_YEAR_GAP = 5        # 이보다 오래된 선례보다 현행 규칙을 앞세운다
TOP_K = 3               # 일치도·출처 다양성을 볼 상위 근거 수

# 경로를 고른 이유 → 기권했을 때의 사유 코드
ABSTAIN_FOR = {
    "R1": AbstentionReason.CONFLICTING_EVIDENCE,   # 규칙 경로 가드
    "R2": AbstentionReason.NO_EVIDENCE,
    "R5": AbstentionReason.SURFACE_ONLY,
    "R6": AbstentionReason.CONFLICTING_EVIDENCE,
    "R9": AbstentionReason.AMBIGUOUS_MARGIN,
}


def signals_from(
    precedents: list[Evidence],
    rules: list[Evidence],
    trap_risk: float,
    request_year: int | None = None,
) -> RouterSignals:
    """근거 목록에서 경로 결정에 쓸 값을 뽑는다. 전부 산술이다."""
    usable = [e for e in precedents if e.score >= DOUBT]
    top = sorted(usable, key=lambda e: -e.score)[:TOP_K]

    if len(top) >= 2:
        margin = top[0].score - top[1].score
    elif top:
        margin = top[0].score
    else:
        margin = 0.0

    agreement = 1.0
    if top:
        agreement = sum(1 for e in top if e.label == top[0].label) / len(top)

    gap = None
    if top and request_year and top[0].year:
        gap = request_year - top[0].year

    rule_labels = {e.label for e in rules}
    return RouterSignals(
        evidence_count=len(usable),
        top_similarity=top[0].score if top else 0.0,
        margin=margin,
        label_agreement=agreement,
        source_diversity=len({e.source for e in top if e.source}),
        recency_gap=gap,
        rule_fired=len(rules),
        rule_conflict=len(rule_labels) > 1,
        trap_risk=trap_risk,
    )


def route(signals: RouterSignals) -> tuple[Path, str]:
    """경로와 **발화한 줄 번호**를 돌려준다.

    위에서부터 먼저 맞는 줄이 이긴다. 줄 번호를 함께 돌려주는 것이 요점이다 —
    "왜 이 경로인가" 를 사람의 서술이 아니라 데이터가 답해야 한다.
    """
    s = signals
    # 규칙 경로를 쓸 수 있는가. 규칙끼리 반대를 가리키면 쓸 수 없다.
    #
    # 초안은 이것을 **맨 위 줄(R1)** 로 뒀다. 그러자 dev 85건 중 29건이 규칙
    # 충돌로 기권했고, **그중 10건은 유사도 0.60 이상의 선례를 갖고 있었다.**
    # 약한 근거(n-gram 규칙, E6 에서 조치 전이 100%→20%)가 강한 근거(선례)를
    # 덮는 것은 증거의 위계를 뒤집는 것이다. 규칙 충돌은 **규칙 경로에 대한
    # 정보**이지 선례 경로에 대한 정보가 아니다. 그래서 가드로 강등했다.
    rule_usable = s.rule_fired > 0 and not s.rule_conflict

    def to_rules(reason: str) -> tuple[Path, str]:
        if rule_usable:
            return Path.RULE, reason
        return Path.ABSTAIN, "R1" if s.rule_conflict else reason

    # R2 — 선례도 규칙도 없다.
    if s.evidence_count == 0 and s.rule_fired == 0:
        return Path.ABSTAIN, "R2"

    # R3 — 선례가 문턱에 닿지 않는다. 규칙으로 간다.
    # (설계 초안의 R4 "선례도 규칙도 없다" 는 R2 가 이미 잡으므로 **도달할 수
    #  없었다.** 죽은 줄은 지운다 — 안 발화하는 규칙은 규칙이 아니다.)
    if s.evidence_count == 0:
        return to_rules("R3")

    # R5 — 표면만 닮았을 위험이 크다. 선례를 버린다.
    if s.trap_risk > RISK_CEILING:
        return to_rules("R5")

    # R6 — 서로 다른 출처가 서로 다른 결론을 가리킨다.
    if s.label_agreement < MIN_AGREEMENT and s.source_diversity >= 2:
        return Path.ABSTAIN, "R6"

    # R7 — 오래된 선례보다 현행 규칙.
    if s.recency_gap is not None and s.recency_gap > MAX_YEAR_GAP and rule_usable:
        return Path.RULE, "R7"

    # R9 — 1등과 2등이 붙어 있는데 결론이 갈린다. (R8 보다 먼저 본다)
    if s.evidence_count >= 2 and s.margin < MIN_MARGIN and s.label_agreement < 1.0:
        return Path.ABSTAIN, "R9"

    # R8 — 매우 닮았고 함정 위험이 낮다.
    if s.top_similarity >= TRUST:
        return Path.PRECEDENT, "R8"

    # R10 — 남은 경우. 선례를 쓰되 Validator 가 본다.
    return Path.PRECEDENT, "R10"


def decide(evidence: list[Evidence]) -> tuple[str | None, list[str]]:
    """고른 경로의 근거들에서 결론을 뽑는다 — 다수결, 동점은 점수 합으로.

    모델은 여기 관여하지 않는다. 근거가 무엇을 가리키는지는 이미 데이터에
    있다.
    """
    if not evidence:
        return None, []
    votes = Counter(e.label for e in evidence)
    best = max(votes, key=lambda lab: (votes[lab],
                                       sum(e.score for e in evidence if e.label == lab)))
    used = [e.id for e in evidence if e.label == best]
    return best, used
