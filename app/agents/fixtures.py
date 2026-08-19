"""Workflow 극단 케이스 — **Retriever·Router 보다 먼저 만든다.**

## 왜 먼저인가

지난 단계들에서 반복한 실수가 하나 있다. 만들고 → 잘 되는 경우로 확인하고 →
돈 드는 실행으로 넘겼다. 그래서 실패가 매번 실행 중에, 사용자 쪽에서 드러났다.

    스키마의 maxItems       332 요청이 전부 400
    출력 상한 1200          기준이 88개가 되자 응답이 잘림
    weights 의 부분 표본     26/85 위에서 "넘는다" 판정

셋 다 **극단을 안 넣어 봐서** 생긴 일이다. 그래서 이번에는 극단부터 만든다.
Router 는 이 fixture 들을 통과하도록 쓰는 것이지, 쓰고 나서 맞춰 보는 것이
아니다.

## 무엇을 담는가

설계서 §8 의 아홉 가지다. 여덟 개는 합성이고 **F9 하나는 실제 데이터**다 —
`조치` 앵커링은 합성으로 흉내 내면 의미가 없다. E5 가 실측한 그 14건을 쓴다.

각 시나리오는 (이름, 요청문, 선례 목록, 규칙 발화 여부, **기대 경로**) 를 갖는다.
기대 경로를 여기 적어 두는 것이 요점이다. Router 를 쓴 뒤에 "이 정도면 맞네"
라고 판정하면 그건 검사가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.state import Path


@dataclass(frozen=True)
class Precedent:
    """fixture 안의 선례 하나."""

    serial: str
    request: str
    label: str
    similarity: float          # Retriever 를 거치지 않고 직접 준다
    source: str = "fixture집"
    year: int = 2024


@dataclass(frozen=True)
class Scenario:
    """극단 케이스 하나. **기대 경로를 미리 적는다.**"""

    name: str
    what: str
    request: str
    precedents: list[Precedent]
    rules_fire: list[str] = field(default_factory=list)   # 발화한 규칙의 라벨
    expect_route: Path = Path.ABSTAIN
    expect_rule: str = ""      # 발화해야 할 결정 표의 줄 (R1~R10)
    year: int = 2025


# 유사도 값은 도메인 문턱을 기준으로 고른다 — 숫자를 손으로 흩뿌리지 않는다.
from app.domain.similarity import DOUBT, TRUST  # noqa: E402

TRUSTED = TRUST + 0.15        # 0.75 — 확실히 믿음 구간
MIDDLE = (DOUBT + TRUST) / 2  # 0.375 — 가운데 (dev 에 5건뿐인 구간)
WEAK = DOUBT + 0.01           # 0.16 — 겨우 문턱 위
BELOW = DOUBT - 0.01          # 0.14 — 문턱 아래


SCENARIOS: list[Scenario] = [
    Scenario(
        name="F1_no_evidence",
        what="선례도 규칙도 없다",
        request="전에 없던 유형의 요청입니다.",
        precedents=[],
        rules_fire=[],
        expect_route=Path.ABSTAIN,
        expect_rule="R2",
    ),
    Scenario(
        name="F2_single_weak",
        what="겨우 문턱을 넘은 선례 하나뿐",
        request="내부망과 외부망 사이 자료 전송에 관한 질의입니다.",
        precedents=[Precedent("w1", "망분리 관련 질의", "비조치", WEAK)],
        rules_fire=["비조치"],
        # 약한 선례 하나를 무조건 따르지 않는다. 규칙이 있으면 규칙으로.
        expect_route=Path.RULE,
        expect_rule="R5",
    ),
    Scenario(
        name="F3_multiple_agreeing",
        what="같은 결론을 가리키는 강한 선례 다수",
        request="클라우드 이용 시 망분리 예외 적용이 가능한지 질의합니다.",
        precedents=[
            Precedent("a1", "클라우드 망분리 예외 질의", "비조치", TRUSTED),
            Precedent("a2", "클라우드 구간 접속 질의", "비조치", TRUSTED - 0.05,
                      source="다른집"),
            Precedent("a3", "망분리 대체통제 질의", "비조치", TRUSTED - 0.08),
        ],
        rules_fire=[],
        expect_route=Path.PRECEDENT,
        expect_rule="R8",
    ),
    Scenario(
        name="F4_conflicting",
        what="비슷하게 닮은 선례들이 반대를 가리킨다",
        request="외부 단말기에서 내부 시스템에 접속하는 행위에 관한 질의입니다.",
        precedents=[
            Precedent("c1", "외부 단말 내부망 접속", "비조치", TRUSTED),
            Precedent("c2", "외부 단말 내부망 연결", "조치", TRUSTED - 0.005,
                      source="다른집"),
        ],
        rules_fire=[],
        expect_route=Path.ABSTAIN,
        # 처음에는 R9(여유 부족)를 적었다. Router 를 돌려 보니 R6(출처가 다른데
        # 결론이 갈린다)이 먼저 발화했고, **그쪽이 더 나은 답이다** — 여유가
        # 좁은 것은 증상이고 충돌이 원인이다. 기대를 바꿨다는 사실을 남긴다.
        expect_rule="R6",
    ),
    Scenario(
        name="F5_wrong_precedent",
        what="표면은 매우 닮았으나 결론이 갈리는 선례 — **TRAP**",
        request="전자금융거래와 무관한 임직원 인사정보를 클라우드로 처리합니다.",
        precedents=[Precedent("t1", "클라우드로 고객정보를 처리합니다", "조치", MIDDLE)],
        rules_fire=["비조치"],
        # 가운데 구간은 dev 에 5건뿐이라 믿을 근거가 없다. 규칙으로 간다.
        expect_route=Path.RULE,
        expect_rule="R5",
    ),
    Scenario(
        name="F6_partial_evidence",
        what="선례는 있는데 인용이 원문에 없다",
        request="망분리 대체통제를 적용한 클라우드 연계에 관한 질의입니다.",
        precedents=[Precedent("p1", "존재하지 않는 인용", "비조치", TRUSTED)],
        rules_fire=[],
        # 라우팅은 A 로 가되 Validator(V3)가 근거를 폐기한다 — 그 결과는
        # workflow 검사에서 본다. 여기서는 경로만 못 박는다.
        expect_route=Path.PRECEDENT,
        expect_rule="R8",
    ),
    Scenario(
        name="F7_version_mismatch",
        what="선례가 너무 오래됐다",
        request="2025년 개정 감독규정에 따른 신규 사안입니다.",
        precedents=[Precedent("v1", "구 규정 하의 유사 사안", "비조치", TRUSTED,
                              year=2018)],
        rules_fire=["비조치"],
        expect_route=Path.RULE,
        expect_rule="R7",
    ),
    Scenario(
        name="F8_ambiguous_request",
        what="요청문이 너무 짧아 무엇을 묻는지 알 수 없다",
        request="가능한지요.",
        precedents=[
            Precedent("m1", "어떤 질의", "비조치", BELOW),
            Precedent("m2", "다른 질의", "조치", BELOW - 0.005),
        ],
        rules_fire=[],
        expect_route=Path.ABSTAIN,
        expect_rule="R2",
    ),
]


SCENARIOS.append(Scenario(
    name="F4b_same_source_tie",
    what="같은 출처의 선례 둘이 붙어 있는데 결론이 갈린다 — R6 이 못 잡는 자리",
    request="전자지급결제대행 등록 대상인지 질의합니다.",
    precedents=[
        Precedent("s1", "전자지급결제대행 등록 여부", "비조치", TRUSTED),
        Precedent("s2", "전자지급결제대행 해당 여부", "조치", TRUSTED - 0.005),
    ],
    rules_fire=[],
    # 출처가 같으므로 R6(source_diversity >= 2)은 발화하지 않는다.
    # 그래도 1등과 2등이 반대를 가리키면 고를 수 없다 -> R9.
    expect_route=Path.ABSTAIN,
    expect_rule="R9",
))


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"그런 시나리오가 없습니다: {name}")


def action_cases(gold_rows: list[dict]) -> list[dict]:
    """F9 — 실제 `조치` 사례들. 합성으로 대체하지 않는다.

    E5 실측: test 의 `조치` 14건 중 dev 에 닮은 선례가 있는 것은 1건(7.1%)뿐이다.
    그러므로 Router 는 이들 대부분에서 선례를 쓸 수 없어야 하고, 규칙마저 없으면
    기권해야 한다. **여기서 A 로 가는 건이 많다면 그것이 곧 실패다.**
    """
    return [r for r in gold_rows if r.get("label") == "조치"]
