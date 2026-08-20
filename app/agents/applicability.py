"""E11b — 선례가 이 요청에 **실제로 적용되는가**. 준비만 하고 호출하지 않는다.

## 왜 이것인가, 그리고 어디에 쓸 수 있는가

E8~E11a 가 남긴 숙제는 하나다 — **과잉 기권**. 78건 기권 중 50건은 답했어도
맞았다. 그런데 그 50건이 어디서 나오는지 세어 보면 설계가 바뀐다.

    기권 사유           건수   아까움   비율
    NO_EVIDENCE (R2)     38     25    65.8%   선례도 규칙도 없다
    CONFLICTING (R1)     30     19    63.3%   규칙끼리 반대
    SURFACE_ONLY (R5)     7      5    71.4%   표면만 닮았다
    (R6 다출처 불일치)     3      1    33.3%

**선례가 문턱 위에 있는 채로 기권한 것은 22건뿐이고, 그중 아까운 것은 14건이다.**
설계서가 V3+ 로 적어 둔 "이 선례가 적용되는가" 는 **그 22건에만 쓸 수 있다.**
나머지 56건에는 검사할 선례 자체가 없다.

그러므로 이 모듈의 사정거리를 정직하게 적는다.

    사정거리   기권 78건 중 22건 (28%)
    회수 가능  많아야 14건
    비용       22건 × 소형 호출 ≈ $0.3   (설계서의 $2.4 추정은 87건 기준이었다)

$2.4 가 아니라 $0.3 이다. 그리고 나머지 56건은 **다른 처방이 필요하다** —
그것은 이 모듈이 아니라 Router 에 근거 원천을 하나 더 붙이는 문제다.

## 모델에게 무엇을 묻는가

**결론을 묻지 않는다.** 결론을 물으면 그것은 그냥 LLM 분류기이고, 이 프로젝트가
이미 갖고 있다(`sector`, AURC 0.124). 여기서 묻는 것은 하나다.

> 두 사안의 차이가 **결론을 바꿀 만한가?**

모델은 `applies` / `differs` / `unclear` 셋 중 하나와, 그렇게 본 근거 구절을
요청문·선례에서 각각 인용한다. 인용은 Validator V3 가 원문과 글자 단위로
대조한다 — 지어낸 인용은 그 자리에서 폐기된다.

## 순환을 막는다

프롬프트에 **선례의 결론을 넣지 않는다.** 넣으면 모델이 "이 선례는 비조치니까
적용된다" 는 식으로 결론에서 거꾸로 판단할 수 있고, 그러면 이 검사는 결론을
한 번 더 베끼는 장치가 된다. 요청문 두 개만 준다.

## 판정은 셋뿐이다 — `partial` 을 만들지 않는다

사안 하나가 여러 쟁점을 담고 선례가 그중 일부만 덮는 경우는 흔하다. 그렇다고
`partial` 을 네 번째 값으로 두면 **그 값에 대한 회수 규칙을 또 정해야 하고,
5건 실측으로는 그 규칙을 검증할 수 없다.** 부분 적용은 `differs` 또는
`unclear` 로 흡수한다 — 둘 다 기권을 유지하므로 보수적이다.

회수는 `applies` **하나에서만** 일어난다. 스키마 밖의 값이 새어 들어와도
회수하지 않는다(fail-closed).

## 검증되지 않은 가설 H1 — 반대 근거가 남아 있으면 회수하지 않는다

`apply_verdict` 는 모델이 top-1 선례를 보고 `applies` 라고 답하면 기권을
거둔다. 그런데 **모델은 top-1 밖에 보지 않았다.** 문턱 위에 반대 결론을
가리키는 선례가 따로 남아 있다면, "top-1 이 적용된다" 는 말은 그 반대 선례가
적용되지 **않는다**는 뜻이 아니다. 그 상태로 답하면 반대 근거를 못 본 척하는
것이다. 그래서 막는다.

실측(test 170건 · Lexical · router 정책):

    사정거리                     22건
    문턱 위 반대 근거가 남은 건    5건 (22.7%)
    그 5건의 기권 사유            전부 CONFLICTING_EVIDENCE
    1등과 최상위 반대의 여유       0.008 · 0.047 · 0.106 · 0.226 · 0.565

여유가 0.008 부터 0.565 까지 흩어져 있어서 **여유로는 가를 수 없다.** 그래서
새 문턱을 만들지 않고 기존 `DOUBT` 하나만 쓴다 — 문턱 위이면서 반대면 막는다.

**이것은 아직 사실이 아니라 가설이다.** 이 가드가 막는 5건이 실제로 틀렸을
건인지, 아니면 맞았을 건을 버리는 것인지는 E11b 실측 전에는 알 수 없다.
반증 조건을 미리 적어 둔다.

    가설이 틀렸다고 볼 조건
      막힌 건의 정답이 top-1 라벨과 일치하는 비율이 과반이면,
      이 가드는 회수 가능한 건을 버리고 있는 것이다.

참고로 5건의 정답은 각각 비조치·비조치·기타·기타·비조치이고 top-1 라벨은
기타·조치·비조치·비조치·비조치다 — **일치 1건 / 5건.** 다만 이것은 모델이
그 5건에 `applies` 를 줄지 모르는 상태에서 센 수치이므로 검증이 아니다.
"""

from __future__ import annotations

from enum import Enum

MAX_TOKENS = 700


class Applicability(str, Enum):
    APPLIES = "applies"     # 차이가 결론을 바꿀 만하지 않다 — 선례를 써도 된다
    DIFFERS = "differs"     # 결론을 바꿀 만한 차이가 있다 — 선례를 버려야 한다
    UNCLEAR = "unclear"     # 요청문만으로는 알 수 없다


SYSTEM = """\
당신은 금융규제 사안 두 건의 **요청 내용**을 비교합니다.

당신은 두 사안의 결론(비조치/조치/기타)을 알지 못하며, 추측해서도 안 됩니다.
결론을 말하지 마십시오. 당신이 답할 것은 하나입니다.

    두 사안의 차이가 **결론을 바꿀 만한가?**

    applies   차이가 있더라도 규제 판단을 가를 만한 것이 아니다
    differs   규제 판단을 가를 만한 차이가 있다
    unclear   요청문만으로는 알 수 없다

**모르면 unclear 를 쓰십시오.** 억지로 고르면 그 답은 근거가 없습니다.

근거로 드는 구절은 **각 요청문에 있는 그대로** 옮기십시오. 요약하거나 다듬지
마십시오 — 원문과 글자 단위로 대조합니다."""


def build_prompt(request: str, precedent_request: str) -> str:
    """요청문 둘만 준다. **선례의 결론은 넣지 않는다** — 순환을 막는다."""
    from app.core.text import clean_for_prompt

    return (f"[사안 A — 지금 판단할 요청]\n{clean_for_prompt(request)}\n\n"
            f"[사안 B — 닮은 선례의 요청]\n{clean_for_prompt(precedent_request)}")


def schema() -> dict:
    """구조화 출력. 배열 길이 제약은 쓰지 않는다 — API 가 거부한다(IN-10)."""
    return {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [a.value for a in Applicability],
                "description": "두 사안의 차이가 결론을 바꿀 만한가",
            },
            "quote_a": {
                "type": "string",
                "description": "사안 A 요청문에서 그대로 옮긴 근거 구절",
            },
            "quote_b": {
                "type": "string",
                "description": "사안 B 요청문에서 그대로 옮긴 근거 구절",
            },
        },
        "required": ["verdict", "quote_a", "quote_b"],
        "additionalProperties": False,
    }


def targets(states, rows, floor: float) -> list[int]:
    """이 검사를 **쓸 수 있는** 건의 인덱스.

    선례가 문턱 위에 있으면서 기권한 것만이다. 선례가 없으면 검사할 것이 없다.
    사정거리를 코드로 못 박아 두면, 나중에 "왜 22건뿐이냐" 를 다시 세지 않아도
    된다.
    """
    return [i for i, state in enumerate(states)
            if state.abstained and state.precedent_score >= floor]


def estimate_cost(n_targets: int, prompt_chars: int = 1400) -> float:
    """호출 전 규모. 한글은 대략 글자당 1토큰으로 잡는다."""
    from app.infrastructure.anthropic_client import PRICE_IN, PRICE_OUT

    inp = (prompt_chars + len(SYSTEM)) * n_targets
    return inp / 1e6 * PRICE_IN + MAX_TOKENS * n_targets / 1e6 * PRICE_OUT


def opposing_evidence(state) -> list[str]:
    """회수했을 때 **못 본 척하게 되는** 근거의 이름들 (가설 H1).

    모델은 top-1 선례 하나만 보고 판정한다. 문턱 위에 반대 결론을 가리키는
    선례가 따로 남아 있으면, "top-1 이 적용된다" 는 답은 그 선례를 배제하지
    않는다. 그 목록을 돌려준다 — 비어 있지 않으면 회수하지 않는다.

    문턱은 `DOUBT` 하나만 쓴다. 1등과의 여유로 가르지 않는 이유는 실측에서
    여유가 0.008~0.565 로 흩어져 있어 가를 자리가 없기 때문이다(모듈 문서 H1).
    """
    from app.domain.similarity import DOUBT

    top = next((e for e in state.retrieved_evidence if e.rank == 0), None)
    if top is None:
        return []
    return [e.id for e in state.retrieved_evidence
            if e.rank != 0 and e.score >= DOUBT and e.label != top.label]


def apply_verdict(state, verdict: str) -> tuple[bool, str]:
    """판정을 상태에 반영한다. **회수만 하고 새로 기권시키지 않는다.**

    이 검사는 이미 기권한 건에만 걸린다. `applies` 면 선례를 믿어도 된다는
    뜻이므로 기권을 거두고 선례 경로로 되돌린다. `differs` 와 `unclear` 는
    기권을 그대로 둔다 — 원래 기권이었으므로 아무것도 나빠지지 않는다.
    `applies` 가 아닌 값은 **무엇이든** 기권을 유지한다(fail-closed) — 스키마
    밖의 값이 새어 들어와도 회수 쪽으로 기울지 않는다.

    `applies` 라도 문턱 위에 반대 근거가 남아 있으면 회수하지 않는다(가설 H1).

    되돌린 경로는 흔적에 남는다. "왜 답하게 됐는가" 를 추적할 수 있어야 한다.
    """
    from app.agents.router import decide
    from app.agents.state import Path

    if verdict != Applicability.APPLIES.value:
        state.step("applicability", f"{verdict} — 기권 유지", verdict=verdict)
        return False, verdict

    opposing = opposing_evidence(state)
    if opposing:
        state.step("applicability",
                   f"applies 이지만 반대 근거 {len(opposing)}건 — 기권 유지",
                   verdict=verdict, opposing=opposing)
        return False, verdict

    usable = [e for e in state.retrieved_evidence if e.rank == 0]
    label, used = decide(usable)
    if label is None:
        state.step("applicability", "applies 이지만 쓸 근거가 없다", verdict=verdict)
        return False, verdict

    state.abstained = False
    state.abstention_reason = None
    state.route, state.route_reason = Path.PRECEDENT, "V3+"
    state.decision = label
    state.provisional = label
    state.evidence_used = used
    state.confidence = "medium"     # 모델이 거들었으므로 high 를 주지 않는다
    state.step("applicability", f"applies — 기권 회수 · {label}",
               verdict=verdict, evidence=used)
    return True, verdict


def quotes_are_grounded(result: dict, request: str, precedent: str) -> list[str]:
    """인용 둘이 각자의 원문에 글자 그대로 있는가. 없는 것의 이름을 돌려준다."""
    from app.agents.criteria import quote_is_grounded

    bad = []
    for field, origin in (("quote_a", request), ("quote_b", precedent)):
        quote = (result or {}).get(field) or ""
        if not quote_is_grounded(quote, origin):
            bad.append(field)
    return bad
