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
