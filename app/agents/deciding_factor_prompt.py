"""S5 Deciding-Factor Analysis의 LLM 출력 계약.

모델은 조건과 차이를 구조화할 뿐 **최종 applicability basis를 정하지 않는다.**
그 값은 `deciding_factor.evaluate_diff_coverage`가 원문 대조 후 계산한다.
"""

from __future__ import annotations

from app.core.text import clean_for_prompt

MAX_TOKENS = 1400

SYSTEM = """\
당신은 금융규제 요청 두 건을 비교해 **조건 단위 차이**를 구조화합니다.
두 사안의 결론(비조치/조치/기타)은 알지 못하며 추측해서도 안 됩니다.

해야 할 일:
1. 두 요청에 실제로 공통으로 존재하는 조건을 shared_factors에 적습니다.
2. 요청 A에만 있는 실질 조건은 only_in_request에 적습니다.
3. 선례 B에만 있는 실질 조건은 only_in_precedent에 적습니다.
4. 각 차이가 규제 판단을 가를 수 있다고 보면 decisive=true로 표시합니다.
5. decisive=false라면 왜 판단을 가르지 않는지 why_not_decisive를 반드시 적습니다.

중요:
- factor의 text는 원문에서 **그대로** 옮긴 충분한 길이의 절이어야 합니다.
- 공통 문장을 결정적 차이라고 표시하지 마십시오.
- 차이를 빠뜨리지 마십시오. 결정론적 코드가 원문의 실제 차집합과 대조합니다.
- 날짜/일련번호/요청기관/문서 머리말 같은 메타데이터는 metadata_candidates에만
  제안할 수 있습니다. 최종 메타데이터 판정은 코드가 합니다.
- applicability_basis, applies/differs/unclear, 최종 결론을 출력하지 마십시오.
"""


def build_prompt(request: str, precedent_request: str) -> str:
    """두 요청문만 제공한다. 선례 라벨/정답은 받을 인자 자체가 없다."""
    return (
        f"[A — 지금 판단할 요청]\n{clean_for_prompt(request)}\n\n"
        f"[B — 비교 선례의 요청]\n{clean_for_prompt(precedent_request)}"
    )


def _factor_schema(side: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "side": {"type": "string", "enum": [side]},
            "axis": {"type": "string"},
            "value_in_request": {"type": ["string", "null"]},
            "value_in_precedent": {"type": ["string", "null"]},
            "decisive": {"type": "boolean"},
            "why_not_decisive": {"type": ["string", "null"]},
        },
        "required": [
            "id",
            "text",
            "side",
            "axis",
            "value_in_request",
            "value_in_precedent",
            "decisive",
            "why_not_decisive",
        ],
        "additionalProperties": False,
    }


def schema() -> dict:
    """Anthropic structured-output 제약에 맞춘 S5 계약. 배열 길이 키워드는 안 쓴다."""
    shared = _factor_schema("both")
    request_only = _factor_schema("request")
    precedent_only = _factor_schema("precedent")
    return {
        "type": "object",
        "properties": {
            "shared_factors": {"type": "array", "items": shared},
            "only_in_request": {"type": "array", "items": request_only},
            "only_in_precedent": {"type": "array", "items": precedent_only},
            "metadata_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "side": {
                            "type": "string",
                            "enum": ["request", "precedent"],
                        },
                    },
                    "required": ["text", "side"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "shared_factors",
            "only_in_request",
            "only_in_precedent",
            "metadata_candidates",
        ],
        "additionalProperties": False,
    }
