"""신뢰도 등급과 그 순서.

등급을 문자열로 여기저기 흩어 놓으면, 어느 한 곳에서 순서를 다르게 매기는
순간 위험-커버리지 곡선이 조용히 뒤집힌다. 정의처를 하나로 둔다.

`?` 는 신뢰도를 내지 않는 예측을 위한 자리다. 가장 낮게 취급한다 — 모른다고
말하지 않은 것을 자신 있다고 읽으면 안 된다.
"""

from __future__ import annotations

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "?": 0}

# 사람이 읽는 순서 (높은 것부터)
CONFIDENCE_ORDER = ("high", "medium", "low")


def rank(value: str | None) -> int:
    return CONFIDENCE_RANK.get(value or "?", 0)


def meets(value: str | None, minimum: str) -> bool:
    """이 신뢰도가 운영 문턱을 넘는가. 기권 판정의 유일한 기준이다."""
    return rank(value) >= rank(minimum)
