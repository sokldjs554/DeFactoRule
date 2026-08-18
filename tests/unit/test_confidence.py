"""신뢰도 순서 — 기권 판정의 유일한 기준.

이 순서가 어느 한 곳에서 달라지면 위험-커버리지 곡선이 조용히 뒤집힌다.
그래서 정의처를 하나로 모았고, 여기서 그 계약을 못박는다.
"""

from __future__ import annotations

import pytest

from app.domain.confidence import CONFIDENCE_ORDER, CONFIDENCE_RANK, meets, rank


def test_order_is_strictly_descending():
    ranks = [rank(v) for v in CONFIDENCE_ORDER]
    assert ranks == sorted(ranks, reverse=True)
    assert len(set(ranks)) == len(ranks)


def test_unknown_confidence_is_lowest():
    """신뢰도를 내지 않은 예측을 자신 있다고 읽으면 안 된다."""
    assert rank(None) == 0
    assert rank("?") == 0
    assert rank(None) < min(rank(v) for v in CONFIDENCE_ORDER)


def test_unseen_value_does_not_crash_and_ranks_lowest():
    assert rank("아주높음") == 0


@pytest.mark.parametrize("value,minimum,expected", [
    ("high", "low", True), ("high", "high", True),
    ("medium", "high", False), ("low", "medium", False),
    ("low", "low", True), (None, "low", False),
])
def test_threshold_comparison(value, minimum, expected):
    assert meets(value, minimum) is expected


def test_every_named_rank_is_in_the_order_tuple():
    named = set(CONFIDENCE_RANK) - {"?"}
    assert named == set(CONFIDENCE_ORDER)
