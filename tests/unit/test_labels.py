"""라벨 체계의 일관성.

라벨 문자열이 코드 여기저기에 흩어지면 오타 하나로 채점이 조용히 틀어진다.
정의처는 app/domain/labels.py 하나뿐이어야 한다.
"""

from __future__ import annotations

from app.domain.labels import (
    GUIDELINE,
    LABEL_SETS,
    NON_ACTIONS,
    VERDICTS,
    NonAction,
    Verdict,
)


def test_tuples_mirror_the_enums():
    assert VERDICTS == tuple(v.value for v in Verdict)
    assert NON_ACTIONS == tuple(v.value for v in NonAction)


def test_label_sets_expose_both_tracks():
    assert LABEL_SETS == {"verdict": VERDICTS, "nonaction": NON_ACTIONS}


def test_labels_are_unique_within_a_track():
    for labels in LABEL_SETS.values():
        assert len(set(labels)) == len(labels)


def test_guideline_covers_every_verdict():
    """지침에 빠진 라벨이 있으면 사람과 LLM 이 서로 다른 기준을 쓰게 된다."""
    for label in VERDICTS:
        assert label in GUIDELINE, f"판정 지침에 '{label}' 설명이 없습니다"


def test_abstain_is_a_domain_label_not_an_escape_hatch():
    """판단유보는 '당국이 유보한 것'이지 '라벨 정하기 어려운 것'이 아니다."""
    assert Verdict.ABSTAIN.value == "판단유보"
    assert "라벨을 정하기 어려운 것" in GUIDELINE
