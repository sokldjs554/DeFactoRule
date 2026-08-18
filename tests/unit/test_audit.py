"""버린 것을 기록하는 공용 장치.

같은 실수를 네 번 했다 — API 오류는 상태 코드만, 결측 검사는 오류 행만,
기준 검증은 화면에만, 규칙 학습기는 후보 165개를 버리면서 아무 데도 안 남겼다.
넷 다 "걸러내는 코드가 걸러낸 것을 기록하지 않았다" 는 하나의 패턴이다.
"""

from __future__ import annotations

import pytest

from app.core.audit import Discards


def test_dropping_requires_a_reason():
    """이유 없는 폐기는 기록이 아니다. 이유가 곧 진단이다."""
    d = Discards("테스트")
    with pytest.raises(ValueError):
        d.drop({"a": 1}, [])


def test_records_keep_the_item_and_every_reason():
    d = Discards("테스트")
    d.drop({"question": "조치 대상인가?"}, ["질문이 결론을 되묻는다", "인용이 원문에 없다"])
    (record,) = d.records()
    assert record["question"] == "조치 대상인가?"
    assert record["rejected_for"] == ["질문이 결론을 되묻는다", "인용이 원문에 없다"], (
        "이유를 하나만 남기면 나머지를 못 본다"
    )


def test_summary_counts_every_reason_not_just_the_first():
    d = Discards("테스트")
    d.drop({"x": 1}, ["A", "B"])
    d.drop({"x": 2}, ["A"])
    assert d.summary()["reasons"] == {"A": 2, "B": 1}
    assert d.summary()["dropped"] == 2


def test_keep_if_returns_whether_the_item_survives():
    d = Discards("테스트")
    assert d.keep_if({"x": 1}, []) is True
    assert d.keep_if({"x": 2}, ["문제"]) is False
    assert len(d) == 1


def test_non_dict_items_are_still_recorded():
    d = Discards("테스트")
    d.drop("문자열 항목", ["이유"])
    assert "문자열 항목" in d.records()[0]["value"]


def test_report_is_empty_when_nothing_was_dropped():
    assert "없음" in Discards("테스트").report()
