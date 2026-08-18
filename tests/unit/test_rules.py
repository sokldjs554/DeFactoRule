"""결정론적 기준선.

기준선이 조용히 바뀌면 LLM 과의 비교가 통째로 흔들린다. E1~E4 의 모든 주장이
"규칙 대비"로 서술되어 있으므로, 규칙 쪽이 고정되어 있어야 그 주장이 유지된다.
"""

from __future__ import annotations

import pytest

from app.domain.labels import NonAction, Verdict
from app.rules import nonaction as na
from app.rules import verdict as vd


# ── 비조치 ─────────────────────────────────────────────────────────
def test_majority_always_predicts_the_majority_class():
    for text in ("아무 말", "망분리 관련 질의입니다", ""):
        label, rule, conf = na.classify(text, "majority")
        assert (label, rule) == (NonAction.NO_ACTION.value, "majority")


def test_majority_never_claims_high_confidence():
    """곡선이 평평해지는 것이 곧 '자기가 틀릴 때를 모른다'는 진단이다.

    여기서 high 를 내보내기 시작하면 AURC 비교가 거짓말이 된다.
    """
    _, _, conf = na.classify("망분리", "majority")
    assert conf == "low"


def test_keyword_rule_fires_with_high_confidence():
    label, rule, conf = na.classify("내부망과 외부망의 망연계 구간에 대하여", "keyword")
    assert label == NonAction.ACTION.value
    assert rule.startswith("rule:")
    assert conf == "high"


def test_keyword_falls_back_to_majority_with_low_confidence():
    label, rule, conf = na.classify("전혀 무관한 문장입니다", "keyword")
    assert label == NonAction.NO_ACTION.value
    assert rule == "fallback:majority"
    assert conf == "low"


def test_keyword_rule_survives_line_breaks():
    """PDF 에서 온 텍스트는 줄바꿈이 섞여 있다. 정규화 없이 매칭하면 놓친다."""
    label, _, _ = na.classify("정보처리시스템\n을 외부와  연결", "keyword")
    assert label == NonAction.ACTION.value


# ── 법령해석 ───────────────────────────────────────────────────────
def test_unknown_is_not_folded_into_abstain():
    """규칙이 못 읽은 것과 당국이 유보한 것은 다른 사건이다."""
    label, rule = vd.classify("")
    assert label == vd.UNKNOWN
    assert label != Verdict.ABSTAIN.value
    assert rule == "empty"


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("구체적 사실관계에 따라 판단할 사항입니다.", Verdict.ABSTAIN.value),
        ("해당하지 않는 것으로 봅니다.", Verdict.DENY.value),
        ("가능한 것으로 판단됩니다.", Verdict.AFFIRM.value),
    ],
)
def test_conclusion_patterns(answer, expected):
    assert vd.classify(answer)[0] == expected


def test_abstain_beats_a_co_occurring_affirmation():
    """'가능하나 개별 사실관계에 따라 다르다' 는 긍정이 아니라 유보다."""
    answer = "질의하신 행위는 가능합니다. 다만 구체적 사실관계에 따라 달라질 수 있습니다."
    assert vd.classify(answer)[0] == Verdict.ABSTAIN.value


def test_only_the_tail_is_read():
    """앞쪽 인용문에 낚이면 안 된다. 결론절은 회답 끝에 온다."""
    quoted = "질의인은 '해당하지 않는다'고 주장합니다. " + "가" * 400
    answer = quoted + " 따라서 가능한 것으로 판단됩니다."
    assert vd.classify(answer)[0] == Verdict.AFFIRM.value
