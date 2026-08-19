"""매크로 F1 과 부트스트랩 구간.

이 프로젝트의 대표 지표라서 손으로 계산한 값과 맞춰 둔다. 지표 자체가 틀리면
아래 실험 다섯 개가 전부 무의미해진다.
"""

from __future__ import annotations

import pytest

from app.evaluation.metrics import bootstrap_macro_f1, macro_f1

NA = ("비조치", "조치", "기타")


def test_macro_f1_matches_hand_calculation():
    pairs = [
        ("비조치", "비조치"),
        ("비조치", "비조치"),
        ("비조치", "비조치"),
        ("조치", "비조치"),
        ("기타", "기타"),
    ]
    macro, per = macro_f1(pairs, NA)
    # 비조치 P=3/4 R=3/3 F1=6/7 · 조치 전부 0 · 기타 완전 일치 1
    assert per["비조치"]["precision"] == pytest.approx(0.75)
    assert per["비조치"]["recall"] == pytest.approx(1.0)
    assert per["비조치"]["f1"] == pytest.approx(6 / 7)
    assert per["조치"] == {"support": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert per["기타"]["f1"] == pytest.approx(1.0)
    assert macro == pytest.approx((6 / 7 + 0.0 + 1.0) / 3)


def test_macro_f1_punishes_majority_only_prediction():
    """정확도 대신 매크로 F1 을 쓰는 이유 그 자체.

    74%가 다수 클래스인 분포에서 다수만 찍으면 정확도는 74%지만 매크로 F1 은
    0.3 아래로 주저앉는다. 이 간극이 사라지면 지표 선택의 근거가 사라진다.
    """
    pairs = [("비조치", "비조치")] * 74 + [("조치", "비조치")] * 13 + [("기타", "비조치")] * 13
    macro, _ = macro_f1(pairs, NA)
    accuracy = sum(1 for g, p in pairs if g == p) / len(pairs)
    assert accuracy == pytest.approx(0.74)
    assert macro < 0.30


def test_labels_absent_from_gold_do_not_dilute_the_average():
    """지원이 0인 라벨은 평균에서 뺀다.

    빼지 않으면 라벨을 하나 더 정의하는 것만으로 점수가 내려간다. 실제 오답은
    이미 다른 라벨의 재현율에 반영되어 있다.
    """
    pairs = [("긍정", "긍정"), ("부정", "부정")]
    four = ("긍정", "부정", "조건부", "판단유보")
    macro, per = macro_f1(pairs, four)
    assert macro == pytest.approx(1.0)
    assert per["조건부"]["support"] == 0


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    pairs = [("비조치", "비조치")] * 40 + [("조치", "비조치")] * 10
    point, _ = macro_f1(pairs, NA)
    lo, hi = bootstrap_macro_f1(pairs, NA, rounds=400, seed=0)
    again = bootstrap_macro_f1(pairs, NA, rounds=400, seed=0)
    assert (lo, hi) == again, "같은 시드는 같은 구간을 내야 한다"
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_refuses_degenerate_samples():
    assert bootstrap_macro_f1([("비조치", "비조치")], NA) == (0.0, 0.0)


# ── 비율 신뢰구간 — 분모가 한 자리일 때 ───────────────────────────
def test_wilson_interval_is_wide_when_the_denominator_is_tiny():
    """4건 중 3건이 0.750 으로 **확정**돼 읽히지 않게 한다."""
    from app.evaluation.metrics import wilson_interval

    lo, hi = wilson_interval(3, 4)
    assert lo < 0.35 and hi > 0.9, f"4건짜리 구간이 [{lo:.3f}, {hi:.3f}] 로 좁습니다"
    wide = hi - lo
    lo2, hi2 = wilson_interval(30, 40)
    assert (hi2 - lo2) < wide / 2, "표본이 열 배인데 구간이 절반 이하로 줄지 않았습니다"


def test_wilson_interval_stays_inside_zero_and_one():
    """0 과 1 에 붙어도 구간이 범위를 벗어나지 않는가 — Wald 는 벗어난다."""
    from app.evaluation.metrics import wilson_interval

    for successes, total in ((0, 4), (4, 4), (0, 1), (1, 1)):
        lo, hi = wilson_interval(successes, total)
        assert 0.0 <= lo <= hi <= 1.0, f"{successes}/{total} -> [{lo}, {hi}]"
    assert wilson_interval(0, 4)[1] > 0.2, "0건이라고 상한이 0 이 되면 안 됩니다"
    assert wilson_interval(4, 4)[0] < 0.9, "4건 전부라고 하한이 1 에 붙으면 안 됩니다"


def test_verdict_refuses_when_the_interval_straddles_the_threshold():
    """구간이 문턱을 걸치면 판정하지 않는가."""
    from app.evaluation.metrics import verdict_against, wilson_interval

    assert "보류" in verdict_against(0.286, *wilson_interval(0, 4))
    assert "보류" in verdict_against(0.286, *wilson_interval(2, 8))
    assert verdict_against(0.286, *wilson_interval(30, 40)) == "넘는다"
    assert verdict_against(0.286, *wilson_interval(1, 40)) == "못 넘는다"


def test_empty_denominator_does_not_pretend_to_know():
    from app.evaluation.metrics import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 1.0)
