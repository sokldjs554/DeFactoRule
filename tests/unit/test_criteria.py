"""회답 근거 구조화의 안전장치.

이 단계의 가장 큰 위험은 순환이다. 회답에는 결론이 함께 적혀 있고, 그것을
그대로 쓰면 100%가 나오면서 아무것도 배우지 못한다. 코드가 막아야 한다.
"""

from __future__ import annotations

import pytest

from app.agents.criteria import (
    question_is_circular,
    quote_is_grounded,
    validate_criterion,
)
from app.core.text import clean_for_prompt, normalize_for_match
from app.rules.criteria_vote import confidence, fit, score

REASONING = "「전자금융감독규정」 제15조에 따라 내부망과 외부망을 분리하여야 하며"


# ── 순환 차단 ────────────────────────────────────────────────────
@pytest.mark.parametrize("question", [
    "이 사안은 조치 대상인가?",
    "비조치 의견을 받을 수 있는가?",
    "당국이 어떻게 판단했는가?",
    "제재 가능성이 있는가?",
    "판단 결과가 무엇인가?",
])
def test_conclusion_questions_are_rejected(question):
    assert question_is_circular(question), question


@pytest.mark.parametrize("question", [
    "요청인이 내부망과 외부망을 물리적으로 분리하고 있는가?",
    "요청 행위가 전자금융감독규정 제15조가 정한 예외에 해당하는가?",
    "위탁 대상이 계열회사인가?",
])
def test_factual_questions_are_kept(question):
    assert not question_is_circular(question), question


# ── 인용 대조 ────────────────────────────────────────────────────
def test_verbatim_quote_passes():
    assert quote_is_grounded("내부망과 외부망을 분리", REASONING)


def test_quote_survives_typographic_junk():
    """판단이유 98.8%에 잔재가 있다. 그것 때문에 정상 인용이 버려지면 안 된다."""
    dirty = "≄\n‌「전자금융감독규정」 제15조에 따라 ≄\n내부망과 외부망을 분리"
    assert quote_is_grounded("「전자금융감독규정」 제15조에 따라 내부망과 외부망을 분리", dirty)


def test_fabricated_quote_fails():
    assert not quote_is_grounded("내부망과 외부망을 통합", REASONING)


def test_empty_quote_fails():
    """분류기에서는 인용 포기를 환각이 아니라고 봤지만, 여기서는 다르다.
    기준의 근거가 없으면 그 기준은 근거가 없는 것이다."""
    assert not quote_is_grounded("", REASONING)


def test_validate_collects_every_problem():
    problems = validate_criterion(
        {"name": "", "question": "조치 대상인가?", "quote": "없는 말", "implies": "몰라"},
        REASONING,
    )
    assert len(problems) == 4, problems


def test_validate_accepts_a_good_criterion():
    assert not validate_criterion(
        {"name": "망분리 여부", "question": "내부망과 외부망이 분리되어 있는가?",
         "quote": "내부망과 외부망을 분리", "implies": "조치"},
        REASONING,
    )


# ── 텍스트 위생 ──────────────────────────────────────────────────
def test_bullets_survive_for_reading_but_not_for_matching():
    text = "□ 첫째\n○ 둘째"
    assert "□" in clean_for_prompt(text)
    assert "□" not in normalize_for_match(text)


def test_invisible_characters_are_always_removed():
    text = "가‌나≄다"
    assert clean_for_prompt(text) == "가나다"
    assert normalize_for_match(text) == "가나다"


# ── 집계 (결정론) ────────────────────────────────────────────────
ROWS = (
    [{"source": "d", "page": i, "serial": str(i), "pair_index": 1, "label": "조치"}
     for i in range(1, 7)]
    + [{"source": "d", "page": 10 + i, "serial": str(10 + i), "pair_index": 1,
        "label": "비조치"} for i in range(1, 15)]
)
# 0번 기준은 조치에만 yes, 1번은 아무 정보 없음
ANSWERS = {
    ("d", r["page"], r["serial"], 1): (
        ["yes", "yes"] if r["label"] == "조치" else ["no", "yes"]
    )
    for r in ROWS
}


def test_weights_point_at_the_associated_label():
    model = fit(ANSWERS, ROWS, n_criteria=2)
    informative = model["criteria"][0]["weights"]
    assert informative["조치"] > informative["비조치"]


def test_uninformative_criterion_gets_flat_weights():
    """모두에게 yes 인 기준은 아무것도 가르지 못한다. 가중치가 기저율과 같아야 한다."""
    model = fit(ANSWERS, ROWS, n_criteria=2)
    flat = model["criteria"][1]["weights"]
    assert max(flat.values()) - min(flat.values()) < 0.15, flat


def test_unknown_is_not_evidence():
    model = fit(ANSWERS, ROWS, n_criteria=2)
    assert score(model, ["unknown", "unknown"])["fired"] == []


def test_no_evidence_means_low_confidence():
    model = fit(ANSWERS, ROWS, n_criteria=2)
    assert confidence(score(model, ["no", "no"]), high=1.0, medium=0.4) == "low"


def test_scoring_is_deterministic():
    model = fit(ANSWERS, ROWS, n_criteria=2)
    assert score(model, ["yes", "no"]) == score(model, ["yes", "no"])


# ── 파이프라인 안내 ──────────────────────────────────────────────
def test_missing_criteria_file_points_at_the_previous_step(tmp_path):
    """세 종류의 파일을 쓰는 단계라 헷갈리기 쉽다. 다음에 칠 명령을 알려줘야 한다."""
    import pytest

    from app.agents.criteria import load_criteria

    with pytest.raises(SystemExit) as exc:
        load_criteria(tmp_path / "criteria.jsonl")
    message = str(exc.value)
    assert "consolidate" in message
    assert "status" in message


def test_wrong_shaped_criteria_file_is_rejected(tmp_path):
    import json

    import pytest

    from app.agents.criteria import load_criteria

    path = tmp_path / "criteria.jsonl"
    path.write_text(json.dumps({"label": "비조치"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        load_criteria(path)
    assert "question" in str(exc.value)
