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
    "판단 결과는 무엇인가?",
    "본 요청은 조치 여부가 어떻게 되는가?",
    "어떤 결론이 내려졌는가?",
    "조치 대상에 해당하는가?",
    "비조치의견서를 발급받을 수 있는가?",
])
def test_conclusion_questions_are_rejected(question):
    assert question_is_circular(question), question


@pytest.mark.parametrize("question", [
    "요청인이 내부망과 외부망을 물리적으로 분리하고 있는가?",
    "요청 행위가 전자금융감독규정 제15조가 정한 예외에 해당하는가?",
    "위탁 대상이 계열회사인가?",
    # 아래 넷은 처음 구현이 잘못 걸러냈다. '불이익' 은 되묻기가 아니라
    # 진짜 규제 판단 기준이고, '제재 이력' 은 요청인에 관한 사실이다.
    "요청인이 제재 이력을 보유하고 있는가?",
    "요청 내용이 감독규정 위반에 해당할 소지가 있는가?",
    "이미 유사한 결론이 내려진 선례가 있는가?",
    "요청 행위가 소비자에게 불이익을 초래할 수 있는가?",
])
def test_factual_questions_are_kept(question):
    """걸러야 할 것은 낱말이 아니라 **묻는 대상**이다.

    이 사안의 처분이 무엇이냐를 물으면 순환이고, 사안의 성질을 물으면 기준이다.
    낱말 단위로 거르면 정당한 기준을 절반 가까이 잘라 낸다 — 실측 4/9.
    """
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


# ── 가중치 산술 — 증거가 없으면 가중치도 없다 ─────────────────────
def _dev_rows():
    """dev 와 같은 모양의 치우친 표본. 비조치 58 · 기타 19 · 조치 8."""
    rows = []
    for i, label in enumerate(["비조치"] * 58 + ["기타"] * 19 + ["조치"] * 8):
        rows.append({"source": "t", "page": i, "serial": str(i),
                     "pair_index": 1, "label": label})
    return rows


def _answers(rows, pick_label, k):
    """pick_label 인 사례 k 건만 'yes' 라고 답한 것으로 둔다."""
    from app.core.io import key_of

    out, seen = {}, 0
    for r in rows:
        if r["label"] == pick_label and seen < k:
            out[key_of(r)] = ["yes"]
            seen += 1
        else:
            out[key_of(r)] = ["no"]
    return out


def test_evidence_free_label_gets_no_positive_weight():
    """증거가 0 건인 라벨에 양수 가중치가 붙지 않는가.

    라플라스 평활(+1)을 쓰던 시절, 비조치 5건·조치 0건인 기준이 조치에
    **더 큰** 가중치를 줬다(비조치 +0.112 vs 조치 +0.201). 균등 사전분포로
    평활한 값을 치우친 실제 기저율로 나누면 희귀 클래스가 부풀려진다.
    """
    from app.rules.criteria_vote import fit

    rows = _dev_rows()
    w = fit(_answers(rows, "비조치", 5), rows, 1)["criteria"][0]["weights"]
    assert w["조치"] < 0, f"증거가 없는 조치에 {w['조치']:+.3f} 가 붙었습니다"
    assert w["기타"] < 0
    assert max(w, key=w.get) == "비조치", "증거가 있는 쪽이 이겨야 합니다"


def test_real_minority_evidence_still_wins():
    """소수 클래스에 **실제** 증거가 있으면 여전히 강하게 잡는가.

    유령 가중치를 없애면서 진짜 신호까지 눌러 버리면 고친 것이 아니다.
    """
    from app.rules.criteria_vote import fit

    rows = _dev_rows()
    w = fit(_answers(rows, "조치", 3), rows, 1)["criteria"][0]["weights"]
    assert max(w, key=w.get) == "조치", f"조치 3건 증거인데 {max(w, key=w.get)} 로 갔습니다"
    assert w["조치"] > 1.0, f"소수 클래스 신호가 눌렸습니다: {w['조치']:+.3f}"


def test_zero_evidence_criterion_is_exactly_neutral():
    """아무도 yes 라고 답하지 않은 기준의 가중치는 정확히 0 인가."""
    from app.core.io import key_of
    from app.rules.criteria_vote import fit

    rows = _dev_rows()
    answers = {key_of(r): ["no"] for r in rows}
    w = fit(answers, rows, 1)["criteria"][0]["weights"]
    assert all(abs(v) < 1e-9 for v in w.values()), f"증거 없는 기준에 가중치가 붙었습니다: {w}"


def test_nothing_fired_falls_back_to_the_base_rate():
    """발화한 기준이 없을 때 자모 순서가 아니라 기저율로 돌아가는가.

    점수가 전부 0 이면 이름순 정렬이 '기타' 를 골랐다. 그것은 판단이 아니다.
    """
    from app.core.io import key_of
    from app.rules.criteria_vote import confidence, fit, score

    rows = _dev_rows()
    model = fit({key_of(r): ["no"] for r in rows}, rows, 1)
    result = score(model, ["no"])
    assert result["predicted"] == "비조치", f"기저율 최빈이 아니라 {result['predicted']}"
    assert confidence(result, 0.5, 0.2) == "low", "근거가 없으면 low 여야 합니다"


# ── 통합 문턱 — 클래스마다 같은 비율을 요구하는가 ──────────────────
def test_class_floors_equalise_the_evidence_rate():
    """일률 문턱이 소수 클래스에만 엄해지는 것을 막는가."""
    from app.agents.criteria import class_floors

    floors = class_floors({"비조치": 58, "기타": 19, "조치": 8}, 2)
    assert floors["비조치"] == 2
    assert floors["조치"] == 1, "조치에 2건을 요구하면 25% 를 요구하는 셈입니다"
    # 요구 비율이 대략 같은 자리에 있는가 (일률 문턱은 7배 차이였다)
    rates = {k: floors[k] / n for k, n in {"비조치": 58, "기타": 19, "조치": 8}.items()}
    assert max(rates.values()) / min(rates.values()) < 4, f"요구 비율이 여전히 치우침: {rates}"


def test_class_floors_never_reach_zero():
    """아무리 드문 클래스도 문턱이 0 이 되지는 않는가."""
    from app.agents.criteria import class_floors

    assert class_floors({"많음": 1000, "하나": 1}, 2)["하나"] == 1
    assert class_floors({}, 2) == {}


def test_criterion_that_fires_on_everything_is_exactly_neutral():
    """모두에게 yes 인 기준은 아무것도 가르지 못한다 — 가중치가 정확히 0.

    기저율은 평활하고 조건부는 다르게 평활하면 이 값이 0 에서 벗어난다.
    실제로 그렇게 어긋나 있었고, 없는 라벨에 -2.04 가 붙었다.
    """
    from app.core.io import key_of
    from app.rules.criteria_vote import fit

    rows = _dev_rows()
    answers = {key_of(r): ["yes"] for r in rows}
    w = fit(answers, rows, 1)["criteria"][0]["weights"]
    assert all(abs(v) < 1e-9 for v in w.values()), f"가르지 못하는 기준에 가중치: {w}"


def test_label_absent_from_dev_gets_zero_not_a_verdict():
    """dev 에 한 번도 없는 라벨에는 큰 음수가 아니라 0 을 준다.

    없는 정보를 지어내지 않는다. 근거가 없으면 밀지도 당기지도 않는다.
    """
    from app.core.io import key_of
    from app.rules.criteria_vote import fit

    rows = [{"source": "t", "page": i, "serial": str(i), "pair_index": 1,
             "label": "비조치" if i % 2 else "조치"} for i in range(1, 21)]
    answers = {key_of(r): (["yes"] if r["label"] == "조치" else ["no"]) for r in rows}
    w = fit(answers, rows, 1)["criteria"][0]["weights"]
    assert w["기타"] == 0.0, f"dev 에 없는 '기타' 에 {w['기타']:+.3f} 가 붙었습니다"
    assert w["조치"] > 0 and w["비조치"] < 0, "실제 증거는 그대로 반영돼야 합니다"


# ── 출력 예산 — 상한이 답을 담을 만큼 큰가 ─────────────────────────
def test_token_cap_fits_the_answer_json():
    """상한이 실제 답 JSON 을 담는가.

    기준 88개일 때 답 JSON 만 2,907자(대략 830~1,160토큰)인데 상한이
    1200 으로 못 박혀 있었다. 적응형 사고 몫까지 그 안에서 나눠 써야 하므로
    응답이 잘리고, 잘린 JSON 은 파싱 실패로 나타난다 — 돈을 다 쓴 뒤에.
    """
    import json

    from app.agents.criteria import apply_token_cap

    for n in (1, 15, 88, 244):
        body = json.dumps(
            {"answers": [{"id": i, "answer": "unknown"} for i in range(n)]},
            ensure_ascii=False,
        )
        # 가장 빡빡하게 잡아도 글자당 1/2.5 토큰. 그보다 넉넉해야 한다.
        need = len(body) / 2.5
        assert apply_token_cap(n) > need * 2, (
            f"기준 {n}개: 상한 {apply_token_cap(n)} 이 답 {need:.0f}토큰에 비해 빠듯합니다"
        )


def test_cap_is_never_below_the_estimate():
    """상한이 비용 추정치보다 작아지는 일이 없는가.

    둘이 따로 계산되던 시절에는 기준 수가 늘자 추정만 따라가고 상한은
    제자리였다. 같은 자리에서 나오게 했으니 그 관계가 유지되는지 본다.
    """
    from app.agents.criteria import apply_output_tokens, apply_token_cap

    for n in range(1, 300, 7):
        assert apply_token_cap(n) > apply_output_tokens(n), f"기준 {n}개에서 역전"


# ── 낱말 겹침 미리보기 — 돈 쓰기 전의 눈금 ────────────────────────
def test_relevance_preview_separates_matching_from_unrelated():
    """기준과 상관없는 사례를 상관있는 사례와 가르는가."""
    from app.agents.criteria import relevance_preview

    criteria = [{"question": "요청이 부동산PF 사업장 재구조화와 관련되는가?"}]
    rows = [
        {"request": "부동산PF 사업장 재구조화 관련 신규자금 지원이 가능한지",
         "label": "비조치"},
        {"request": "금융지주회사의 IT 전문 자회사에 전산 운영 업무를 위탁할 수 있는지",
         "label": "조치"},
    ]
    out = relevance_preview(rows, criteria)
    assert "1/2" in out, f"겹치는 사례를 하나로 세지 못했습니다:\n{out}"


def test_relevance_preview_warns_when_nothing_overlaps():
    """아무것도 겹치지 않으면 그렇게 말하는가.

    조용히 0% 를 적어 두면 읽는 사람은 그냥 지나친다. 돈을 쓸지 정하는
    자리이므로 분명히 말해야 한다.
    """
    from app.agents.criteria import relevance_preview

    out = relevance_preview(
        [{"request": "전혀 다른 이야기입니다", "label": "조치"}],
        [{"question": "부동산PF 사업장 재구조화와 관련되는가?"}],
    )
    assert "0/1" in out
    assert "기준이 이 사례들에 대해 말할 것이 없다" in out
