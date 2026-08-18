"""규칙 학습기의 불변조건.

학습기가 조용히 망가지면 "규칙이 전이되지 않는다" 는 결론 자체가 무의미해진다.
탐색이 결정론적인지, 문턱이 소수 클래스를 구조적으로 배제하지 않는지를 못박는다.
"""

from __future__ import annotations

import pytest

from app.rules.induction import (
    Atom,
    apply_rules,
    coverage_masks,
    induce,
    laplace,
    length_bucket,
    maximal_form,
    popcount,
    squeeze,
)

LABELS = ("비조치", "조치", "기타")


def row(i: int, text: str, label: str, sector: str = "공통") -> dict:
    return {"source": "t", "page": i, "serial": str(i), "pair_index": 1,
            "request": text, "label": label, "sector": sector}


# 앞 6건은 '망연계' 를 공유하고 전부 조치, 나머지는 비조치
SECTORS = ("공통", "전자금융", "보험", "은행")
ROWS = (
    [row(i, f"내부망과 외부망의 망연계 구간 관련 질의 {i}", "조치", SECTORS[i % 4])
     for i in range(1, 7)]
    + [row(10 + i, f"겸영업무 신고 대상 여부에 관한 질의 {i}", "비조치", SECTORS[i % 4])
       for i in range(1, 9)]
)


def test_squeeze_removes_all_whitespace():
    assert squeeze(" 가 나\n다\t") == "가나다"


def test_length_buckets_are_ordered_and_exhaustive():
    assert length_bucket("가" * 10).startswith("짧음")
    assert length_bucket("가" * 300).startswith("보통")
    assert length_bucket("가" * 900).startswith("긺")


def test_popcount_matches_bin():
    for n in (0, 1, 7, 255, 1 << 40):
        assert popcount(n) == bin(n).count("1")


def test_laplace_penalises_tiny_support():
    assert laplace(3, 3, 3) < laplace(30, 30, 3)


def test_laplace_is_not_used_as_a_threshold():
    """laplace(4,4,3)=0.714 다. 이것을 0.80 문턱에 쓰면 dev 8건짜리 소수
    클래스는 완벽한 규칙을 찾아도 통과할 수 없다 — 구조적 배제다."""
    assert laplace(4, 4, 3) < 0.80
    rules, _ = induce(ROWS, LABELS, min_support=4, min_precision=0.80, max_depth=2)
    assert any(r.label == "조치" for r in rules), (
        "소수 클래스 규칙이 문턱에 막혔습니다 — laplace 를 문턱으로 쓰고 있습니다"
    )


def test_learner_separates_the_planted_classes():
    """규칙 목록을 **순서대로** 적용했을 때 두 무리가 갈리는지를 본다.

    규칙 하나를 떼어 전체 행에 적용하면 안 된다. 순차 피복에서 뒤쪽 규칙은
    앞 규칙이 걸러낸 나머지에만 적용되므로, 홀로 보면 과하게 걸리는 것이
    정상이다. 실제 사용 방식과 같은 방식으로 재야 한다.

    어떤 n-gram 을 골랐는지도 못박지 않는다. 덮는 집합이 같은 표현은 대표
    하나로 합쳐지고 그 대표는 최대 확장형이라, 표면형을 고정하면 테스트가
    구현 세부에 묶인다.
    """
    rules, default = induce(ROWS, LABELS, min_support=4, min_precision=0.9, max_depth=2)
    assert rules, "심어 둔 규칙을 못 찾았습니다"

    predicted = [apply_rules(rules, default, r)[0] for r in ROWS]
    truth = [r["label"] for r in ROWS]
    assert predicted == truth, list(zip(truth, predicted))
    assert default == "비조치"


def test_induction_is_deterministic():
    a, da = induce(ROWS, LABELS, min_support=4, min_precision=0.9, max_depth=2)
    b, db = induce(list(reversed(ROWS)), LABELS, min_support=4, min_precision=0.9, max_depth=2)
    assert da == db
    assert [r.describe() for r in a] == [r.describe() for r in b], (
        "입력 순서가 규칙을 바꾸면 실험을 재현할 수 없습니다"
    )


def test_duplicate_coverage_atoms_are_collapsed():
    """겹치는 n-gram 이 같은 규칙을 여러 벌 만들면 규칙 목록을 읽을 수 없다."""
    atoms = [Atom("ngram", "망연계"), Atom("ngram", "망연"), Atom("ngram", "연계")]
    masks = coverage_masks(ROWS, atoms)
    assert len(masks) == 1, f"덮는 집합이 같은데 {len(masks)}개가 남았습니다"


def test_representative_atom_is_the_maximal_form():
    """대표는 최대 확장형이어야 한다 — 그래야 파편이 파편으로 보인다.

    '것이전' 은 규칙처럼 보이지만 '…하는것이전자금융감독규정제1…' 의 조각이다.
    지지도를 잃지 않는 한 늘려 놓으면 사람이 읽고 가릴 수 있다.
    """
    masks = coverage_masks(ROWS, [Atom("ngram", "망연계")])
    rep = next(iter(masks)).value
    assert len(rep) > len("망연계"), rep
    assert "망연계" in rep


def test_maximal_form_stops_when_support_would_drop():
    texts = ["가나다라", "가나다마"]
    assert maximal_form("나다", texts) == "가나다"  # 라/마 에서 갈리므로 더 못 늘린다


def test_no_rule_is_learned_from_pure_noise():
    noise = [row(i, f"문장 {i} 입니다", "비조치" if i % 2 else "조치") for i in range(1, 21)]
    rules, _ = induce(noise, LABELS, min_support=4, min_precision=0.95, max_depth=2)
    assert not rules, [r.describe() for r in rules]


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_rules_never_exceed_the_depth_limit(depth):
    rules, _ = induce(ROWS, LABELS, min_support=4, min_precision=0.8, max_depth=depth)
    assert all(len(r.atoms) <= depth for r in rules)


# ── 고정 어휘와 교차검증 ─────────────────────────────────────────
def test_prepare_atoms_returns_maximal_deduped_vocabulary():
    from app.rules.induction import prepare_atoms

    vocab = prepare_atoms(ROWS)
    values = [a.value for a in vocab if a.kind == "ngram"]
    assert values, "n-gram 조건이 하나도 없습니다"
    assert len(values) == len(set(values)), "중복된 표현이 남았습니다"
    # 최대 확장형이므로 짧은 조각이 그대로 남아 있으면 안 된다
    assert not any(v == "망연" for v in values), values[:10]


def test_fixed_vocabulary_preserves_behaviour():
    """어휘를 미리 만들어 넘겨도 결과가 같아야 한다.

    이 등식이 깨지면 교차검증이 전체 학습과 다른 것을 재게 된다.
    """
    from app.rules.induction import prepare_atoms

    a, da = induce(ROWS, LABELS, min_support=4, min_precision=0.8, max_depth=2)
    b, db = induce(ROWS, LABELS, min_support=4, min_precision=0.8, max_depth=2,
                   atoms=prepare_atoms(ROWS))
    assert da == db
    assert [r.describe() for r in a] == [r.describe() for r in b]


def test_fixed_vocabulary_makes_fold_keys_comparable():
    """조각마다 어휘를 다시 만들면 같은 개념이 다른 문자열이 된다.

    실측으로 확인했다 — 전체 dev 규칙 11개와 한 조각을 뺀 규칙 10개 중
    키가 겹치는 것이 2개뿐이었다. 고정 어휘는 최소한 그 원인을 없앤다.
    """
    from app.rules.induction import fold_of, prepare_atoms

    vocab = prepare_atoms(ROWS)
    train = [r for i, r in enumerate(ROWS) if fold_of(i, 3) != 0]
    fixed, _ = induce(train, LABELS, min_support=3, min_precision=0.8,
                      max_depth=2, atoms=vocab)
    for rule in fixed:
        for atom in rule.atoms:
            assert atom in vocab, f"어휘 밖의 조건이 나왔습니다: {atom}"


def test_cross_validate_is_deterministic_and_reports_out_of_fold():
    from app.rules.induction import cross_validate

    a = cross_validate(ROWS, LABELS, folds=3, min_support=3, min_precision=0.8, max_depth=2)
    b = cross_validate(ROWS, LABELS, folds=3, min_support=3, min_precision=0.8, max_depth=2)
    assert a.keys() == b.keys()
    for key, stats in a.items():
        assert stats["oof_support"] == b[key]["oof_support"]
        assert stats["folds"] >= 1
        if stats["oof_support"]:
            assert 0.0 <= stats["oof_precision"] <= 1.0


def test_folds_partition_the_rows():
    from app.rules.induction import fold_of

    assigned = [fold_of(i, 5) for i in range(len(ROWS))]
    assert set(assigned) <= set(range(5))
    for f in range(5):
        train = [r for i, r in enumerate(ROWS) if fold_of(i, 5) != f]
        held = [r for i, r in enumerate(ROWS) if fold_of(i, 5) == f]
        assert len(train) + len(held) == len(ROWS)
        assert not ({id(r) for r in train} & {id(r) for r in held})


# ── 버린 후보 기록 ───────────────────────────────────────────────
def test_rejected_candidates_are_recorded_with_reasons():
    """규칙이 안 나왔을 때 '후보가 없었나, 문턱이 높았나' 를 가를 수 있어야 한다."""
    from app.rules.induction import induce_with_audit

    rules, _, discards = induce_with_audit(
        ROWS, LABELS, min_support=4, min_precision=0.99, max_depth=2
    )
    assert len(discards) > 0, "문턱을 0.99 로 올렸는데 버린 후보가 없습니다"
    for item in discards.records():
        assert item["rejected_for"], item
        assert "정밀도" in item["rejected_for"][0]
    summary = discards.summary()
    assert summary["stage"] == "rule-induction"
    assert summary["dropped"] == len(discards)
