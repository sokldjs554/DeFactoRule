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
