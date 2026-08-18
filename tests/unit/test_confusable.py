"""최근접 선례 분석의 불변조건.

이 지표는 이 프로젝트를 위해 만든 것이라 기성 구현과 대조할 수 없다.
그래서 정의가 스스로 모순되지 않는지를 테스트로 못박는다.
"""

from __future__ import annotations

import pytest

from app.evaluation.confusable import (
    anchoring_by_class,
    cosine,
    idf_table,
    nearest,
    normalize,
    partition,
    weighted_vector,
)
from app.retrieval.neighbor import HIGH, MEDIUM, band

DEV = [
    {"source": "d", "page": 1, "serial": "1", "pair_index": 1,
     "request": "망분리 예외 적용 요청", "label": "조치"},
    {"source": "d", "page": 2, "serial": "2", "pair_index": 1,
     "request": "부동산PF 대주단협의회 채권 재조정", "label": "비조치"},
]
TEST = [
    # DEV[1] 과 사실상 동일 — 같은 라벨이므로 순응
    {"source": "t", "page": 1, "serial": "1", "pair_index": 1,
     "request": "부동산PF 대주단협의회 채권 재조정", "label": "비조치"},
    # DEV[1] 과 사실상 동일한데 라벨이 다르다 — 함정
    {"source": "t", "page": 2, "serial": "2", "pair_index": 1,
     "request": "부동산PF 대주단협의회 채권 재조정", "label": "기타"},
    # 어느 선례와도 안 닮았다
    {"source": "t", "page": 3, "serial": "3", "pair_index": 1,
     "request": "완전히 무관한 겸영업무 신고 질의입니다", "label": "비조치"},
]

IDF = idf_table([r["request"] for r in DEV + TEST])


def test_normalize_only_strips_whitespace():
    assert normalize(" 가 나\n다 ") == "가나다"
    assert normalize("망분리") == "망분리"


def test_identical_text_has_similarity_one():
    v = weighted_vector("부동산PF 대주단협의회", IDF)
    assert cosine(v, v) == pytest.approx(1.0)


def test_similarity_is_symmetric():
    a = weighted_vector(DEV[0]["request"], IDF)
    b = weighted_vector(DEV[1]["request"], IDF)
    assert cosine(a, b) == pytest.approx(cosine(b, a))


def test_idf_never_sees_labels():
    """라벨을 지워도 IDF 가 같아야 한다 — 그래야 누출이 아니다."""
    stripped = [{k: v for k, v in r.items() if k != "label"} for r in DEV + TEST]
    assert idf_table([r["request"] for r in stripped]) == IDF


def test_partition_splits_agree_trap_and_unanchored():
    groups = partition(nearest(TEST, DEV, IDF), floor=0.25)
    assert [x["row"]["serial"] for x in groups["agree"]] == ["1"]
    assert [x["row"]["serial"] for x in groups["trap"]] == ["2"]
    assert [x["row"]["serial"] for x in groups["unanchored"]] == ["3"]


def test_every_case_lands_in_exactly_one_group():
    groups = partition(nearest(TEST, DEV, IDF), floor=0.25)
    assert sum(len(v) for v in groups.values()) == len(TEST)


def test_nearest_is_deterministic():
    a = nearest(TEST, DEV, IDF)
    b = nearest(TEST, list(DEV), IDF)
    assert [x["neighbor"]["serial"] for x in a] == [x["neighbor"]["serial"] for x in b]


def test_anchoring_counts_are_per_class():
    table = anchoring_by_class(nearest(TEST, DEV, IDF), floor=0.25)
    assert table["비조치"]["n"] == 2
    assert table["비조치"]["anchored"] == 1  # 하나는 이웃이 없다
    assert table["기타"]["trap"] == 1


def test_confidence_bands_are_ordered():
    assert HIGH > MEDIUM > 0
    assert band(HIGH) == "high"
    assert band(MEDIUM) == "medium"
    assert band(MEDIUM - 0.01) == "low"
