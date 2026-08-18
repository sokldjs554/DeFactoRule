"""검색 접근의 사각지대 — 이 프로젝트의 핵심 근거 중 하나.

`조치` 사례는 dev 에 닮은 선례가 사실상 없다. 그래서 최근접 선례를 따라가는
전략은 평균 성능이 멀쩡해 보이면서도 정작 판별이 필요한 클래스에서 무력하다.
이 프로젝트가 검색이 아니라 규칙 역추출로 가야 하는 이유가 여기 있다.

이 테스트가 깨지면 그 주장이 더는 성립하지 않는 것이므로, 숫자를 고치기 전에
docs/13 을 먼저 고쳐야 한다.
"""

from __future__ import annotations

import pytest

from app.core.io import load_jsonl
from app.core.paths import EVAL, PROCESSED
from app.evaluation.confusable import anchoring_by_class, idf_table, nearest

FLOOR = 0.15
MINORITY_ANCHOR_CEILING = 0.25   # 조치가 이보다 잘 앵커되면 주장이 약해진다
MAJORITY_ANCHOR_FLOOR = 0.35     # 비조치는 이보다는 앵커돼야 대비가 성립한다


@pytest.fixture(scope="module")
def table() -> dict:
    for path in (EVAL / "nonaction_test.jsonl", EVAL / "nonaction_dev.jsonl"):
        if not path.exists():
            pytest.skip(f"{path.name} 이 없습니다")
    corpus = PROCESSED / "cases_nonaction.jsonl"
    texts = (
        [(c["fields"].get("요청대상행위") or "") for c in load_jsonl(corpus)]
        if corpus.exists()
        else None
    )
    test = [r for r in load_jsonl(EVAL / "nonaction_test.jsonl") if r.get("label")]
    dev = [r for r in load_jsonl(EVAL / "nonaction_dev.jsonl") if r.get("label")]
    idf = idf_table(texts or [r["request"] for r in test + dev])
    return anchoring_by_class(nearest(test, dev, idf), FLOOR)


def test_minority_class_has_almost_no_precedent(table):
    slot = table.get("조치")
    assert slot, "조치 사례가 test 에 없습니다"
    assert slot["anchor_rate"] <= MINORITY_ANCHOR_CEILING, (
        f"조치 앵커링 {slot['anchor_rate']:.1%} — 검색 사각지대 주장이 약해졌습니다. "
        "docs/13 을 먼저 고치세요."
    )


def test_majority_class_is_well_anchored(table):
    """대비가 성립해야 발견이 발견이다. 전부 앵커가 없으면 그냥 유사도가 낮은 것이다."""
    slot = table["비조치"]
    assert slot["anchor_rate"] >= MAJORITY_ANCHOR_FLOOR, (
        f"비조치 앵커링 {slot['anchor_rate']:.1%} — 클래스 간 대비가 사라졌습니다."
    )


def test_the_gap_is_large(table):
    gap = table["비조치"]["anchor_rate"] - table["조치"]["anchor_rate"]
    assert gap >= 0.30, f"클래스 간 앵커링 격차 {gap:.1%}"
