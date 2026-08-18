"""커밋된 파싱 결과의 불변조건을 검증한다.

원본 PDF 는 저장소에 없으므로 서식 회귀 테스트는 CI 에서 건너뛴다.
대신 파싱 결과(data/processed/*.jsonl)는 커밋되어 있으므로, 그 산출물이
만족해야 할 조건을 여기서 확인한다. 파서를 고쳤는데 산출물을 갱신하지
않았거나, 갱신하면서 무언가 망가뜨린 경우를 잡는다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

DECISIONS = {"비조치", "조치", "기타"}
SPLIT_MODES = {"single", "paired"}


def load(name: str) -> list[dict]:
    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{path} 가 없습니다. 파서를 먼저 실행하세요.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def nonaction() -> list[dict]:
    return load("cases_nonaction.jsonl")


@pytest.fixture(scope="module")
def interpretation() -> list[dict]:
    return load("cases_interpretation.jsonl")


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    return load("qa_pairs.jsonl")


def test_corpus_is_not_shrinking(nonaction, interpretation):
    """확보한 코퍼스가 줄어들면 파서 회귀다. 늘어나는 것은 정상이다."""
    assert len(nonaction) >= 133, f"비조치의견서 {len(nonaction)}건 (기준 133)"
    assert len(interpretation) >= 300, f"법령해석 {len(interpretation)}건 (기준 300)"


def test_decision_labels_are_valid(nonaction):
    bad = {c["decision"] for c in nonaction if c["decision"] not in DECISIONS | {None}}
    assert not bad, f"알 수 없는 결론 라벨: {bad}"


def test_decision_coverage(nonaction):
    covered = sum(1 for c in nonaction if c["decision"])
    rate = covered / len(nonaction)
    assert rate >= 0.95, f"결론 검출률 {rate:.1%} — 체크 표시 서식을 확인하세요"


def test_minority_class_is_tracked(nonaction):
    """비조치 편중은 알려진 문제다. 사라지거나 뒤집히면 무언가 잘못된 것이다."""
    dist = Counter(c["decision"] for c in nonaction if c["decision"])
    assert dist["비조치"] > dist["조치"], "비조치가 다수 클래스여야 합니다"
    assert dist["조치"] + dist["기타"] >= 10, (
        f"소수 클래스가 {dist['조치'] + dist['기타']}건입니다. "
        "학습 가능성이 무너진 상태이므로 확인이 필요합니다."
    )


def test_sector_is_assigned(nonaction, interpretation):
    for label, rows in (("비조치", nonaction), ("법령해석", interpretation)):
        missing = sum(1 for r in rows if not r.get("sector"))
        rate = 1 - missing / len(rows)
        assert rate >= 0.95, f"{label} 업권 부여율 {rate:.1%} — 구분 페이지 서식 확인"


def test_required_keys_present(nonaction, interpretation):
    required = {"source", "doc_type", "serial", "sector", "page", "fields", "raw"}
    for rows in (nonaction, interpretation):
        for r in rows[:50]:
            assert required <= set(r), f"누락 키: {required - set(r)}"


def test_pairs_cover_every_case(nonaction, interpretation, pairs):
    """모든 사례가 최소 한 쌍으로 나타나야 한다 — 분할 중 유실 방지."""
    cases = {(c["source"], c["page"], c["serial"]) for c in nonaction + interpretation}
    covered = {(p["source"], p["page"], p["serial"]) for p in pairs}
    assert cases <= covered, f"쌍으로 나오지 않은 사례 {len(cases - covered)}건"
    assert len(pairs) >= len(cases), "쌍 수가 사례 수보다 적습니다"


def test_split_modes_are_conservative(pairs):
    """회답 단독 분할은 오분할이 확인되어 제거했다. 되살아나면 실패한다."""
    modes = set(p["split_mode"] for p in pairs)
    assert modes <= SPLIT_MODES, (
        f"허용되지 않은 분할 모드: {modes - SPLIT_MODES}. "
        "회답에만 있는 순번은 요건 열거이지 질의 구분이 아닙니다 "
        "(docs/02-w1-gate.md 참고)."
    )


def test_paired_splits_are_internally_consistent(pairs):
    grouped: dict[tuple, list[dict]] = {}
    for p in pairs:
        if p["split_mode"] == "paired":
            grouped.setdefault((p["source"], p["page"], p["serial"]), []).append(p)
    for key, group in grouped.items():
        count = group[0]["pair_count"]
        assert len(group) == count, f"{key}: pair_count={count} 인데 {len(group)}쌍"
        assert {p["pair_index"] for p in group} == set(range(1, count + 1)), (
            f"{key}: 순번이 연속하지 않습니다"
        )
        for p in group:
            assert p["question"].strip(), f"{key}: 빈 질의"
            assert p["answer"].strip(), f"{key}: 빈 회답"
