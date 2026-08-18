"""실패 케이스 레지스트리 — 고친 것이 계속 고쳐진 채로 있는가.

명세 §11 은 실패 케이스를 taxonomy 로 분류하고 개선 전/후를 숫자로 비교할
것을 요구한다. 산문으로 적힌 목록은 그 요구를 만족하지 못한다 — 다음에 같은
자리가 깨졌을 때 아무도 모르기 때문이다.

그래서 케이스마다 실행 가능한 probe 를 붙이고, 여기서 전부 돌린다.
`status` 와 실제 실행 결과가 어긋나면 실패한다. 어느 방향이든 어긋나는 것이
문제다 — 고쳤다는 케이스가 깨졌으면 회귀이고, 열려 있다는 케이스가 통과하면
레지스트리가 낡은 것이다.
"""

from __future__ import annotations

import pytest

from app.evaluation.failure_taxonomy import MIN_CASES, TAXONOMY, load_registry, validate
from app.evaluation.probes import PROBES

REGISTRY = load_registry()
WITH_PROBE = [c for c in REGISTRY if c.get("probe")]


def test_registry_meets_the_minimum():
    assert len(REGISTRY) >= MIN_CASES, (
        f"실패 케이스 {len(REGISTRY)}건 — 명세는 최소 {MIN_CASES}건을 요구한다"
    )


def test_ids_are_unique():
    ids = [c["id"] for c in REGISTRY]
    assert len(set(ids)) == len(ids), "중복 ID"


@pytest.mark.parametrize("case", REGISTRY, ids=lambda c: c["id"])
def test_case_schema_is_valid(case):
    problems = validate(case)
    assert not problems, f"{case['id']}: {'; '.join(problems)}"


def test_every_layer_is_represented():
    """한 계층에만 실패가 몰려 있다면 다른 계층을 안 들여다본 것이다."""
    covered = {c["layer"] for c in REGISTRY}
    missing = set(TAXONOMY) - covered
    assert not missing, f"실패 케이스가 하나도 없는 계층: {sorted(missing)}"


def test_most_cases_are_executable():
    """기록만 남긴 케이스가 많으면 레지스트리가 산문으로 되돌아간 것이다."""
    ratio = len(WITH_PROBE) / len(REGISTRY)
    assert ratio >= 0.75, f"probe 가 붙은 비율 {ratio:.0%} — 실행 가능한 케이스가 너무 적다"


def test_enough_cases_carry_numbers():
    """§11 은 개선 전/후를 숫자로 비교할 것을 요구한다."""
    with_metric = [c for c in REGISTRY if c.get("metric")]
    assert len(with_metric) >= 15, f"수치가 있는 케이스 {len(with_metric)}건"


def test_probe_names_resolve():
    unknown = [c["id"] for c in WITH_PROBE if c["probe"] not in PROBES]
    assert not unknown, f"probe 를 찾을 수 없는 케이스: {unknown}"


def test_every_probe_is_registered():
    """probe 를 만들어 놓고 레지스트리에 안 넣으면 보고서에서 사라진다."""
    used = {c["probe"] for c in WITH_PROBE}
    orphans = sorted(set(PROBES) - used)
    assert not orphans, f"레지스트리에 없는 probe: {orphans}"


@pytest.mark.parametrize("case", WITH_PROBE, ids=lambda c: c["id"])
def test_probe_matches_recorded_status(case):
    passed, detail = PROBES[case["probe"]]()
    if case["status"] == "fixed":
        assert passed, f"{case['id']} 수정이 풀렸습니다 — {detail}"
    else:
        assert not passed, (
            f"{case['id']} 는 열린 케이스로 기록돼 있는데 통과했습니다. "
            f"레지스트리를 갱신하세요 — {detail}"
        )
