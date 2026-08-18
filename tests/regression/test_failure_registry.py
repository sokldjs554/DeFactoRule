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
    """probe 를 만들어 놓고 어디에도 안 걸면 보고서에서 사라진다.

    등록 경로는 둘이다 — 개별 사례의 `probe` 필드, 또는 재발 패턴의 가드로서
    `PATTERN_GUARDS`. 패턴 가드는 특정 사례가 아니라 패턴 전체를 지키므로
    한 사례에 매달지 않는다.
    """
    from app.evaluation.failure_taxonomy import PATTERN_GUARDS

    used = {c["probe"] for c in WITH_PROBE} | set(PATTERN_GUARDS.values())
    orphans = sorted(set(PROBES) - used)
    assert not orphans, f"어디에도 등록되지 않은 probe: {orphans}"


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


# ── 재발 패턴 ────────────────────────────────────────────────────
def test_every_recurring_pattern_has_a_guard():
    """2건 이상 붙은 패턴에는 패턴 단위 가드가 있어야 한다.

    레지스트리의 probe 는 **과거의 그 자리**를 지킨다. 같은 실수가 다른
    자리에서 다시 나오는 것은 막지 못한다. 실제로 "걸러낸 것을 기록하지
    않는다" 가 네 곳에서 나왔고, 넷 다 따로 고쳤는데도 다섯 번째를 막을
    장치가 없었다.

    패턴이 두 번째로 나타나는 순간 이 테스트가 가드를 요구한다.
    """
    from app.evaluation.failure_taxonomy import PATTERN_GUARDS, patterns_needing_guards

    needing = patterns_needing_guards(REGISTRY)
    missing = {
        name: ids for name, ids in needing.items() if name not in PATTERN_GUARDS
    }
    assert not missing, (
        "패턴 가드가 없는 반복 패턴이 있습니다:\n"
        + "\n".join(f"  {name}: {ids}" for name, ids in missing.items())
        + "\n  app/evaluation/patterns.py 에 가드를 만들고 PATTERN_GUARDS 에 등록하세요."
    )


def test_pattern_guards_exist_and_pass():
    from app.evaluation.failure_taxonomy import PATTERN_GUARDS

    for pattern, probe_name in PATTERN_GUARDS.items():
        assert probe_name in PROBES, f"{pattern} 의 가드 '{probe_name}' 를 찾을 수 없습니다"
        passed, detail = PROBES[probe_name]()
        assert passed, f"{pattern} 가드 실패 — {detail}"


def test_declared_patterns_are_all_in_use():
    """쓰지 않는 패턴 이름은 지운다. 목록만 길어지면 아무도 안 본다."""
    from app.evaluation.failure_taxonomy import PATTERNS

    used = {c["pattern"] for c in REGISTRY if c.get("pattern")}
    unused = set(PATTERNS) - used
    assert not unused, f"어느 사례에도 붙지 않은 패턴: {sorted(unused)}"


def test_every_filter_stage_is_registered():
    """걸러내기 단계를 만들고 등록을 잊으면 패턴 가드가 그것을 모른다.

    이 한계는 없앨 수 없지만, 최소한 등록된 단계가 몇 개인지는 눈에 띄게 둔다.
    새 단계를 만들면 여기 숫자가 걸린다.
    """
    from app.evaluation.patterns import FILTER_STAGES

    assert len(FILTER_STAGES) >= 6, (
        f"등록된 걸러내기 단계 {len(FILTER_STAGES)}개 — 줄었다면 왜 지웠는지 확인하세요"
    )
