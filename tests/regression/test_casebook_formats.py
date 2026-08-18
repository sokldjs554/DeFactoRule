"""사례집 서식 회귀 테스트.

새 연도 사례집을 넣었을 때 파서가 조용히 망가지는 것을 막는다.
파서는 서식을 못 읽어도 예외를 내지 않고 빈 값을 남기기 때문에,
건수만 보면 정상으로 보인다. 실제로 2024년판 투입 시 결론 검출률이
97% → 38% 로 떨어졌는데 파싱은 "성공"했다.

기준선은 tests/regression/baseline.json 이며,
서식이 바뀌면 다음 명령으로 갱신하고 diff 를 리뷰한다.

    python scripts/format_inventory.py \
        --input data/raw/casebooks --output tests/regression/baseline.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

BASELINE_PATH = ROOT / "tests" / "regression" / "baseline.json"
CASEBOOK_DIR = ROOT / "data" / "raw" / "casebooks"


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip("기준선이 없습니다. format_inventory.py 를 먼저 실행하세요.")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    pdfs = [p for p in CASEBOOK_DIR.rglob("*.pdf") if "__MACOSX" not in str(p)]
    if not pdfs:
        pytest.skip(
            f"사례집 PDF 가 없습니다: {CASEBOOK_DIR} — data/SOURCES.md 참고"
        )
    from format_inventory import build

    return build(CASEBOOK_DIR)


def test_no_unknown_checkbox_format(baseline, current):
    """알려지지 않은 체크 표시 서식이 나오면 즉시 실패한다.

    이 테스트가 있었다면 2024년판의 Wingdings 'þ' 와 '상자 앞 체크' 서식을
    커밋 전에 잡았을 것이다.
    """
    known = set(baseline["known_checkbox_patterns"])
    seen: set[str] = set()
    for entry in current["sources"].values():
        seen |= set(entry.get("checkbox_patterns", {}))
    unknown = seen - known
    assert not unknown, (
        "처음 보는 체크박스 서식입니다. 파서의 RE_DECISION 을 확인하고, "
        "정당한 서식이면 기준선을 갱신하세요:\n  "
        + "\n  ".join(sorted(unknown))
    )


def test_no_unknown_check_glyph(baseline, current):
    known = set(baseline["known_check_glyphs"])
    seen = set(current["known_check_glyphs"])
    unknown = seen - known
    assert not unknown, f"처음 보는 체크 문자: {sorted(unknown)}"


@pytest.mark.parametrize("metric", ["decision_rate", "serial_rate", "sector_rate"])
def test_extraction_rates_hold(baseline, current, metric):
    """검출률이 기준선 아래로 떨어지면 실패한다."""
    floor = baseline["thresholds"][f"{metric}_min"]
    failures = []
    for name, entry in current["sources"].items():
        if metric not in entry:
            continue
        if entry[metric] < floor:
            failures.append(f"{name}: {metric}={entry[metric]:.2%} < {floor:.0%}")
    assert not failures, "검출률 미달:\n  " + "\n  ".join(failures)


def test_field_rates_hold(baseline, current):
    floor = baseline["thresholds"]["field_rate_min"]
    failures = []
    for name, entry in current["sources"].items():
        for field, rate in entry["field_rates"].items():
            if rate < floor:
                failures.append(f"{name} · {field}={rate:.2%} < {floor:.0%}")
    assert not failures, "항목 추출률 미달:\n  " + "\n  ".join(failures)


def test_known_sources_unchanged(baseline, current):
    """기존 자료의 사례 수가 달라지면 파서 변경의 부작용이다."""
    drift = []
    for name, base_entry in baseline["sources"].items():
        cur = current["sources"].get(name)
        if cur is None:
            continue  # 자료가 빠진 것은 별도 테스트에서 다룬다
        if cur["case_count"] != base_entry["case_count"]:
            drift.append(
                f"{name}: {base_entry['case_count']} → {cur['case_count']}"
            )
    assert not drift, (
        "기존 자료의 사례 수가 변했습니다. 파서 변경이 의도된 것이면 "
        "기준선을 갱신하세요:\n  " + "\n  ".join(drift)
    )


def test_new_sources_are_declared(baseline, current):
    """기준선에 없는 자료가 들어오면 알린다 — 실패가 아니라 갱신 요구다."""
    new = set(current["sources"]) - set(baseline["sources"])
    assert not new, (
        "기준선에 없는 자료입니다. 파싱 결과를 검토한 뒤 기준선을 갱신하세요:\n  "
        + "\n  ".join(sorted(new))
    )
