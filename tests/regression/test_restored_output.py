"""복원이 적용된 산출물의 불변조건."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def test_restored_cases_are_present_and_spaced():
    path = PROCESSED / "cases_nonaction.jsonl"
    if not path.exists():
        pytest.skip("파싱 결과가 없습니다")
    rows = [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    lost = [r for r in rows if "spacing_lost" in r.get("warnings", [])]
    assert lost, "spacing_lost 사례가 사라졌습니다 — 탐지 로직을 확인하세요"

    restored = [r for r in lost if "raw_restored" in r]
    assert len(restored) == len(lost), (
        f"복원 미적용 {len(lost) - len(restored)}건 — "
        "scripts/restore_spacing.py apply 를 실행하세요"
    )
    for r in restored:
        assert r["raw_restored"].count(" ") > r["raw"].count(" "), (
            f"{r['serial']}: 공백이 늘지 않았습니다"
        )
