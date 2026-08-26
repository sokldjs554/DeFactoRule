#!/usr/bin/env python3
"""이미 저장된 C-4 결과를 API 호출 없이 현재 gate로 재채점한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.deciding_factor import Factor, evaluate_diff_coverage  # noqa: E402
from app.agents.deciding_factor_run import resolve  # noqa: E402
from app.core.paths import RESULTS  # noqa: E402

RESULT = RESULTS / "clean" / "c4_s5_5cases.json"


def _factor_line(item: dict) -> str:
    return (
        f"{item.get('id')} [{item.get('side')}] decisive={item.get('decisive')} "
        f"axis={item.get('axis')!r} text={item.get('text')!r}"
    )


def _factor(item: dict) -> Factor:
    return Factor(
        id=str(item.get("id", "")),
        text=str(item.get("text", "")),
        side=item.get("side", "both"),
        axis=str(item.get("axis", "")),
        value_in_request=item.get("value_in_request"),
        value_in_precedent=item.get("value_in_precedent"),
        decisive=bool(item.get("decisive", False)),
        why_not_decisive=item.get("why_not_decisive"),
    )


def _regrade(record: dict, case: dict):
    data = record.get("model_output") or {}
    shared = [_factor(item) for item in data.get("shared_factors", [])]
    differences = [
        _factor(item)
        for key in ("only_in_request", "only_in_precedent")
        for item in data.get(key, [])
    ]
    return evaluate_diff_coverage(
        case["row"]["request"],
        case["precedent_request"],
        shared,
        differences,
    )


def main() -> None:
    if not RESULT.exists():
        raise SystemExit(f"결과 파일이 없습니다: {RESULT}")
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    resolved, drift = resolve()
    if drift:
        raise SystemExit(f"현재 temporal plan drift: {drift}")
    by_serial = {item["plan"]["serial"]: item for item in resolved}

    print(f"C-4 offline regrade · records={len(records)} · API 호출 0회")
    for record in records:
        serial = str(record.get("serial"))
        case = by_serial.get(serial)
        if case is None:
            raise SystemExit(f"저장 결과에 PLAN 밖 serial: {serial}")
        current = _regrade(record, case)

        print("\n" + "=" * 72)
        print(
            f"{serial} · saved={record.get('basis')}/{record.get('fired_rule')} · "
            f"CURRENT={current.basis}/{current.fired_rule} · error={record.get('error')}"
        )
        print(
            f"tokens in/out={record.get('input_tokens')}/{record.get('output_tokens')} · "
            f"expected={record.get('expected_basis')}"
        )
        print(f"CURRENT grounded_shared={list(current.grounded_shared_factor_ids)}")
        print(f"CURRENT grounded_diff={list(current.grounded_factor_ids)}")
        print(f"CURRENT rejected={list(current.rejected_factor_ids)}")
        print(f"CURRENT decisive_confirmed={list(current.decisive_confirmed_ids)}")
        print("CURRENT uncovered:")
        for segment in current.uncovered_differences:
            print(f"  - {segment.text}")
        print("CURRENT unresolved:")
        for segment in current.unresolved_differences:
            print(f"  - {segment.text}")

        data = record.get("model_output") or {}
        for key in ("shared_factors", "only_in_request", "only_in_precedent"):
            print(f"{key}:")
            for item in data.get(key, []):
                print(f"  - {_factor_line(item)}")
        if record.get("error"):
            print("NOTE: 최초 결과에는 truncated raw 전체가 보존되지 않았습니다.")


if __name__ == "__main__":
    main()
