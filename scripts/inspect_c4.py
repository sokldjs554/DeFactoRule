#!/usr/bin/env python3
"""이미 저장된 C-4 결과를 API 호출 없이 진단한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.paths import RESULTS  # noqa: E402

RESULT = RESULTS / "clean" / "c4_s5_5cases.json"


def _factor_line(item: dict) -> str:
    return (
        f"{item.get('id')} [{item.get('side')}] decisive={item.get('decisive')} "
        f"axis={item.get('axis')!r} text={item.get('text')!r}"
    )


def main() -> None:
    if not RESULT.exists():
        raise SystemExit(f"결과 파일이 없습니다: {RESULT}")
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    print(f"C-4 offline inspect · records={len(records)} · API 호출 0회")
    for record in records:
        print("\n" + "=" * 72)
        print(
            f"{record.get('serial')} · basis={record.get('basis')} · "
            f"rule={record.get('fired_rule')} · error={record.get('error')}"
        )
        print(
            f"tokens in/out={record.get('input_tokens')}/{record.get('output_tokens')} · "
            f"expected={record.get('expected_basis')}"
        )
        print(f"grounded_shared={record.get('grounded_shared_factor_ids')}")
        print(f"grounded_diff={record.get('grounded_factor_ids')}")
        print(f"rejected={record.get('rejected_factor_ids')}")
        print(f"decisive_confirmed={record.get('decisive_confirmed_ids')}")
        print("uncovered:")
        for text in record.get("uncovered_differences", []):
            print(f"  - {text}")
        print("unresolved:")
        for text in record.get("unresolved_differences", []):
            print(f"  - {text}")

        data = record.get("model_output") or {}
        for key in ("shared_factors", "only_in_request", "only_in_precedent"):
            print(f"{key}:")
            for item in data.get(key, []):
                print(f"  - {_factor_line(item)}")
        if record.get("error"):
            print("NOTE: 기존 결과 레코드는 raw truncated text를 보존하지 않았습니다.")


if __name__ == "__main__":
    main()
