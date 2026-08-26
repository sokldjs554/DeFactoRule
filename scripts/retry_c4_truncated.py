#!/usr/bin/env python3
"""C-4 1차 실측에서 output truncation이 난 케이스만 별도 재측정한다.

기본은 dry-run/API 0회. `--go`를 명시해야 호출한다. 최초 5건 결과 파일은 절대
수정하지 않고, unparseable_output이었던 serial만 최대 2건 별도 checkpoint/result에
저장한다. 성공한 3건은 재호출하지 않는다.

API를 쓰기 전에 최초 성공 3건을 **현재 gate로 offline 재채점**한다. 고유사도 충돌
240006과 AG-13형 220070이 literal-grounded decisive rejection(G4)으로 올라오지 않으면
retry 2건도 호출하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.deciding_factor_prompt import MAX_TOKENS, SYSTEM, schema  # noqa: E402
from app.agents.deciding_factor_run import _evaluate, resolve  # noqa: E402
from app.core.paths import RESULTS  # noqa: E402

ORIGINAL = RESULTS / "clean" / "c4_s5_5cases.json"
OUT = RESULTS / "clean" / "c4_s5_retry_truncated.json"
CHECKPOINT = RESULTS / "clean" / "c4_s5_retry_truncated.checkpoint.json"
EXPECTED_SERIALS = {"250055", "240022"}
REQUIRED_G4 = {"240006", "220070"}
SAFE_230067 = {"decisive_difference", "incomplete_analysis", "unresolved_difference"}
MAX_CALLS = 2


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"결과 파일을 읽을 수 없습니다: {path}") from exc


def _original_records() -> list[dict]:
    if not ORIGINAL.exists():
        raise SystemExit(f"최초 C-4 결과가 없습니다: {ORIGINAL}")
    records = _load_json(ORIGINAL).get("records", [])
    if not isinstance(records, list):
        raise SystemExit(f"최초 C-4 결과 형식이 잘못됐습니다: {ORIGINAL}")
    return records


def _retry_serials(records: list[dict]) -> list[str]:
    serials = [
        str(record.get("serial"))
        for record in records
        if record.get("error") == "unparseable_output"
    ]
    if set(serials) != EXPECTED_SERIALS:
        raise SystemExit(
            "최초 truncation 대상 drift: "
            f"expected={sorted(EXPECTED_SERIALS)} actual={sorted(serials)}"
        )
    return serials


def _offline_precheck(original: list[dict], resolved: list[dict]) -> None:
    """저장된 성공 3건을 현재 gate로 재채점하고 unsafe drift면 API 전에 중단한다."""
    record_by_serial = {str(record.get("serial")): record for record in original}
    item_by_serial = {item["plan"]["serial"]: item for item in resolved}
    failures: list[str] = []

    print("  current-gate offline precheck (API 0회):")
    for serial in ("240006", "230067", "220070"):
        record = record_by_serial.get(serial)
        item = item_by_serial.get(serial)
        if not record or not item:
            failures.append(f"{serial}: 저장 결과/고정 plan 누락")
            continue
        if record.get("error"):
            failures.append(f"{serial}: 최초 성공 레코드에 error={record.get('error')}")
            continue
        gate = _evaluate(
            record.get("model_output") or {},
            item["row"]["request"],
            item["precedent_request"],
        )
        print(
            f"    {serial}: {gate.basis}/{gate.fired_rule} · "
            f"decisive={list(gate.decisive_confirmed_ids)} · "
            f"uncovered={len(gate.uncovered_differences)}"
        )
        if serial in REQUIRED_G4 and gate.basis != "decisive_difference":
            failures.append(f"{serial}: decisive_difference가 아님 ({gate.basis})")
        if serial == "230067" and gate.basis not in SAFE_230067:
            failures.append(f"230067: unsafe recovery basis={gate.basis}")

    if failures:
        raise SystemExit(
            "current gate offline precheck 실패 — API 호출 금지:\n  - "
            + "\n  - ".join(failures)
        )


def _load_checkpoint() -> list[dict]:
    if not CHECKPOINT.exists():
        return []
    payload = _load_json(CHECKPOINT)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise SystemExit(f"checkpoint 형식이 잘못됐습니다: {CHECKPOINT}")
    return records


def _save_checkpoint(records: list[dict]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps(
            {
                "experiment": "C-4 truncated-only retry",
                "records": records,
                "completed_serials": [r.get("serial") for r in records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_one(client, item: dict) -> dict:
    from app.infrastructure.anthropic_client import MODEL, call_structured

    started = time.perf_counter()
    raw = call_structured(
        client,
        SYSTEM,
        item["prompt"],
        schema(),
        MAX_TOKENS,
        effort="low",
    )
    latency = time.perf_counter() - started
    data = raw.get("data") or {}
    gate = _evaluate(data, item["row"]["request"], item["precedent_request"])
    top = item["top"]
    return {
        "serial": item["plan"]["serial"],
        "retry_of": "c4_s5_5cases.json",
        "gold": item["row"]["label"],
        "expected_basis": item["plan"]["expect"],
        "top": str(top.serial) if top else None,
        "top_label": top.label if top else None,
        "top_score": top.score if top else None,
        "opposing": len(item["opposing"]),
        "provisional": item["state"].provisional,
        "basis": gate.basis,
        "fired_rule": gate.fired_rule,
        "grounded_shared_factor_ids": gate.grounded_shared_factor_ids,
        "grounded_factor_ids": gate.grounded_factor_ids,
        "rejected_factor_ids": gate.rejected_factor_ids,
        "decisive_confirmed_ids": gate.decisive_confirmed_ids,
        "uncovered_differences": [s.text for s in gate.uncovered_differences],
        "unresolved_differences": [s.text for s in gate.unresolved_differences],
        "model_output": data,
        "model": MODEL,
        "effort": "low",
        "max_tokens": MAX_TOKENS,
        "input_tokens": raw.get("input_tokens"),
        "output_tokens": raw.get("output_tokens"),
        "error": raw.get("error"),
        "latency_s": round(latency, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", action="store_true", help="truncation 2건 중 미완료 건만 호출")
    args = parser.parse_args()

    original = _original_records()
    target_serials = _retry_serials(original)
    resolved, drift = resolve()
    if drift:
        raise SystemExit(f"C-4 plan drift가 있어 호출하지 않습니다: {drift}")
    _offline_precheck(original, resolved)

    by_serial = {item["plan"]["serial"]: item for item in resolved}
    targets = [by_serial[serial] for serial in target_serials]

    records = _load_checkpoint()
    completed = {str(record.get("serial")) for record in records}
    unknown = completed - EXPECTED_SERIALS
    if unknown:
        raise SystemExit(f"checkpoint에 retry 대상 밖 serial이 있습니다: {sorted(unknown)}")
    pending = [item for item in targets if item["plan"]["serial"] not in completed]

    print("C-4 truncation-only retry — 기본 dry-run/API 0회")
    print(f"  최초 성공 3건 재호출 금지 · 대상={target_serials}")
    print(f"  checkpoint 완료 {len(records)}건 · 이번 실행 예정 {len(pending)}건")
    print(f"  effort=low · max_tokens={MAX_TOKENS} · retry 상한={MAX_CALLS}")
    if not args.go:
        return
    if len(records) > MAX_CALLS or len(records) + len(pending) > MAX_CALLS:
        raise SystemExit("C-4 truncation retry 호출 상한 2회를 넘길 수 없습니다")

    from app.infrastructure.anthropic_client import connect, estimate_cost

    if pending:
        client = connect()
        for item in pending:
            record = _run_one(client, item)
            records.append(record)
            _save_checkpoint(records)
            print(
                f"  ✓ {record['serial']} 저장 · {record['basis']}/{record['fired_rule']} · "
                f"error={record['error'] or 'none'} · out={record['output_tokens']}"
            )
            print(
                f"    decisive={record['decisive_confirmed_ids']} · "
                f"uncovered={len(record['uncovered_differences'])}"
            )

    payload = {
        "experiment": "C-4 truncated-only retry",
        "api_calls": len(records),
        "retry": 0,
        "records": records,
        "estimated_cost_usd": estimate_cost(records),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\nretry records {len(records)} · cost ${payload['estimated_cost_usd']:.4f}\n"
        f"-> {OUT}"
    )


if __name__ == "__main__":
    main()
