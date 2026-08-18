"""실패 케이스 레지스트리를 돌리고 표로 낸다.

    python scripts/failure_report.py
    python scripts/failure_report.py --layer extraction
    python scripts/failure_report.py --report experiments/results/failures.json

probe 를 전부 실행하므로 **지금 이 순간의 상태**를 보여준다. 문서에 적힌
과거형 서술과 달리, 여기서 FAIL 이 나오면 그 수정은 지금 풀려 있는 것이다.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from app.core.io import write_json
from app.evaluation.failure_taxonomy import (
    MIN_CASES,
    TAXONOMY,
    load_registry,
    validate,
)
from app.evaluation.probes import PROBES


def run_case(case: dict) -> dict:
    """한 케이스의 probe 를 돌린다. probe 가 없으면 실행하지 않는다."""
    name = case.get("probe")
    if not name:
        return {"ran": False, "passed": None, "detail": "probe 없음 (기록만)"}
    fn = PROBES.get(name)
    if fn is None:
        return {"ran": False, "passed": None, "detail": f"probe 를 찾을 수 없음: {name}"}
    try:
        passed, detail = fn()
    except Exception as exc:  # noqa: BLE001 — probe 가 죽는 것도 결과다
        return {"ran": True, "passed": False, "detail": f"예외 {type(exc).__name__}: {exc}"}
    return {"ran": True, "passed": passed, "detail": detail}


def fmt_metric(metric: dict | None) -> str:
    if not metric:
        return ""
    unit = metric.get("unit", "")

    def num(v):
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)

    return f"{num(metric['before'])} → {num(metric['after'])} {unit}".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", choices=sorted(TAXONOMY), help="한 계층만 본다")
    ap.add_argument("--report", help="결과를 JSON 으로 저장")
    args = ap.parse_args()

    cases = load_registry()
    problems = {c.get("id", "?"): validate(c) for c in cases}
    broken = {k: v for k, v in problems.items() if v}
    if broken:
        print("레지스트리 형식 오류")
        for cid, issues in broken.items():
            print(f"  {cid}: {'; '.join(issues)}")
        raise SystemExit(1)

    shown = [c for c in cases if not args.layer or c["layer"] == args.layer]
    results = {c["id"]: run_case(c) for c in shown}

    # ── taxonomy 분포 ────────────────────────────────────────────
    print(f"실패 케이스 {len(cases)}건 (최소 요구 {MIN_CASES}건)\n")
    by_layer: dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        by_layer[c["layer"]][c["category"]] += 1
    print(f"{'계층':>16}  {'건수':>4}  범주별")
    for layer in TAXONOMY:
        counts = by_layer.get(layer)
        if not counts:
            continue
        parts = " · ".join(f"{k} {v}" for k, v in counts.most_common())
        print(f"{layer:>16}  {sum(counts.values()):>4}  {parts}")

    # ── 개선 전/후 ───────────────────────────────────────────────
    with_metric = [c for c in shown if c.get("metric")]
    print(f"\n{'─' * 78}\n개선 전 → 후 ({len(with_metric)}건에 수치가 있다)\n")
    print(f"{'ID':>6}  {'지표':>22}  {'전 → 후':>26}  출처")
    for c in with_metric:
        m = c["metric"]
        src = m.get("source", "probe 실행값" if m["kind"] == "live" else "")
        print(f"{c['id']:>6}  {m['name']:>22}  {fmt_metric(m):>26}  {src}")

    # ── probe 실행 결과 ──────────────────────────────────────────
    print(f"\n{'─' * 78}\nprobe 실행 — 수정이 지금도 유지되는가\n")
    for c in shown:
        r = results[c["id"]]
        if not r["ran"]:
            mark = " -- "
        elif r["passed"]:
            mark = " OK "
        else:
            mark = "FAIL"
        flag = "  ← 열린 케이스" if c["status"] == "open" else ""
        print(f"  [{mark}] {c['id']} {c['title']}{flag}")
        print(f"         {r['detail']}")

    ran = [c for c in shown if results[c["id"]]["ran"]]
    failed = [c for c in ran if not results[c["id"]]["passed"]]
    unexpected = [c for c in failed if c["status"] != "open"]
    stale_open = [
        c for c in shown
        if c["status"] == "open" and results[c["id"]]["ran"] and results[c["id"]]["passed"]
    ]

    print(f"\n{'─' * 78}")
    print(
        f"probe 실행 {len(ran)}/{len(shown)} · "
        f"통과 {len(ran) - len(failed)} · 실패 {len(failed)}"
    )
    if unexpected:
        print("\n  ⚠ 고쳤다고 기록된 케이스가 실패했습니다 — 수정이 풀렸습니다:")
        for c in unexpected:
            print(f"      {c['id']} {c['title']}")
    if stale_open:
        print("\n  ⚠ 열려 있다고 기록된 케이스가 통과했습니다 — 레지스트리를 갱신하세요:")
        for c in stale_open:
            print(f"      {c['id']} {c['title']}")
    if not unexpected and not stale_open:
        print("  레지스트리와 실제 상태가 일치합니다.")

    if args.report:
        write_json(
            args.report,
            {
                "total": len(cases),
                "min_required": MIN_CASES,
                "by_layer": {k: dict(v) for k, v in by_layer.items()},
                "cases": [{**c, "result": results[c["id"]]} for c in shown],
            },
        )
        print(f"\n-> {args.report}")


if __name__ == "__main__":
    main()
