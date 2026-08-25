"""C-4 S5 Deciding-Factor 실측 — temporal clean 5건만, 각 1회.

기본 실행은 dry-run이다. `--go`를 명시해야만 Anthropic API를 호출한다.
재시도/대체모델은 없고 정확히 이 파일의 PLAN 5건만 호출할 수 있다.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from app.agents.applicability import opposing_evidence
from app.agents.deciding_factor import Factor, evaluate_diff_coverage
from app.agents.deciding_factor_prompt import MAX_TOKENS, SYSTEM, build_prompt, schema
from app.agents.workflow import VARIANTS, Workflow
from app.core.io import load_jsonl
from app.core.paths import EVAL, PROCESSED, RESULTS
from app.domain.similarity import DOUBT
from app.retrieval.lexical import LexicalRetriever

# C-4 temporal clean 사정거리 19건에서 역할별로 고정한 5건.
PLAN = [
    {"serial": "250055", "kind": "적용 후보", "top": "250050", "route": "R5",
     "opposing": 0, "expect": "no_decisive_difference"},
    {"serial": "240006", "kind": "고유사도 충돌", "top": "230095", "route": "R9",
     "opposing": 3, "expect": "decisive_difference"},
    {"serial": "230067", "kind": "모호·부분 적용", "top": "220046", "route": "R5",
     "opposing": 0, "expect": "incomplete_or_difference"},
    {"serial": "240022", "kind": "잘못된 top + 반대근거", "top": "230028", "route": "R1",
     "opposing": 1, "expect": "decisive_difference"},
    {"serial": "220070", "kind": "잘못된 top + 반대근거 0", "top": "220036", "route": "R1",
     "opposing": 0, "expect": "decisive_difference"},
]
MAX_CALLS = 5
OUT = RESULTS / "clean" / "c4_s5_5cases.json"


class CallBudget:
    def __init__(self) -> None:
        self.used = 0

    def spend(self) -> None:
        if self.used >= MAX_CALLS:
            raise RuntimeError(f"C-4 호출 상한 {MAX_CALLS}회를 넘길 수 없습니다")
        self.used += 1


def _world():
    dev = [r for r in load_jsonl(EVAL / "nonaction_dev_clean.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test_clean.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [
        c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
        for c in cases
    ]
    corpus = [text for text in corpus if text]
    rules = json.loads((RESULTS / "e6_rules_clean.json").read_text(encoding="utf-8"))["rules"]
    risk = json.loads(
        (RESULTS / "trap_risk_clean_temporal.json").read_text(encoding="utf-8")
    )
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]
    flow = Workflow(
        LexicalRetriever().fit(dev, corpus),
        dev,
        rules,
        risk,
        policy=VARIANTS["router-temporal"],
        fallback=fallback,
    )
    by_key = {(r["source"], str(r["serial"])): r for r in dev}
    return flow, test, by_key


def resolve() -> tuple[list[dict], list[str]]:
    flow, test, by_key = _world()
    test_by_serial = {str(row["serial"]): row for row in test}
    resolved: list[dict] = []
    drift: list[str] = []
    for plan in PLAN:
        row = test_by_serial.get(plan["serial"])
        if row is None:
            drift.append(f"{plan['serial']}: clean test에 없음")
            continue
        state = flow.run(row)
        top = next((e for e in state.retrieved_evidence if e.rank == 0), None)
        opposing = opposing_evidence(state)
        if not state.abstained or state.precedent_score < DOUBT:
            drift.append(f"{plan['serial']}: S5 사정거리 밖")
        if state.route_reason != plan["route"]:
            drift.append(f"{plan['serial']}: route {plan['route']} -> {state.route_reason}")
        if top is None or str(top.serial) != plan["top"]:
            drift.append(
                f"{plan['serial']}: top {plan['top']} -> {top.serial if top else '없음'}"
            )
        if len(opposing) != plan["opposing"]:
            drift.append(
                f"{plan['serial']}: opposing {plan['opposing']} -> {len(opposing)}"
            )
        precedent = by_key.get((top.source, str(top.serial))) if top else None
        if not precedent:
            drift.append(f"{plan['serial']}: top 선례 원문을 dev에서 못 찾음")
            precedent = {}
        resolved.append(
            {
                "plan": plan,
                "row": row,
                "state": state,
                "top": top,
                "opposing": opposing,
                "precedent_request": precedent.get("request", ""),
                "prompt": build_prompt(row["request"], precedent.get("request", "")),
            }
        )
    return resolved, drift


def preview(resolved: list[dict], drift: list[str]) -> None:
    print("C-4 S5 temporal clean 5건 — 기본은 dry-run, API 호출 0회")
    for item in resolved:
        p, row, state, top = item["plan"], item["row"], item["state"], item["top"]
        print(f"  {p['serial']} {p['kind']} · gold {row['label']} · {state.route_reason}")
        if top:
            print(
                f"    top {top.serial}/{top.label} {top.score:.4f} · "
                f"opposing {len(item['opposing'])} · provisional {state.provisional}"
            )
        print(f"    예상 basis {p['expect']} · prompt {len(item['prompt'])}자")
    if drift:
        print("\nDRIFT — 호출 금지")
        for line in drift:
            print(f"  ✗ {line}")


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


def _evaluate(data: dict, request: str, precedent: str):
    shared = [_factor(x) for x in data.get("shared_factors", [])]
    differences = [
        _factor(x)
        for key in ("only_in_request", "only_in_precedent")
        for x in data.get(key, [])
    ]
    # 첫 실측에서는 A1~A4 admissibility를 별도 구현하지 않았으므로 G5는 fail-closed.
    return evaluate_diff_coverage(request, precedent, shared, differences)


def run_one(client, item: dict, budget: CallBudget) -> dict:
    from app.infrastructure.anthropic_client import MODEL, call_structured

    budget.spend()
    started = time.perf_counter()
    raw = call_structured(client, SYSTEM, item["prompt"], schema(), MAX_TOKENS)
    latency = time.perf_counter() - started
    data = raw.get("data") or {}
    gate = _evaluate(data, item["row"]["request"], item["precedent_request"])
    top = item["top"]
    return {
        "serial": item["plan"]["serial"],
        "kind": item["plan"]["kind"],
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
        "input_tokens": raw.get("input_tokens"),
        "output_tokens": raw.get("output_tokens"),
        "error": raw.get("error"),
        "latency_s": round(latency, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true", help="고정 5건을 실제로 각 1회 호출")
    args = ap.parse_args()

    resolved, drift = resolve()
    preview(resolved, drift)
    if drift:
        raise SystemExit("선정 상태 drift가 있어 API를 호출하지 않습니다")
    if not args.go:
        return

    from app.infrastructure.anthropic_client import connect, estimate_cost

    client = connect()
    budget = CallBudget()
    records = [run_one(client, item, budget) for item in resolved]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "C-4 S5 deciding-factor gate qualitative audit",
        "api_calls": budget.used,
        "retry": 0,
        "records": records,
        "estimated_cost_usd": estimate_cost(records),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAPI calls {budget.used} · retry 0 · cost ${payload['estimated_cost_usd']:.4f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()