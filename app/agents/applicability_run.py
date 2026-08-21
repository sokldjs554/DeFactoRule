"""E11b 실측 — **사전 등록한 5건만, 각 1회씩.** 재시도하지 않는다.

## 이 파일이 지키는 것

`docs/18-e11b-case-selection.md` 에 호출 **전에** 못 박은 5건이 있다. 이
실행기는 그 5건 말고는 아무것도 부르지 못한다. 목록이 상수로 박혀 있고,
호출 수에 상한이 걸려 있으며, 재시도 경로가 아예 없다.

    각 건 1회      재시도 없음 · 대체 모델 없음
    상한 5회       여섯 번째 호출을 시도하면 예외로 죽는다
    이상해도 끝     결과가 예상 밖이어도 다시 부르지 않는다

## 왜 사전 점검(preflight)을 하지 않는가

`preflight` 는 본 스키마로 한 번 찔러 보는 장치이고, 332 요청이 400 으로
죽었던 IN-10 이후에 만들었다. 그런데 이번 지시는 **정확히 5회**다. 여기서
쓰는 스키마는 배열이 없는 평평한 객체 셋이라 IN-10 계열의 위험이 없고,
설령 계약이 거절되더라도 400 은 과금되지 않으므로 5건이 전부 $0 으로
실패하고 끝난다. 손해가 없으므로 지시를 그대로 따른다.

## 왜 호출 전에 상태를 대조하는가

선정안은 "이 사건의 1등 선례는 저것" 이라는 전제 위에 있다. 코퍼스나 검색기가
조금이라도 달라지면 **다른 것을 재고서 같은 이름으로 보고하게 된다.** 그래서
부르기 전에 경로·1등 선례 일련번호·반대 근거 수를 대조하고, 하나라도 어긋나면
**돈을 쓰기 전에** 멈춘다.
"""

from __future__ import annotations

import json
import pathlib
import time

from app.agents.applicability import (
    MAX_TOKENS,
    SYSTEM,
    apply_verdict,
    build_prompt,
    opposing_evidence,
    quotes_are_grounded,
    schema,
)

# 사전 등록한 5건. docs/18-e11b-case-selection.md 와 같아야 한다.
#   (일련번호, 범주, 기대 판정, 대조용 경로, 대조용 1등 선례, 대조용 반대 근거 수)
PLAN: list[dict] = [
    {"serial": "250055", "kind": "① 명백히 적용", "expect": "applies",
     "route": "R5", "top": "250050", "n_opposing": 0},
    {"serial": "240047", "kind": "② 표면 유사·비적용", "expect": "differs",
     "route": "R1", "top": "240046", "n_opposing": 0},
    {"serial": "230110", "kind": "③ 모호·부분 적용", "expect": "unclear",
     "route": "R5", "top": "220041", "n_opposing": 0},
    {"serial": "240023", "kind": "④ 충돌 선례 · H1 반증", "expect": "applies",
     "route": "R6", "top": "240057", "n_opposing": 2},
    {"serial": "220049", "kind": "⑤ 최고 난이도", "expect": "differs",
     "route": "R6", "top": "220041", "n_opposing": 1},
]

MAX_CALLS = 5

# ⑤ 220049 의 결론을 가른 구절. 판정이 맞아도 근거가 여기 없으면 운이다.
DECIDING_PHRASE = "내부 업무용 시스템과는 연결되지 않음"


class CallBudget:
    """여섯 번째 호출을 **막는다.** 세는 것으로는 부족하다."""

    def __init__(self, limit: int = MAX_CALLS) -> None:
        self.limit, self.used = limit, 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError(
                f"호출 상한 {self.limit} 회를 넘기려 했습니다. 재시도도 추가 건도 "
                f"이 실행기에는 없습니다."
            )
        self.used += 1


def build_world():
    """실측 상태를 만든다. E11b 이전 단계와 **같은 배선**이어야 한다."""
    from app.agents.workflow import VARIANTS, Workflow
    from app.core.io import load_jsonl
    from app.core.paths import EVAL, PROCESSED, RESULTS
    from app.retrieval.lexical import LexicalRetriever

    dev = [r for r in load_jsonl(EVAL / "nonaction_dev.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
              for c in cases]
    rules = json.loads((RESULTS / "e6_rules.json").read_text(encoding="utf-8"))["rules"]
    risk = json.loads((RESULTS / "trap_risk.json").read_text(encoding="utf-8"))

    flow = Workflow(LexicalRetriever().fit(dev, [t for t in corpus if t]), dev,
                    rules, risk, policy=VARIANTS["router"])
    by_key = {(r["source"], str(r["serial"])): r for r in dev}
    return flow, dev, test, by_key


def resolve(flow, test, by_key):
    """사전 등록한 5건을 실제 상태에 붙이고, **어긋나면 이름을 댄다.**"""
    index = {str(row["serial"]): i for i, row in enumerate(test)}
    resolved: list[dict] = []
    drift: list[str] = []

    for item in PLAN:
        serial = item["serial"]
        if serial not in index:
            drift.append(f"{serial}: test 에 없습니다")
            continue
        row = test[index[serial]]
        state = flow.run(row)
        top = next((e for e in state.retrieved_evidence if e.rank == 0), None)
        opposing = opposing_evidence(state)

        if not state.abstained:
            drift.append(f"{serial}: 더 이상 기권이 아닙니다 ({state.route_reason})")
        if state.route_reason != item["route"]:
            drift.append(f"{serial}: 경로 {item['route']} -> {state.route_reason}")
        if top is None or str(top.serial) != item["top"]:
            drift.append(f"{serial}: 1등 선례 {item['top']} -> "
                         f"{top.serial if top else '없음'}")
        if len(opposing) != item["n_opposing"]:
            drift.append(f"{serial}: 반대 근거 {item['n_opposing']} -> {len(opposing)}")

        precedent = (by_key.get((top.source, str(top.serial))) if top else None) or {}
        resolved.append({
            "plan": item, "row": row, "state": state, "top": top,
            "opposing": opposing, "precedent_request": precedent.get("request", ""),
            "prompt": build_prompt(row["request"], precedent.get("request", "")),
        })

    return resolved, drift


def preview(resolved, drift) -> None:
    """부르기 전에 무엇을 부를지 보여 준다. **여기서는 돈이 들지 않는다.**"""
    from app.agents.applicability import estimate_cost

    print(f"E11b 실측 계획 — {len(resolved)}건, 각 1회, 재시도 없음\n")
    for item in resolved:
        plan, state, top = item["plan"], item["state"], item["top"]
        print(f"  {plan['serial']}  {plan['kind']}")
        print(f"    정답 {item['row']['label']} · 경로 {state.route_reason}"
              f"/{state.abstention_reason.value} · 기대 {plan['expect']}")
        print(f"    1등 선례 {top.serial} · {top.label} · {top.score:.3f}"
              f" · 반대 근거 {len(item['opposing'])}건")
        print(f"    프롬프트 {len(item['prompt'])}자")
    total = sum(len(i["prompt"]) for i in resolved) / max(len(resolved), 1)
    print(f"\n추정 비용 약 ${estimate_cost(len(resolved), int(total)):.3f}"
          f" · 모델 정보는 app/infrastructure/anthropic_client.py 의 MODEL")
    if drift:
        print("\n선정안과 어긋난 것:")
        for line in drift:
            print(f"  ✗ {line}")


def run_one(client, item, budget: CallBudget) -> dict:
    """한 건. **한 번만 부른다.** 실패해도 다시 부르지 않는다."""
    from app.infrastructure.anthropic_client import MODEL, call_structured

    plan, row, state = item["plan"], item["row"], item["state"]
    before = {
        "route": state.route.value if state.route else None,
        "route_reason": state.route_reason,
        "abstained": state.abstained,
        "abstention_reason": (state.abstention_reason.value
                              if state.abstention_reason else None),
        "decision": state.decision,
        "provisional": state.provisional,
    }

    budget.spend()
    started = time.perf_counter()
    result = call_structured(client, SYSTEM, item["prompt"], schema(), MAX_TOKENS)
    latency = time.perf_counter() - started

    data = result.get("data") or {}
    verdict = data.get("verdict")
    grounding = quotes_are_grounded(data, row["request"], item["precedent_request"])

    recovered = None
    if verdict is not None:
        recovered, _ = apply_verdict(state, verdict)

    return {
        "case_id": plan["serial"],
        "kind": plan["kind"],
        "expected_verdict": plan["expect"],
        "truth_label": row["label"],
        # 원본 그대로
        "model_output_raw": result.get("raw"),
        "parsed_result": data or None,
        "api_error": result.get("error"),
        "api_error_detail": result.get("error_detail"),
        "verdict": verdict,
        "quote_a": data.get("quote_a"),
        "quote_b": data.get("quote_b"),
        "quotes_ungrounded": grounding,
        "quote_a_has_deciding_phrase": (
            DECIDING_PHRASE.replace(" ", "") in (data.get("quote_a") or "").replace(" ", "")
        ),
        # 근거
        "top_precedent": {
            "serial": str(item["top"].serial), "label": item["top"].label,
            "score": round(item["top"].score, 4), "id": item["top"].id,
        } if item["top"] else None,
        "opposing_evidence": item["opposing"],
        # 반영 결과
        "recovered": recovered,
        "state_before": before,
        "final_decision": state.decision,
        "provisional": state.provisional,
        "confidence": state.confidence,
        "abstained": state.abstained,
        "abstention_reason": (state.abstention_reason.value
                              if state.abstention_reason else None),
        "route": state.route.value if state.route else None,
        "route_reason": state.route_reason,
        "evidence_used": list(state.evidence_used),
        "execution_trace": [
            {"node": s.node, "summary": s.summary, "detail": s.detail}
            for s in state.execution_trace
        ],
        # 계측
        "latency_seconds": round(latency, 3),
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "effort": "medium",
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "prompt_chars": len(item["prompt"]),
    }


def main() -> None:
    import argparse

    from app.core.paths import RESULTS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true",
                    help="실제로 호출한다. 없으면 계획만 보여 준다.")
    ap.add_argument("--output", default=str(RESULTS / "e11b_5cases.json"))
    args = ap.parse_args()

    flow, _dev, test, by_key = build_world()
    resolved, drift = resolve(flow, test, by_key)
    preview(resolved, drift)

    if drift:
        raise SystemExit(
            "\n선정안과 어긋납니다 — **호출하지 않았습니다.**\n"
            "  다른 것을 재고서 같은 이름으로 보고하게 됩니다. 먼저 확인하세요."
        )
    if len(resolved) != MAX_CALLS:
        raise SystemExit(f"\n{MAX_CALLS}건이어야 하는데 {len(resolved)}건입니다.")

    if not args.go:
        print("\n아직 호출하지 않았습니다. 실행하려면 --go 를 붙이세요.")
        return

    from app.infrastructure.anthropic_client import FatalApiError, connect

    client = connect()
    budget = CallBudget()
    records: list[dict] = []
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'':-<70}")
    for item in resolved:
        serial = item["plan"]["serial"]
        try:
            record = run_one(client, item, budget)
        except FatalApiError as exc:
            # 계정 수준 오류 — 다음도 똑같이 실패한다. 남은 것을 던지지 않는다.
            print(f"  {serial}  중단: {exc}")
            break
        records.append(record)
        # 부른 것은 **곧바로** 남긴다. 뒤에서 죽어도 돈 쓴 결과는 지킨다.
        output.write_text(json.dumps(_bundle(records, budget), ensure_ascii=False,
                                     indent=1), encoding="utf-8")
        mark = "✓" if record["verdict"] == record["expected_verdict"] else "·"
        print(f"  {mark} {serial}  {item['plan']['kind']}  "
              f"판정 {record['verdict'] or record['api_error']}"
              f" (기대 {record['expected_verdict']}) · 회수 {record['recovered']}"
              f" · {record['latency_seconds']}s")

    bundle = _bundle(records, budget)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'':-<70}")
    print(f"호출 {budget.used}회 · 실제 비용 ${bundle['actual_cost']:.4f}")
    print(f"-> {output}")


def _bundle(records: list[dict], budget: CallBudget) -> dict:
    from app.infrastructure.anthropic_client import MODEL, PRICE_IN, PRICE_OUT

    cost = sum((r.get("input_tokens") or 0) / 1e6 * PRICE_IN
               + (r.get("output_tokens") or 0) / 1e6 * PRICE_OUT for r in records)
    return {
        "experiment": "E11b",
        "plan_document": "docs/18-e11b-case-selection.md",
        "model": MODEL,
        "calls_made": budget.used,
        "call_limit": budget.limit,
        "retries": 0,
        "actual_cost": round(cost, 4),
        "records": records,
    }
