#!/usr/bin/env python3
"""Agent Workflow — CLI. 구현은 `app.agents.*` 에 있다.

    python3 scripts/agent.py run --variant router     예측 파일을 낸다
    python3 scripts/agent.py experiment               E8~E11a 를 한 번에

`run` 이 내는 파일은 **기존 하네스가 읽는 형식**이다. 그래서 evaluate.py ·
comparison.py · risk_coverage.py 를 그대로 쓸 수 있다 — 새 채점기를 만들지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter  # noqa: E402

from app.agents.experiments import (  # noqa: E402
    COMPARISONS,
    dump,
    report,
    run_variant,
    trap_keys,
)
from app.agents.workflow import VARIANTS  # noqa: E402
from app.core.io import load_jsonl, write_jsonl  # noqa: E402
from app.core.paths import EVAL, PROCESSED, RESULTS  # noqa: E402
from app.retrieval.lexical import LexicalRetriever  # noqa: E402


def load_everything(args):
    dev = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]
    test = [r for r in load_jsonl(Path(args.gold)) if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
              for c in cases]
    corpus = [t for t in corpus if t]
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))["rules"]
    risk = json.loads(Path(args.risk).read_text(encoding="utf-8"))
    from collections import Counter
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]
    return dev, test, corpus, rules, risk, fallback


def cmd_run(args) -> None:
    dev, test, corpus, rules, risk, fallback = load_everything(args)
    states = run_variant(args.variant, LexicalRetriever, dev, rules, risk, test,
                         corpus, fallback)
    out = Path(args.output or PROCESSED / f"pred_nonaction_agent_{args.variant}.jsonl")
    write_jsonl(out, [s.to_prediction() for s in states])
    answered = sum(1 for s in states if not s.abstained)
    print(f"{args.variant}: {len(states)}건 · 답변 {answered} "
          f"({answered / len(states):.1%}) -> {out}")


def cmd_experiment(args) -> None:
    dev, test, corpus, rules, risk, fallback = load_everything(args)
    trap = trap_keys(dev, test, corpus)

    results = {}
    for name in VARIANTS:
        results[name] = run_variant(name, LexicalRetriever, dev, rules, risk, test,
                                    corpus, fallback)
        write_jsonl(PROCESSED / f"pred_nonaction_agent_{name}.jsonl",
                    [s.to_prediction() for s in results[name]])

    payload = report(test, results, trap, f"cases_nonaction {len(corpus)}건")
    payload["comparisons"] = [
        {"id": eid, "a": a, "b": b, "question": q} for eid, a, b, q in COMPARISONS]

    print(f"test {len(test)}건 · 함정 구간 {len(trap)}건 · 선례 풀 {len(dev)}건\n")
    print(f"{'변형':<20}{'커버리지':>9}{'답한 것 정확도':>15}{'매크로 F1':>11}"
          f"{'근거없는 주장':>13}")
    for name, stats in payload["variants"].items():
        acc = stats["accuracy_on_answered"]
        print(f"{name:<20}{stats['coverage']:>9.1%}"
              f"{(f'{acc:.3f}' if acc is not None else '—'):>15}"
              f"{stats['macro_f1_on_answered']:>11.3f}"
              f"{stats['unsupported_claim_rate']:>13.1%}")

    print("\n기권이 정당했는가 — 답했다면 틀렸을 비율 (커버리지와 함께 볼 것)")
    print(f"{'변형':<20}{'커버리지':>9}{'기권':>6}{'정당':>6}{'비율':>8}   95% CI")
    for name, stats in payload["variants"].items():
        a = stats["abstention_accuracy"]
        rate = f"{a['rate']:.3f}" if a["rate"] is not None else "—"
        ci = a["ci95"]
        print(f"{name:<20}{stats['coverage']:>9.1%}{a['abstained']:>6}"
              f"{a['justified']:>6}{rate:>8}   [{ci[0]:.3f}, {ci[1]:.3f}]")

    print(f"\n주 종점 — 함정 구간 {len(trap)}건 (사전 등록)")
    print(f"{'변형':<20}{'맞힘':>5}{'틀림':>5}{'기권':>5}{'정확도':>9}   95% CI")
    for name, stats in payload["variants"].items():
        t = stats["trap"]
        acc = f"{t['accuracy']:.3f}" if t["accuracy"] is not None else "—"
        ci = t["accuracy_ci95"]
        print(f"{name:<20}{t['correct']:>5}{t['wrong']:>5}{t['abstained']:>5}"
              f"{acc:>9}   [{ci[0]:.3f}, {ci[1]:.3f}]")

    path = RESULTS / "e8_e11_agent.json"
    dump(path, payload)
    print(f"\n-> {path}")


def cmd_applicability(args) -> None:
    """E11b — 선례가 실제로 적용되는가. **사정거리가 좁다는 것을 먼저 보여준다.**"""
    from app.agents.applicability import (
        MAX_TOKENS,
        SYSTEM,
        apply_verdict,
        build_prompt,
        estimate_cost,
        quotes_are_grounded,
        schema,
        targets,
    )
    from app.core.audit import Discards
    from app.domain.similarity import DOUBT

    dev, test, corpus, rules, risk, fallback = load_everything(args)
    states = run_variant("router", LexicalRetriever, dev, rules, risk, test,
                         corpus, fallback)
    scope = targets(states, test, DOUBT)
    abstained = [i for i, s in enumerate(states) if s.abstained]
    wasted = [i for i in abstained if states[i].provisional == test[i]["label"]]

    print(f"기권 {len(abstained)}건 · 그중 답했어도 맞았을 것 {len(wasted)}건")
    print(f"이 검사가 닿는 건: {len(scope)}건 "
          f"({len(scope) / max(1, len(abstained)):.0%}) — 선례가 문턱 위인 기권만")
    reachable = [i for i in scope if i in set(wasted)]
    print(f"  그중 아까운 것 {len(reachable)}건 = **회수 상한**")

    limit = args.limit or len(scope)
    chosen = scope[:limit]
    print(f"\n이번에 부를 건: {len(chosen)}건 · 추정 비용 ${estimate_cost(len(chosen)):.2f}")

    if not chosen:
        print("\n부를 건이 없습니다 — 선례가 문턱 위인 기권이 하나도 없습니다.")
        return

    def key_of_row(row):
        return (row["source"], row["page"], str(row["serial"]),
                row.get("pair_index", 1))

    if args.dry_run:
        i = chosen[0]
        neighbor = states[i].retrieved_evidence[0]
        origin = next((p for p in dev
                       if f"prec:{p['source']}#{p['serial']}" == neighbor.id), None)
        print("\n--dry-run 이므로 요청을 보내지 않습니다. 첫 건의 프롬프트:\n")
        print("-" * 70)
        print(build_prompt(test[i]["request"], (origin or {}).get("request", ""))[:1200])
        print("-" * 70)
        print(f"\n시스템 프롬프트 {len(SYSTEM)}자 · 출력 상한 {MAX_TOKENS}")
        return

    from app.infrastructure.anthropic_client import (
        FatalApiError,
        call_structured,
        connect,
        preflight,
    )
    from app.infrastructure.anthropic_client import estimate_cost as actual_cost

    out = Path(args.output or PROCESSED / "applicability.jsonl")
    # 키 -> 레코드. 재시도하면 **덮어쓴다** — 실패 행을 그대로 두고 새 행을
    # 덧붙이면 성공/실패 개수가 이중으로 세어진다.
    saved: dict = {}
    if args.resume and out.exists():
        for record in load_jsonl(out):
            saved[(record["source"], record["page"], record["serial"],
                   record["pair_index"])] = record
        done = {k for k, r in saved.items() if "error" not in r}
        print(f"  이어하기: {len(done)}건은 건너뜁니다.")
    else:
        done = set()

    # **저장된 판정을 상태에 먼저 반영한다.** 안 하면 이어하기 뒤의 회수 건수와
    # 정확도가 이번 실행분만 세어져 통째로 틀린다.
    index_of = {key_of_row(row): i for i, row in enumerate(test)}
    for key, record in saved.items():
        i = index_of.get(key)
        if i is None or "verdict" not in record or record.get("ungrounded"):
            continue
        apply_verdict(states[i], record["verdict"])

    client = connect()
    preflight(client, schema())
    discards = Discards("applicability")

    try:
        for n, i in enumerate(chosen, 1):
            row, state = test[i], states[i]
            key = key_of_row(row)
            if key in done:
                continue
            neighbor = state.retrieved_evidence[0]
            origin = next((p for p in dev
                           if f"prec:{p['source']}#{p['serial']}" == neighbor.id), {})
            prompt = build_prompt(row["request"], origin.get("request", ""))
            result = call_structured(client, SYSTEM, prompt, schema(),
                                     max_tokens=MAX_TOKENS, effort="low")
            record = {"source": key[0], "page": key[1], "serial": key[2],
                      "pair_index": key[3], "neighbor": neighbor.id,
                      "similarity": round(neighbor.score, 4)}
            if "error" in result:
                record.update(result)
            else:
                data = result["data"]
                bad = quotes_are_grounded(data, row["request"],
                                          origin.get("request", ""))
                if bad:
                    discards.drop({"key": str(key), "fields": bad},
                                  ["인용이 원문에 없다"])
                else:
                    apply_verdict(state, data["verdict"])
                record.update({**data, "ungrounded": bad,
                               "input_tokens": result["input_tokens"],
                               "output_tokens": result["output_tokens"]})
            saved[key] = record        # 재시도면 덮어쓴다
            write_jsonl(out, list(saved.values()))
            if n % 5 == 0:
                print(f"  {n}/{len(chosen)}")
    except FatalApiError as exc:
        print(f"\n중단 — 계정 수준 오류입니다.\n  {exc}")
        print(f"  여기까지 {len(saved)}건 저장. --resume 으로 이어가세요.")

    # 보고는 **저장된 전체**에서 센다 — 이번 실행분만 세면 이어하기 뒤에 틀린다.
    records = list(saved.values())
    ok = [r for r in records if "error" not in r]
    verdicts = Counter(r["verdict"] for r in ok if "verdict" in r)
    recovered = sum(1 for i in chosen
                    if not states[i].abstained and states[i].route_reason == "V3+")

    print(f"\n{len(records)}건 처리 · 성공 {len(ok)} · 실패 {len(records) - len(ok)}")
    print(f"  판정 분포: {dict(verdicts)}")
    ungrounded = sum(1 for r in ok if r.get("ungrounded"))
    rate = ungrounded / len(ok) if ok else 0.0
    print(f"  인용 미대조 {ungrounded}건 ({rate:.1%})"
          + ("  ⚠ 20% 초과 — 사전 등록한 중단 조건입니다" if rate > 0.20 else ""))
    print(f"  기권 회수 {recovered}건 (상한 {len(reachable)})")
    print(f"  실제 비용 ${actual_cost(records):.3f}")
    print(f"-> {out}")

    if recovered:
        correct = sum(1 for i in chosen
                      if states[i].route_reason == "V3+"
                      and states[i].decision == test[i]["label"])
        print(f"\n  회수한 {recovered}건 중 맞은 것 {correct}건")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(EVAL / "nonaction_dev.jsonl"))
    ap.add_argument("--gold", default=str(EVAL / "nonaction_test.jsonl"))
    ap.add_argument("--rules", default=str(RESULTS / "e6_rules.json"))
    ap.add_argument("--risk", default=str(RESULTS / "trap_risk.json"))
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="한 변형을 돌려 예측 파일을 낸다")
    run.add_argument("--variant", choices=sorted(VARIANTS), default="router")
    run.add_argument("--output")
    run.set_defaults(func=cmd_run)

    exp = sub.add_parser("experiment", help="E8~E11a 를 한 번에 (API 없음)")
    exp.set_defaults(func=cmd_experiment)

    app_ = sub.add_parser("applicability",
                          help="E11b — 선례가 실제로 적용되는가 (API)")
    app_.add_argument("--limit", type=int, default=0)
    app_.add_argument("--dry-run", action="store_true")
    app_.add_argument("--resume", action="store_true")
    app_.add_argument("--output")
    app_.set_defaults(func=cmd_applicability)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
