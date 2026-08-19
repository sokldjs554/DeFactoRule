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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
