"""clean split 의 **결정론 프로파일** — 모델 없이 잴 수 있는 것만 잰다.

## 왜 별도 모듈인가

Phase B-2a 의 질문은 "clean split 에서 비-LLM 계층이 어떻게 달라지는가" 다.
기준선 채점은 기존 하네스(`evaluate.py`·`risk_coverage.py`)가 이미 하므로 여기서
다시 만들지 않는다. **하네스가 세지 않는 것만** 이 파일이 센다 —
데이터 무결성, 검색 도달률, 구간 분포, 규칙 전이, `R1~R10` 발화 수.

## 이 파일이 하지 않는 일

**아무것도 고르지 않는다.** 문턱을 다시 뽑지 않고, 규칙을 다시 유도하지 않고,
test 를 보고 무엇도 조정하지 않는다. 읽고 세는 것이 전부다. 그래서 산출물은
`experiments/results/clean/` 아래로만 나가고 legacy 산출물은 건드리지 않는다.

## 계승된 legacy 값은 여기서 씻기지 않는다

clean split 은 **선례 풀의 누수**를 고친 것이다. `DOUBT`·`TRUST`·
`RISK_CEILING`·keyword baseline 어휘는 전부 **legacy dev 85건**에서 나왔고,
그 85건 중 일부는 지금 clean test 안에 있다. 이 겹침을 `integrity()` 가 세어
보고서에 남긴다 — 씻긴 척하지 않기 위해서다.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from app.agents.workflow import year_of
from app.core.io import key_of
from app.domain.similarity import DOUBT, TRUST
from app.evaluation.confusable import idf_table, nearest, partition
from app.evaluation.metrics import wilson_interval

BANDS = ("trust", "middle", "doubt")
ROUTES = tuple(f"R{i}" for i in range(1, 11))
FIELDS = ("source", "serial", "page", "pair_index", "request", "label", "sector")


def band_of(score: float) -> str:
    """`app.agents.calibration.band_of` 와 같은 경계. 문턱은 도메인에서만 온다."""
    if score >= TRUST:
        return "trust"
    if score >= DOUBT:
        return "middle"
    return "doubt"


def counts(values) -> dict:
    """세되 **순서를 고정한다.** 보고서를 diff 하려면 키 순서가 안정해야 한다."""
    return dict(sorted(Counter(values).items()))


# ── A. 무결성 ────────────────────────────────────────────────────────
def integrity(clean_dev: list[dict], clean_test: list[dict],
              legacy_dev: list[dict], legacy_test: list[dict]) -> dict:
    """clean test 를 처음 여는 단계다. **먼저 데이터가 성한지 본다.**"""
    ck, dk = {key_of(r) for r in clean_test}, {key_of(r) for r in clean_dev}
    lk, mk = {key_of(r) for r in legacy_dev}, {key_of(r) for r in legacy_test}
    empty = {f: sum(1 for r in clean_test if not str(r.get(f) or "").strip())
             for f in FIELDS}
    triples = Counter((r["source"], r["page"], r["serial"]) for r in clean_test)
    return {
        "n_clean_test": len(clean_test),
        "n_clean_dev": len(clean_dev),
        "unique_keys_test": len(ck),
        "unique_keys_dev": len(dk),
        "duplicate_rows_test": len(clean_test) - len(ck),
        "dev_test_overlap": len(ck & dk),
        "union_preserved": (dk | ck) == (lk | mk),
        "empty_fields": {k: v for k, v in empty.items() if v},
        "source_page_serial_collisions": sum(1 for v in triples.values() if v > 1),
        "label_sources": counts(r.get("label_source") for r in clean_test),
        "masked_leaks_rows": sum(1 for r in clean_test
                                 if str(r.get("masked_leaks") or "0") != "0"),
        # 계승된 legacy 값이 오염되는 범위. 씻긴 척하지 않는다.
        "clean_test_seen_in_legacy_dev": len(ck & lk),
        "clean_dev_seen_in_legacy_dev": len(dk & lk),
    }


# ── B. 분포 ──────────────────────────────────────────────────────────
def distribution(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "by_label": counts(r["label"] for r in rows),
        "by_year": counts(str(year_of(r.get("source"))) for r in rows),
        "by_sector": counts(r.get("sector") or "—" for r in rows),
    }


# ── C. 검색 도달률 ───────────────────────────────────────────────────
def retrieval_profile(dev: list[dict], test: list[dict],
                      corpus: list[str]) -> tuple[dict, list[dict]]:
    """clean dev 를 선례 풀로 두고 clean test 각 행의 최근접 선례를 본다.

    E5·`trap_keys` 와 **같은 함수**(`nearest`·`partition`)를 쓴다. 함정 구간의
    정의가 실험마다 달라지면 비교가 무의미해진다.
    """
    links = nearest(test, dev, idf_table(corpus))
    groups = partition(links, DOUBT)
    where = {}
    for name, items in groups.items():
        for link in items:
            where[key_of(link["row"])] = name

    per_row = []
    for link in links:
        row, nb = link["row"], link["neighbor"]
        per_row.append({
            "key": list(key_of(row)),
            "label": row["label"],
            "year": year_of(row.get("source")),
            "top_similarity": round(link["similarity"], 4),
            "top_label": nb["label"] if nb else None,
            "band": band_of(link["similarity"]),
            "above_floor": link["similarity"] >= DOUBT,
            "partition": where[key_of(row)],
        })

    by_label: dict = defaultdict(lambda: Counter())
    by_year: dict = defaultdict(lambda: Counter())
    for item in per_row:
        by_label[item["label"]]["n"] += 1
        by_year[str(item["year"])]["n"] += 1
        if item["above_floor"]:
            by_label[item["label"]]["above_floor"] += 1
            by_year[str(item["year"])]["above_floor"] += 1
        by_label[item["label"]][item["partition"]] += 1

    above = sum(1 for i in per_row if i["above_floor"])
    lo, hi = wilson_interval(above, len(per_row))
    summary = {
        "floor": DOUBT,
        "n": len(per_row),
        "above_floor": above,
        "above_floor_rate": above / len(per_row) if per_row else None,
        "above_floor_ci95": [lo, hi],
        "bands": {b: sum(1 for i in per_row if i["band"] == b) for b in BANDS},
        "partition": {k: len(v) for k, v in groups.items()},
        "by_label": {k: dict(sorted(v.items())) for k, v in sorted(by_label.items())},
        "by_year": {k: dict(sorted(v.items())) for k, v in sorted(by_year.items())},
    }
    return summary, per_row


# ── D. E6 규칙 전이 ──────────────────────────────────────────────────
def atoms_of(rule: dict):
    """보고서 dict 를 유도기의 `Atom` 으로 되돌린다. 판정 논리를 베끼지 않는다."""
    from app.rules.induction import Atom

    return [Atom(a["kind"], a["value"]) for a in rule["atoms"]]


def fires_as_induced(rule: dict, row: dict) -> bool:
    """유도기가 보는 발화 — `sector`·`length`·`ngram` 을 모두 판정한다."""
    return all(a.holds(row) for a in atoms_of(rule))


def fires_as_router(rule: dict, row: dict) -> bool:
    """**Router 가 보는 발화.** 같지 않다 — 그것을 재려고 따로 둔다.

    `app.agents.workflow.rule_matches` 는 `ngram` 과 `length` 만 판정하고
    나머지 종류는 무조건 `False` 로 떨어뜨린다. legacy 규칙에는 `sector` 조건이
    없어서 드러나지 않았던 차이다.
    """
    from app.agents.workflow import rule_matches

    return rule_matches(rule, row["request"])


def rule_transfer(rules: list[dict], default: str, test: list[dict],
                  matcher=fires_as_induced) -> dict:
    """clean dev 에서 유도한 규칙이 clean test 에서 얼마나 쓰이는가.

    **재유도하지 않는다.** 규칙은 파일에서 읽은 그대로이고, 여기서 하는 것은
    발화 수와 정밀도를 세는 것뿐이다. `induction.measure_on` 과 같은 규칙 —
    앞선 규칙이 덮은 사례는 뒤 규칙으로 가지 않는다.
    """
    per_rule = [{"order": r["order"], "label": r["label"],
                 "description": r["description"],
                 "dev_support": r["dev_support"], "dev_precision": r["dev_precision"],
                 "test_support": 0, "test_correct": 0, "test_precision": None}
                for r in rules]
    fired_total = correct_total = 0
    default_rows = []
    for row in test:
        hit = None
        for slot, rule in zip(per_rule, rules):
            if matcher(rule, row):
                hit = slot
                break
        if hit is None:
            default_rows.append(row)
            continue
        hit["test_support"] += 1
        hit["test_correct"] += hit["label"] == row["label"]
        fired_total += 1
        correct_total += hit["label"] == row["label"]
    for slot in per_rule:
        if slot["test_support"]:
            slot["test_precision"] = slot["test_correct"] / slot["test_support"]

    default_correct = sum(1 for r in default_rows if r["label"] == default)
    lo, hi = wilson_interval(correct_total, fired_total)
    return {
        "n_rules": len(rules),
        "default_label": default,
        "test_n": len(test),
        "fired": fired_total,
        "coverage": fired_total / len(test) if test else None,
        "correct_when_fired": correct_total,
        "precision_when_fired": correct_total / fired_total if fired_total else None,
        "precision_ci95": [lo, hi],
        "fell_to_default": len(default_rows),
        "default_correct": default_correct,
        "default_label_distribution": counts(r["label"] for r in default_rows),
        "rules_that_never_fire": [s["order"] for s in per_rule
                                  if s["test_support"] == 0],
        "whole_model_accuracy": ((correct_total + default_correct) / len(test)
                                 if test else None),
        "per_rule": per_rule,
    }


# ── E. Router 경로 ───────────────────────────────────────────────────
def router_profile(rows: list[dict], states: list) -> dict:
    """`R1~R10` 이 각각 몇 번 발화했는가. **정책은 건드리지 않는다.**"""
    routes: Counter = Counter()
    paths: Counter = Counter()
    reasons: Counter = Counter()
    per_label: dict = defaultdict(lambda: Counter())
    per_route_label: dict = defaultdict(lambda: Counter())
    answered = correct = wrong = 0
    would_be_correct = 0          # 기권한 것이 답했다면 맞았을까

    for row, state in zip(rows, states):
        routes[state.route_reason or "—"] += 1
        paths[state.route.value if state.route else "—"] += 1
        cell = per_label[row["label"]]
        cell["n"] += 1
        per_route_label[state.route_reason or "—"][row["label"]] += 1
        if state.abstained:
            reasons[state.abstention_reason.value
                    if state.abstention_reason else "—"] += 1
            cell["abstained"] += 1
            if state.provisional == row["label"]:
                would_be_correct += 1
                cell["would_be_correct"] += 1
        else:
            answered += 1
            cell["answered"] += 1
            if state.decision == row["label"]:
                correct += 1
                cell["correct"] += 1
            else:
                wrong += 1
                cell["wrong"] += 1

    lo, hi = wilson_interval(correct, answered)
    return {
        "n": len(rows),
        "routes": {r: routes.get(r, 0) for r in ROUTES},
        "routes_other": {k: v for k, v in routes.items() if k not in ROUTES},
        "paths": dict(sorted(paths.items())),
        "abstention_reasons": dict(sorted(reasons.items())),
        "answered": answered,
        "abstained": len(rows) - answered,
        "coverage": answered / len(rows) if rows else None,
        "correct": correct,
        "wrong": wrong,
        "accuracy_on_answered": correct / answered if answered else None,
        "accuracy_ci95": [lo, hi],
        "abstained_would_be_correct": would_be_correct,
        "by_label": {k: dict(sorted(v.items())) for k, v in sorted(per_label.items())},
        "by_route_label": {k: dict(sorted(v.items()))
                           for k, v in sorted(per_route_label.items())},
    }


def trap_breakdown(rows: list[dict], states: list, per_row: list[dict]) -> dict:
    """함정 / 순응 / 선례없음 구간별로 Router 가 무엇을 했는가."""
    where = {tuple(i["key"]): i["partition"] for i in per_row}
    out: dict = {}
    for name in ("agree", "trap", "unanchored"):
        c: Counter = Counter()
        for row, state in zip(rows, states):
            if where.get(key_of(row)) != name:
                continue
            c["n"] += 1
            if state.abstained:
                c["기권"] += 1
            elif state.decision == row["label"]:
                c["맞힘"] += 1
            else:
                c["틀림"] += 1
        lo, hi = wilson_interval(c["맞힘"], c["맞힘"] + c["틀림"])
        out[name] = {
            "n": c["n"], "correct": c["맞힘"], "wrong": c["틀림"],
            "abstained": c["기권"],
            "accuracy_on_answered": (c["맞힘"] / (c["맞힘"] + c["틀림"])
                                     if c["맞힘"] + c["틀림"] else None),
            "accuracy_ci95": [lo, hi],
        }
    return out


# ── F. `조치` 추적 ───────────────────────────────────────────────────
def action_class_trace(rows: list[dict], states: list, per_row: list[dict]) -> dict:
    """`조치` 14건이 **어디까지 평가되는가.**

    이 코퍼스에서 가장 비싼 오류가 여기 있다. 전체 평균은 `비조치` 121건이
    끌고 가므로, 소수 클래스는 따로 세지 않으면 보이지 않는다.
    """
    info = {tuple(i["key"]): i for i in per_row}
    out = []
    for row, state in zip(rows, states):
        if row["label"] != "조치":
            continue
        item = info[key_of(row)]
        out.append({
            "key": list(key_of(row)), "year": item["year"],
            "top_similarity": item["top_similarity"], "top_label": item["top_label"],
            "band": item["band"], "partition": item["partition"],
            "route_reason": state.route_reason,
            "path": state.route.value if state.route else None,
            "abstained": state.abstained,
            "abstention_reason": (state.abstention_reason.value
                                  if state.abstention_reason else None),
            "decision": state.decision, "provisional": state.provisional,
            "correct": (not state.abstained) and state.decision == row["label"],
        })
    return {
        "n": len(out),
        "above_floor": sum(1 for o in out if o["band"] != "doubt"),
        "answered": sum(1 for o in out if not o["abstained"]),
        "correct": sum(1 for o in out if o["correct"]),
        "top_label_is_조치": sum(1 for o in out if o["top_label"] == "조치"),
        "routes": counts(o["route_reason"] or "—" for o in out),
        "cases": out,
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from app.agents.experiments import run_variant
    from app.core.io import load_jsonl, write_jsonl
    from app.core.paths import EVAL, PROCESSED, RESULTS
    from app.retrieval.lexical import LexicalRetriever

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(EVAL / "nonaction_dev_clean.jsonl"))
    ap.add_argument("--test", default=str(EVAL / "nonaction_test_clean.jsonl"))
    ap.add_argument("--legacy-dev", default=str(EVAL / "nonaction_dev.jsonl"))
    ap.add_argument("--legacy-test", default=str(EVAL / "nonaction_test.jsonl"))
    ap.add_argument("--rules", default=str(RESULTS / "e6_rules_clean.json"))
    ap.add_argument("--risk", default=str(RESULTS / "trap_risk_clean.json"))
    ap.add_argument("--out-dir", default=str(RESULTS / "clean"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.resolve() == Path(RESULTS).resolve():
        raise SystemExit("clean 산출물을 legacy 자리에 쓰려 했습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]
    test = [r for r in load_jsonl(Path(args.test)) if r.get("label")]
    legacy_dev = load_jsonl(Path(args.legacy_dev))
    legacy_test = load_jsonl(Path(args.legacy_test))
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
              for c in cases]
    corpus = [t for t in corpus if t]

    payload = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    rules, default = payload["rules"], payload["default_label"]
    risk = json.loads(Path(args.risk).read_text(encoding="utf-8"))
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]

    # Router 는 결정론이다 — 모델을 부르지 않는다. 여기서 도는 것은 검색과
    # 수치 비교뿐이고, 예측 파일은 **clean 디렉터리로만** 나간다.
    states = run_variant("router", LexicalRetriever, dev, rules, risk, test,
                         corpus, fallback)
    pred_path = out_dir / "pred_agent_router.jsonl"
    write_jsonl(pred_path, [s.to_prediction() for s in states])

    retrieval, per_row = retrieval_profile(dev, test, corpus)
    report = {
        "inputs": {"dev": args.dev, "test": args.test, "rules": args.rules,
                   "risk": args.risk, "corpus_size": len(corpus),
                   "floor": DOUBT, "trust": TRUST,
                   "thresholds_derived_from": "legacy dev — 이 단계에서 재유도하지 않았다"},
        "integrity": integrity(dev, test, legacy_dev, legacy_test),
        "distribution": {"clean_dev": distribution(dev), "clean_test": distribution(test)},
        "retrieval": retrieval,
        "rule_transfer_induction_semantics": rule_transfer(rules, default, test),
        "rule_transfer_router_semantics": rule_transfer(rules, default, test,
                                                        matcher=fires_as_router),
        "router": router_profile(test, states),
        "trap_breakdown": trap_breakdown(test, states, per_row),
        "action_class": action_class_trace(test, states, per_row),
        "predictions": str(pred_path),
    }
    (out_dir / "deterministic_profile.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "retrieval_rows.json").write_text(
        json.dumps(per_row, ensure_ascii=False, indent=2), encoding="utf-8")

    it, rt = report["integrity"], report["retrieval"]
    print(f"clean test {it['n_clean_test']}건 · 유일 키 {it['unique_keys_test']} · "
          f"dev 와 겹침 {it['dev_test_overlap']} · 합집합 보존 {it['union_preserved']}")
    print(f"legacy dev 에 있던 clean test 행 {it['clean_test_seen_in_legacy_dev']}건 "
          f"— 계승 문턱이 오염되는 범위")
    print(f"\n문턱 위 선례 {rt['above_floor']}/{rt['n']} "
          f"({rt['above_floor_rate']:.1%}) CI [{rt['above_floor_ci95'][0]:.3f}, "
          f"{rt['above_floor_ci95'][1]:.3f}]")
    print(f"구간 {rt['bands']} · 분할 {rt['partition']}")

    r = report["router"]
    print("\n경로  " + " · ".join(f"{k} {v}" for k, v in r["routes"].items()))
    acc = r["accuracy_on_answered"]
    print(f"답변 {r['answered']} / 기권 {r['abstained']} · 답한 것 정확도 "
          f"{acc:.3f}" if acc is not None else "답한 것 0건")
    for name, cell in report["trap_breakdown"].items():
        print(f"  {name:<11}{cell['n']:>4}  맞힘 {cell['correct']:>3} "
              f"틀림 {cell['wrong']:>3} 기권 {cell['abstained']:>3}")

    for view in ("rule_transfer_induction_semantics", "rule_transfer_router_semantics"):
        cell = report[view]
        prec = cell["precision_when_fired"]
        print(f"\n{view}: 발화 {cell['fired']}/{cell['test_n']} "
              f"({cell['coverage']:.1%}) · 발화 시 정밀도 "
              f"{'—' if prec is None else round(prec, 3)}"
              f" · 기본 라벨로 {cell['fell_to_default']}건")

    a = report["action_class"]
    print(f"\n조치 {a['n']}건 — 문턱 위 {a['above_floor']} · "
          f"top-1 이 조치인 것 {a['top_label_is_조치']} · "
          f"답변 {a['answered']} · 맞힘 {a['correct']}")
    print(f"\n-> {out_dir / 'deterministic_profile.json'}")
