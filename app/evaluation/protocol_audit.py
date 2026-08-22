"""clean 평가 규약 감사 — **무엇이 아직 legacy 에서 오는가.**

## 왜 이것을 B-2b 앞에 두는가

`docs/24` 가 남긴 두 개의 구멍이다.

    ① clean test 168건 중 54건이 legacy dev 에 있던 행이다.
       그런데 문턱·어휘는 그 legacy dev 에서 정해졌다.
    ② clean E6 8번 규칙이 Router 안에서 발화하지 않는다.

LLM 을 붙이기 전에 **재는 자를 먼저 검사한다.** 자가 틀어져 있으면 모델
수치는 그 위에 쌓인다.

## 이 파일이 하는 일과 하지 않는 일

한다: legacy 값과 clean dev 재계산 값을 **나란히 놓는다.** 각 문턱이 관측된
신호 분포에서 **얼마나 움직여도 출력이 같은지**(무감구간)를 잰다.

하지 않는다: **아무것도 바꾸지 않는다.** production 문턱은 그대로이고, 새 값을
clean test 에 적용하지 않는다. 이 파일이 내는 것은 표 하나뿐이다.

## 문턱마다 성격이 다르다 — 그것부터 갈라야 한다

    학습된 것    데이터에서 계산된다. split 이 바뀌면 다시 뽑아야 한다.
    구조적인 것  자료구조가 값을 정한다. top-3 의 과반은 언제나 2/3 다.
    임의의 것    사람이 골랐고 데이터가 고르지 않았다. 다시 뽑을 절차가 없다.

셋을 뭉뚱그려 "legacy 문턱" 이라고 부르면, 다시 뽑을 수 있는 것과 애초에
뽑은 적이 없는 것을 같은 문제로 취급하게 된다.
"""

from __future__ import annotations

from collections import Counter

from app.agents.calibration import bands_are_separable
from app.evaluation.metrics import wilson_interval

# 후보 문턱을 훑는 격자. 값을 고르려는 것이 아니라 **legacy 값이 clean dev 에서도
# 허용되는 영역 안에 있는지** 보려는 것이다.
DOUBT_GRID = [round(0.05 + 0.01 * i, 2) for i in range(46)]      # 0.05 ~ 0.50
TRUST_GRID = [round(0.30 + 0.01 * i, 2) for i in range(61)]      # 0.30 ~ 0.90
MIN_BAND_N = 5   # 이보다 적은 구간으로 문턱을 정하면 한두 건이 정책을 정한다


def band_table_at(links: list[dict], doubt: float, trust: float) -> dict:
    """후보 문턱 한 쌍에서의 구간표. `calibration.risk_table` 과 같은 셈법."""
    def band(sim: float) -> str:
        return "trust" if sim >= trust else ("middle" if sim >= doubt else "doubt")

    table = {}
    for name in ("trust", "middle", "doubt"):
        group = [x for x in links if band(x["similarity"]) == name]
        wrong = sum(1 for x in group if x["wrong"])
        lo, hi = wilson_interval(wrong, len(group))
        table[name] = {"n": len(group), "wrong": wrong,
                       "risk": wrong / len(group) if group else None,
                       "ci95": [lo, hi]}
    return {"by_band": table}


def admissible(links: list[dict], doubt_grid=DOUBT_GRID,
               trust_grid=TRUST_GRID, min_band_n: int = MIN_BAND_N) -> dict:
    """구간이 갈리는 문턱 쌍을 전부 찾는다.

    원래 절차는 최적화가 아니었다 — 둥근 값을 고르고 **구간 분리로 정당화**했다
    (`app/domain/similarity.py`). 그래서 여기서도 최적값을 고르지 않는다.
    "정당화가 통과하는 영역" 을 통째로 그리고, legacy 값이 그 안에 있는지만 본다.
    """
    ok = []
    for doubt in doubt_grid:
        for trust in trust_grid:
            if trust <= doubt:
                continue
            table = band_table_at(links, doubt, trust)
            bands = table["by_band"]
            if bands["trust"]["n"] < min_band_n or bands["doubt"]["n"] < min_band_n:
                continue
            separable, _ = bands_are_separable(bands)
            if separable:
                ok.append((doubt, trust))
    return {
        "n_candidates": len(doubt_grid) * len(trust_grid),
        "n_admissible": len(ok),
        "doubt_range": [min(d for d, _ in ok), max(d for d, _ in ok)] if ok else None,
        "trust_range": [min(t for _, t in ok), max(t for _, t in ok)] if ok else None,
        "pairs": [list(p) for p in ok],
    }


def gap_around(values: list[float], point: float) -> dict:
    """`point` 를 어디까지 움직여도 **같은 편에 있는 값이 하나도 안 바뀌는가.**

    문턱의 무감구간이다. 관측된 값들 사이의 빈 구간이 넓으면, 그 문턱은
    소수점 둘째 자리를 다툴 이유가 없다.
    """
    below = [v for v in values if v < point]
    above = [v for v in values if v >= point]
    return {
        "lower": max(below) if below else None,
        "upper": min(above) if above else None,
        "n_below": len(below), "n_above": len(above),
    }


def signal_frame(states) -> list[dict]:
    """Router 가 실제로 본 신호를 그대로 꺼낸다. **다시 계산하지 않는다.**"""
    out = []
    for state in states:
        s = state.signals
        if s is None:
            continue
        out.append({
            # `RouterSignals.top_similarity` 는 **이미 DOUBT 로 걸러진 뒤**의 값이다.
            # 문턱의 무감구간을 재려면 걸러지기 전의 원 유사도를 봐야 한다.
            "precedent_score": state.precedent_score,
            "trap_risk": s.trap_risk, "top_similarity": s.top_similarity,
            "margin": s.margin, "label_agreement": s.label_agreement,
            "source_diversity": s.source_diversity, "recency_gap": s.recency_gap,
            "evidence_count": s.evidence_count, "rule_fired": s.rule_fired,
            "rule_conflict": s.rule_conflict,
            "route_reason": state.route_reason,
        })
    return out


def atom_kinds(rules: list[dict]) -> dict:
    """규칙이 쓰는 조건 종류. Router 가 읽을 수 있는 것은 둘뿐이다."""
    kinds: Counter = Counter()
    unreadable = []
    for rule in rules:
        for atom in rule["atoms"]:
            kinds[atom["kind"]] += 1
            if atom["kind"] not in ("ngram", "length"):
                unreadable.append({"order": rule["order"],
                                   "description": rule["description"],
                                   "kind": atom["kind"], "value": atom["value"]})
    return {"kinds": dict(sorted(kinds.items())),
            "unreadable_by_router": unreadable,
            "n_rules_unreadable": len({u["order"] for u in unreadable})}


def _base_rate_summary(table: dict) -> dict:
    """기저율 표에서 프롬프트에 실제로 들어가는 부분만 꺼낸다."""
    return {
        "n": table["n"],
        "overall": {k: round(v, 4) for k, v in table["overall"].items()},
        "reliable_sectors": sorted(s for s, v in table["sectors"].items()
                                   if v["reliable"]),
        "sector_rates": {s: {k: round(x, 4) for k, x in v["rates"].items()}
                         for s, v in sorted(table["sectors"].items())
                         if v["reliable"]},
    }


def keyword_vocabulary(rows: list[dict]) -> dict:
    """`nonaction.RULES` 어휘가 이 행 집합에서 어떻게 걸리는가.

    이 어휘는 **사람이 legacy dev 85건을 읽고 뽑았다**(`app/rules/nonaction.py`).
    스크립트가 없으므로 기계적으로 다시 뽑을 수 없다 — 그 사실을 수치와 함께
    남긴다.
    """
    from app.rules.nonaction import RULES

    out = []
    for label, pattern in RULES:
        hits = [r for r in rows if pattern.search(" ".join(r["request"].split()))]
        correct = sum(1 for r in hits if r["label"] == label)
        lo, hi = wilson_interval(correct, len(hits))
        out.append({"label": label, "pattern": pattern.pattern,
                    "support": len(hits), "correct": correct,
                    "precision": correct / len(hits) if hits else None,
                    "ci95": [lo, hi]})
    return {"n_rows": len(rows), "rules": out}


def provenance_split(gold: list[dict], preds: list[dict],
                     legacy_dev: list[dict]) -> dict:
    """이미 계산된 예측을 **legacy dev 에 있던 행 / 아닌 행**으로 쪼갠다.

    새 파라미터를 적용하는 것이 아니다. `docs/24` 가 낸 예측 파일을 두 무리로
    나눠 세는 것뿐이다. 어휘·문턱이 legacy dev 에서 왔다면, 그 dev 에 실려
    있던 행에서 성적이 더 좋아야 한다 — 그 가설을 수치로 본다.
    """
    from app.core.io import key_of

    seen = {key_of(r) for r in legacy_dev}
    truth = {key_of(r): r["label"] for r in gold}
    groups = {"legacy_dev_에_있던_행": [], "처음_보는_행": []}
    for pred in preds:
        key = key_of(pred)
        if key not in truth:
            continue
        bucket = "legacy_dev_에_있던_행" if key in seen else "처음_보는_행"
        groups[bucket].append((truth[key], pred["predicted"],
                               bool(pred.get("abstained"))))

    out = {}
    for name, items in groups.items():
        answered = [(g, p) for g, p, ab in items if not ab]
        correct = sum(1 for g, p in answered if g == p)
        all_correct = sum(1 for g, p, _ in items if g == p)
        lo, hi = wilson_interval(correct, len(answered))
        out[name] = {
            "n": len(items),
            "accuracy_all_rows": all_correct / len(items) if items else None,
            "answered": len(answered),
            "accuracy_on_answered": correct / len(answered) if answered else None,
            "ci95": [lo, hi],
        }
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from app.agents.calibration import loo_links
    from app.agents.experiments import run_variant
    from app.agents.router import (
        MAX_YEAR_GAP,
        MIN_AGREEMENT,
        MIN_MARGIN,
        RISK_CEILING,
    )
    from app.core.io import load_jsonl
    from app.core.paths import EVAL, PROCESSED, RESULTS
    from app.domain.base_rates import compute as base_rates_compute
    from app.domain.similarity import DOUBT, TRUST
    from app.evaluation.confusable import idf_table
    from app.retrieval.lexical import LexicalRetriever

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(EVAL / "nonaction_dev_clean.jsonl"))
    ap.add_argument("--test", default=str(EVAL / "nonaction_test_clean.jsonl"))
    ap.add_argument("--legacy-dev", default=str(EVAL / "nonaction_dev.jsonl"))
    ap.add_argument("--rules", default=str(RESULTS / "e6_rules_clean.json"))
    ap.add_argument("--legacy-rules", default=str(RESULTS / "e6_rules.json"))
    ap.add_argument("--risk", default=str(RESULTS / "trap_risk_clean.json"))
    ap.add_argument("--out", default=str(RESULTS / "clean" / "protocol_audit.json"))
    args = ap.parse_args()

    dev = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]
    test = [r for r in load_jsonl(Path(args.test)) if r.get("label")]
    legacy_dev = [r for r in load_jsonl(Path(args.legacy_dev)) if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
              for c in cases]
    corpus = [t for t in corpus if t]
    idf = idf_table(corpus)

    clean_links = loo_links(dev, idf)
    legacy_links = loo_links(legacy_dev, idf)

    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))["rules"]
    legacy_rules = json.loads(Path(args.legacy_rules).read_text(encoding="utf-8"))["rules"]
    risk = json.loads(Path(args.risk).read_text(encoding="utf-8"))

    # Router 를 한 번 돌려 **신호만** 꺼낸다. 파라미터는 하나도 바꾸지 않는다.
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]
    states = run_variant("router", LexicalRetriever, dev, rules, risk, test,
                         corpus, fallback)
    signals = signal_frame(states)

    top_sims = sorted(s["precedent_score"] for s in signals)   # 걸러지기 전 원 유사도
    band_risks = sorted({round(s["trap_risk"], 6) for s in signals})
    agreements = sorted({round(s["label_agreement"], 6) for s in signals})
    margins = [s["margin"] for s in signals]
    gaps = [s["recency_gap"] for s in signals if s["recency_gap"] is not None]

    report = {
        "note": "어떤 문턱도 바꾸지 않았다. legacy 값과 clean dev 재계산 값을 나란히 둔 표다.",
        "doubt_trust": {
            "legacy_value": {"doubt": DOUBT, "trust": TRUST},
            "procedure": "값을 고르고 dev LOO 구간 분리로 정당화한다 (최적화 아님)",
            "legacy_dev_admissible": admissible(legacy_links),
            "clean_dev_admissible": admissible(clean_links),
            "legacy_pair_admissible_on_clean_dev":
                [DOUBT, TRUST] in [list(p) for p in admissible(clean_links)["pairs"]],
            "clean_dev_band_table_at_legacy_pair": band_table_at(clean_links, DOUBT, TRUST),
            # 정당화 절차가 값을 좁히지 못한다면 그것부터 적어야 한다.
            "clean_dev_similarity_shape": {
                "below_doubt": sum(1 for x in clean_links if x["similarity"] < DOUBT),
                "middle": sum(1 for x in clean_links
                              if DOUBT <= x["similarity"] < TRUST),
                "above_trust": sum(1 for x in clean_links if x["similarity"] >= TRUST),
            },
            "clean_dev_gap_at_doubt": gap_around(
                sorted(x["similarity"] for x in clean_links), DOUBT),
            "clean_dev_gap_at_trust": gap_around(
                sorted(x["similarity"] for x in clean_links), TRUST),
        },
        "observed_test_signal_gaps": {
            "note": "이미 돌아간 Router 실행의 신호 분포다. 새 값을 적용한 것이 아니다.",
            "doubt": gap_around(top_sims, DOUBT),
            "trust": gap_around(top_sims, TRUST),
        },
        "risk_ceiling": {
            "legacy_value": RISK_CEILING,
            "procedure": "믿음 구간 오류율의 Wilson 95% 상한을 소수 둘째 자리로 올린다",
            "legacy_derived": 0.201,
            "clean_derived": round(
                json.loads((RESULTS / "trap_risk_clean.json").read_text(
                    encoding="utf-8"))["by_band"]["trust"]["ci95"][1], 4),
            "observed_trap_risk_values": band_risks,
            "insensitivity_interval": gap_around(band_risks, RISK_CEILING),
        },
        "min_agreement": {
            "legacy_value": MIN_AGREEMENT,
            "procedure": "top-3 의 과반 — 자료구조가 정한다 (학습 아님)",
            "observed_values": agreements,
            "insensitivity_interval": gap_around(agreements, MIN_AGREEMENT),
        },
        "min_margin": {
            "legacy_value": MIN_MARGIN,
            "procedure": "유사도 분해능 — 사람이 고른 값 (학습 아님)",
            "n_below": sum(1 for m in margins if m < MIN_MARGIN),
            "n_below_with_split_labels": sum(
                1 for s in signals
                if s["margin"] < MIN_MARGIN and s["label_agreement"] < 1.0
                and s["evidence_count"] >= 2),
            "insensitivity_interval": gap_around(sorted(set(margins)), MIN_MARGIN),
        },
        "max_year_gap": {
            "legacy_value": MAX_YEAR_GAP,
            "procedure": "코퍼스 연도 폭에서 잡았다 — 데이터가 고르지 않았다",
            "observed_gap_range": [min(gaps), max(gaps)] if gaps else None,
            "n_with_gap": len(gaps),
            "n_exceeding": sum(1 for g in gaps if g > MAX_YEAR_GAP),
            "corpus_year_span": sorted({int(y) for y in
                                        {c.get("source", "")[:4] for c in cases}
                                        if y.isdigit()}),
        },
        "keyword_vocabulary": {
            "procedure": "사람이 legacy dev 85건을 읽고 뽑았다 — 스크립트가 없다",
            "on_legacy_dev": keyword_vocabulary(legacy_dev),
            "on_clean_dev": keyword_vocabulary(dev),
        },
        # 이 자산은 **LLM 프롬프트로 들어간다**(`app/agents/classifier.py`).
        # B-2b 가 건드리는 유일한 learned legacy 자산이므로 여기서 대조한다.
        "base_rates": {
            "shipped_file": str(EVAL / "dev_base_rates.json"),
            "shipped": _base_rate_summary(json.loads(
                (EVAL / "dev_base_rates.json").read_text(encoding="utf-8"))),
            "clean_dev_would_be": _base_rate_summary(base_rates_compute(dev)),
            "existing_guard": "probes.base_rates_come_from_dev_only 은 source=='dev' 만 본다 "
                              "— 어느 dev 인지는 보지 않는다",
        },
        "rule_atoms": {
            "clean": atom_kinds(rules),
            "legacy": atom_kinds(legacy_rules),
            "router_readable_kinds": ["ngram", "length"],
        },
        "provenance_split": {},
    }

    clean_dir = RESULTS / "clean"
    for name in ("keyword", "neighbor", "agent_router"):
        path = clean_dir / f"pred_{name}.jsonl"
        if path.exists():
            report["provenance_split"][name] = provenance_split(
                test, load_jsonl(path), legacy_dev)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    dt = report["doubt_trust"]
    print(f"DOUBT/TRUST {DOUBT}/{TRUST} — clean dev 에서도 허용되는가: "
          f"{dt['legacy_pair_admissible_on_clean_dev']}")
    print(f"  허용 쌍  legacy dev {dt['legacy_dev_admissible']['n_admissible']} · "
          f"clean dev {dt['clean_dev_admissible']['n_admissible']} "
          f"/ {dt['clean_dev_admissible']['n_candidates']}  <- 절차가 값을 좁히지 못한다")
    shape = dt["clean_dev_similarity_shape"]
    print(f"  clean dev 유사도  문턱 아래 {shape['below_doubt']} · 가운데 "
          f"{shape['middle']} · 믿음 위 {shape['above_trust']}")
    for name in ("clean_dev_gap_at_doubt", "clean_dev_gap_at_trust"):
        g = dt[name]
        print(f"  {name}: ({g['lower']}, {g['upper']})")
    og = report["observed_test_signal_gaps"]
    print(f"  clean test 신호 기준 무감구간  DOUBT ({og['doubt']['lower']}, "
          f"{og['doubt']['upper']}) · TRUST ({og['trust']['lower']}, "
          f"{og['trust']['upper']})")
    rc = report["risk_ceiling"]
    print(f"\nRISK_CEILING {rc['legacy_value']} (legacy 산 {rc['legacy_derived']} · "
          f"clean 산 {rc['clean_derived']})")
    print(f"  관측된 trap_risk {rc['observed_trap_risk_values']} · "
          f"무감구간 ({rc['insensitivity_interval']['lower']}, "
          f"{rc['insensitivity_interval']['upper']})")
    ma = report["min_agreement"]
    print(f"\nMIN_AGREEMENT {ma['legacy_value']} · 관측값 {ma['observed_values']} · "
          f"무감구간 ({ma['insensitivity_interval']['lower']}, "
          f"{ma['insensitivity_interval']['upper']})")
    mm = report["min_margin"]
    print(f"MIN_MARGIN {mm['legacy_value']} · 미만 {mm['n_below']}건 · "
          f"R9 조건까지 맞는 것 {mm['n_below_with_split_labels']}건")
    my = report["max_year_gap"]
    print(f"MAX_YEAR_GAP {my['legacy_value']} · 관측 간극 {my['observed_gap_range']} · "
          f"초과 {my['n_exceeding']}건")
    br = report["base_rates"]
    print(f"\n기저율(프롬프트로 들어간다)  실린 것 n={br['shipped']['n']} "
          f"{br['shipped']['overall']}")
    print(f"                              clean dev n={br['clean_dev_would_be']['n']} "
          f"{br['clean_dev_would_be']['overall']}")
    ra = report["rule_atoms"]
    print(f"\n규칙 조건 종류  clean {ra['clean']['kinds']} · legacy {ra['legacy']['kinds']}")
    print(f"  Router 가 못 읽는 규칙  clean {ra['clean']['n_rules_unreadable']}개 · "
          f"legacy {ra['legacy']['n_rules_unreadable']}개")
    for name, cell in report["provenance_split"].items():
        print(f"\n{name}")
        for group, v in cell.items():
            acc = v["accuracy_on_answered"]
            print(f"  {group:<22} n {v['n']:>3} · 답변 {v['answered']:>3} · "
                  f"답한 것 정확도 {'—' if acc is None else f'{acc:.3f}'} "
                  f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print(f"\n-> {args.out}")
