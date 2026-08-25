from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.core.io import load_jsonl
from app.domain.similarity import SIMILARITY_FLOOR
from app.evaluation.confusable import cosine, idf_table, weighted_vector

DEV = Path("data/eval/nonaction_dev_clean.jsonl")
TEST = Path("data/eval/nonaction_test_clean.jsonl")
CORPUS = Path("data/processed/cases_nonaction.jsonl")
TOP_K = 5
B2B = {"230032", "240006", "230067", "240022", "230041"}


def serial_time(row: dict) -> tuple[int, int]:
    s = str(row["serial"])
    return 2000 + int(s[:2]), int(s[2:])


def eligible(policy: str, p: dict, r: dict) -> bool:
    py, ps = serial_time(p)
    ry, rs = serial_time(r)
    if policy == "none":
        return True
    if policy == "serial":
        return (py, ps) < (ry, rs)
    if policy == "strict":
        return py < ry
    raise ValueError(policy)


def test_clean_temporal_c1_probe() -> None:
    dev = [r for r in load_jsonl(DEV) if r.get("label")]
    test = [r for r in load_jsonl(TEST) if r.get("label")]
    corpus = load_jsonl(CORPUS)
    texts = [(c["fields"].get("요청대상행위") or "") for c in corpus]
    idf = idf_table(texts)
    dvecs = [weighted_vector(r["request"], idf) for r in dev]

    policies = ("none", "serial", "strict")
    ranked: dict[str, list[list[tuple[int, float]]]] = {p: [] for p in policies}
    eligible_counts: dict[str, list[int]] = {p: [] for p in policies}
    discarded_pairs = Counter()
    top5_removed = Counter()
    top5_removed_above = Counter()

    for row in test:
        q = weighted_vector(row["request"], idf)
        all_scores = [(i, cosine(q, dvecs[i])) for i in range(len(dev))]
        nofilter = sorted(all_scores, key=lambda x: (-x[1], x[0]))[:TOP_K]
        for policy in ranked:
            allowed = [(i, s) for i, s in all_scores if eligible(policy, dev[i], row)]
            eligible_counts[policy].append(len(allowed))
            ranked[policy].append(sorted(allowed, key=lambda x: (-x[1], x[0]))[:TOP_K])
            if policy != "none":
                discarded_pairs[policy] += len(dev) - len(allowed)
                removed = [
                    (i, s)
                    for i, s in nofilter
                    if not eligible(policy, dev[i], row)
                ]
                top5_removed[policy] += len(removed)
                top5_removed_above[policy] += sum(
                    s >= SIMILARITY_FLOOR for _, s in removed
                )

    def summarize(policy: str) -> dict:
        agree = trap = unanchored = 0
        anchored_by_year = Counter()
        total_by_year = Counter()
        zero_eligible = 0
        for row, top, nelig in zip(test, ranked[policy], eligible_counts[policy]):
            year, _ = serial_time(row)
            total_by_year[year] += 1
            if nelig == 0:
                zero_eligible += 1
            if not top or top[0][1] < SIMILARITY_FLOOR:
                unanchored += 1
                continue
            anchored_by_year[year] += 1
            if dev[top[0][0]]["label"] == row["label"]:
                agree += 1
            else:
                trap += 1
        counts = sorted(eligible_counts[policy])
        median = counts[len(counts) // 2] if counts else 0
        return {
            "eligible_pool_median": median,
            "anchored": agree + trap,
            "agree": agree,
            "trap": trap,
            "unanchored": unanchored,
            "zero_eligible": zero_eligible,
            "year_coverage": {
                str(y): {"anchored": anchored_by_year[y], "n": total_by_year[y]}
                for y in sorted(total_by_year)
            },
        }

    summary = {p: summarize(p) for p in ranked}
    assert summary["none"]["anchored"] == 60
    assert summary["none"]["agree"] == 48
    assert summary["none"]["trap"] == 12
    assert summary["none"]["unanchored"] == 108

    nofilter_anchored = []
    transitions = {"serial": Counter(), "strict": Counter()}
    future_top1 = Counter()
    future_top5_any = Counter()
    future_top5_above = Counter()

    for idx, row in enumerate(test):
        nf = ranked["none"][idx]
        if nf:
            for policy in ("serial", "strict"):
                if not eligible(policy, dev[nf[0][0]], row):
                    future_top1[policy] += 1
                removed = [
                    (i, s)
                    for i, s in nf
                    if not eligible(policy, dev[i], row)
                ]
                if removed:
                    future_top5_any[policy] += 1
                if any(s >= SIMILARITY_FLOOR for _, s in removed):
                    future_top5_above[policy] += 1
        if nf and nf[0][1] >= SIMILARITY_FLOOR:
            nofilter_anchored.append(idx)
            old_i = nf[0][0]
            for policy in ("serial", "strict"):
                now = ranked[policy][idx]
                if not now or now[0][1] < SIMILARITY_FLOOR:
                    transitions[policy]["lost_all_above_floor"] += 1
                elif now[0][0] == old_i:
                    transitions[policy]["unchanged"] += 1
                else:
                    transitions[policy]["top1_changed"] += 1

    b2b = {}
    for row_idx, row in enumerate(test):
        cid = str(row["serial"])
        if cid not in B2B:
            continue
        slot = {}
        for policy in ("none", "serial", "strict"):
            top = ranked[policy][row_idx]
            if not top:
                slot[policy] = None
                continue
            i, score = top[0]
            slot[policy] = {
                "precedent": str(dev[i]["serial"]),
                "label": dev[i]["label"],
                "score": round(score, 4),
                "above_floor": score >= SIMILARITY_FLOOR,
            }
        b2b[cid] = slot

    result = {
        "n_dev": len(dev),
        "n_test": len(test),
        "floor": SIMILARITY_FLOOR,
        "top_k": TOP_K,
        "policy_summary": summary,
        "discarded_pairs": dict(discarded_pairs),
        "nofilter_top5_removed": dict(top5_removed),
        "nofilter_top5_removed_above_floor": dict(top5_removed_above),
        "future_in_nofilter_top5_case_count": dict(future_top5_any),
        "future_above_floor_in_nofilter_top5_case_count": dict(future_top5_above),
        "future_nofilter_top1_case_count": dict(future_top1),
        "nofilter_anchored_n": len(nofilter_anchored),
        "anchored_top1_transition": {p: dict(c) for p, c in transitions.items()},
        "b2b": b2b,
    }

    payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
    raise AssertionError("C1_TEMPORAL_RESULT=" + payload)
