from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.agents.experiments import run_variant
from app.core.io import load_jsonl
from app.retrieval.lexical import LexicalRetriever

DEV = Path("data/eval/nonaction_dev_clean.jsonl")
TEST = Path("data/eval/nonaction_test_clean.jsonl")
CORPUS = Path("data/processed/cases_nonaction.jsonl")
RULES = Path("experiments/results/e6_rules_clean.json")
RISK = Path("experiments/results/trap_risk_clean.json")
B2B = {"230032", "240006", "230067", "240022", "230041"}


def _score(rows: list[dict], states: list) -> dict:
    answered = [(r, s) for r, s in zip(rows, states) if not s.abstained]
    correct = sum(r["label"] == s.decision for r, s in answered)
    wrong = len(answered) - correct
    return {
        "answered": len(answered),
        "coverage": len(answered) / len(rows),
        "correct": correct,
        "wrong": wrong,
        "answered_accuracy": correct / len(answered) if answered else None,
        "abstained": len(rows) - len(answered),
        "routes": dict(Counter(s.route.value for s in states)),
        "abstain_reasons": dict(
            Counter(
                s.abstention_reason.value
                for s in states
                if s.abstention_reason is not None
            )
        ),
    }


def test_c2_clean_temporal_probe() -> None:
    dev = [r for r in load_jsonl(DEV) if r.get("label")]
    test = [r for r in load_jsonl(TEST) if r.get("label")]
    cases = load_jsonl(CORPUS)
    corpus = [
        c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
        for c in cases
    ]
    corpus = [t for t in corpus if t]
    rules_payload = json.loads(RULES.read_text(encoding="utf-8"))
    rules = rules_payload["rules"]
    risk = json.loads(RISK.read_text(encoding="utf-8"))
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]

    baseline = run_variant(
        "router", LexicalRetriever, dev, rules, risk, test, corpus, fallback
    )
    temporal = run_variant(
        "router-temporal", LexicalRetriever, dev, rules, risk, test, corpus, fallback
    )

    changed = []
    b2b = {}
    for row, before, after in zip(test, baseline, temporal):
        serial = str(row["serial"])
        before_top = before.retrieved_evidence[0] if before.retrieved_evidence else None
        after_top = after.retrieved_evidence[0] if after.retrieved_evidence else None
        snapshot = {
            "gold": row["label"],
            "before_top": (
                [before_top.serial, before_top.label, round(before_top.score, 4)]
                if before_top
                else None
            ),
            "after_top": (
                [after_top.serial, after_top.label, round(after_top.score, 4)]
                if after_top
                else None
            ),
            "before": [before.route.value, before.decision, before.abstained],
            "after": [after.route.value, after.decision, after.abstained],
        }
        if snapshot["before_top"] != snapshot["after_top"] or snapshot["before"] != snapshot["after"]:
            changed.append({"serial": serial, **snapshot})
        if serial in B2B:
            b2b[serial] = snapshot

    payload = {
        "router": _score(test, baseline),
        "router_temporal": _score(test, temporal),
        "changed_cases": len(changed),
        "changed": changed,
        "b2b": b2b,
    }
    raise AssertionError(
        "C2_TEMPORAL_RESULT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
