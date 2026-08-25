from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.agents.experiments import run_variant
from app.agents.temporal_calibration import calibrate_temporal
from app.core.io import load_jsonl
from app.evaluation.confusable import idf_table
from app.retrieval.lexical import LexicalRetriever

DEV = Path("data/eval/nonaction_dev_clean.jsonl")
TEST = Path("data/eval/nonaction_test_clean.jsonl")
CORPUS = Path("data/processed/cases_nonaction.jsonl")
RULES = Path("experiments/results/e6_rules_clean.json")


def _score(rows: list[dict], states: list) -> dict:
    answered = [(r, s) for r, s in zip(rows, states) if not s.abstained]
    correct = sum(r["label"] == s.decision for r, s in answered)
    wrong = len(answered) - correct
    return {
        "answered": len(answered),
        "abstained": len(rows) - len(answered),
        "correct": correct,
        "wrong": wrong,
        "coverage": len(answered) / len(rows),
        "answered_accuracy": correct / len(answered) if answered else None,
    }


def test_c3_temporal_calibration_probe() -> None:
    dev = [r for r in load_jsonl(DEV) if r.get("label")]
    test = [r for r in load_jsonl(TEST) if r.get("label")]
    cases = load_jsonl(CORPUS)
    corpus = [
        c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
        for c in cases
    ]
    corpus = [t for t in corpus if t]
    rules = json.loads(RULES.read_text(encoding="utf-8"))["rules"]
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]

    risk = calibrate_temporal(dev, idf_table(corpus), policy="serial")
    states = run_variant(
        "router-temporal", LexicalRetriever, dev, rules, risk, test, corpus, fallback
    )

    payload = {
        "calibration": risk,
        "router_temporal_recalibrated": _score(test, states),
    }
    raise AssertionError(
        "C3_TEMPORAL_CALIBRATION_RESULT="
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
