from __future__ import annotations

import json
from collections import Counter

from app.agents.applicability import opposing_evidence
from app.agents.workflow import VARIANTS, Workflow
from app.core.io import load_jsonl
from app.core.paths import EVAL, PROCESSED, RESULTS
from app.domain.similarity import DOUBT
from app.retrieval.lexical import LexicalRetriever


def test_c4_scope_probe() -> None:
    dev = [r for r in load_jsonl(EVAL / "nonaction_dev_clean.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test_clean.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [
        c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
        for c in cases
    ]
    corpus = [text for text in corpus if text]
    rules = json.loads(
        (RESULTS / "e6_rules_clean.json").read_text(encoding="utf-8")
    )["rules"]
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

    scope = []
    for row in test:
        state = flow.run(row)
        if not (state.abstained and state.precedent_score >= DOUBT):
            continue
        top = next(e for e in state.retrieved_evidence if e.rank == 0)
        scope.append(
            {
                "serial": str(row["serial"]),
                "gold": row["label"],
                "route": state.route_reason,
                "reason": state.abstention_reason.value,
                "top": str(top.serial),
                "top_label": top.label,
                "score": round(top.score, 4),
                "opposing": len(opposing_evidence(state)),
                "provisional": state.provisional,
                "top_correct": top.label == row["label"],
            }
        )

    risky = [x for x in scope if not x["top_correct"] and x["opposing"] == 0]
    payload = {"n_scope": len(scope), "risky_no_opposition": risky, "scope": scope}
    raise AssertionError(
        "C4_TEMPORAL_SCOPE=" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
