"""C-5 final deterministic freeze.

No LLM/API calls. The already-frozen clean E6 asset is projected onto conditions the live
request-only Router can evaluate, then checked under the clean split, T-serial eligibility
and temporal-matched risk table. No replacement rules are learned from test feedback.
"""

from __future__ import annotations

import json
from collections import Counter

from app.agents.workflow import VARIANTS, Workflow
from app.core.io import load_jsonl
from app.core.paths import EVAL, PROCESSED, RESULTS
from app.evaluation.clean_profile import fires_as_induced, fires_as_router, rule_transfer
from app.retrieval.lexical import LexicalRetriever
from app.rules.runtime_induction import (
    RUNTIME_ATOM_KINDS,
    project_runtime_asset,
    validate_runtime_asset,
)

C3_ANCHOR = {"n": 168, "answered": 76, "abstained": 92, "correct": 63, "wrong": 13}
RULES_OUT = RESULTS / "clean" / "e6_rules_clean_runtime.json"
RESULT_OUT = RESULTS / "clean" / "final_clean_temporal.json"


def _load_world():
    dev = [r for r in load_jsonl(EVAL / "nonaction_dev_clean.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test_clean.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [
        c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
        for c in cases
    ]
    corpus = [text for text in corpus if text]
    frozen_clean = json.loads((RESULTS / "e6_rules_clean.json").read_text(encoding="utf-8"))
    risk = json.loads(
        (RESULTS / "trap_risk_clean_temporal.json").read_text(encoding="utf-8")
    )
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]
    return dev, test, corpus, frozen_clean, risk, fallback


def _run(dev, test, corpus, rules, risk, fallback):
    flow = Workflow(
        LexicalRetriever().fit(dev, corpus),
        dev,
        rules,
        risk,
        policy=VARIANTS["router-temporal"],
        fallback=fallback,
    )
    return [flow.run(row) for row in test]


def _outcome(row: dict, state) -> str:
    if state.abstained:
        return "abstain"
    return "correct" if state.decision == row["label"] else "wrong"


def _profile(rows: list[dict], states: list) -> dict:
    answered = [
        (row, state) for row, state in zip(rows, states) if not state.abstained
    ]
    correct = sum(1 for row, state in answered if state.decision == row["label"])
    wrong = len(answered) - correct
    route_counts = Counter(state.route_reason or "—" for state in states)
    path_counts = Counter(state.route.value if state.route else "—" for state in states)
    abstention_counts = Counter(
        state.abstention_reason.value if state.abstention_reason else "—"
        for state in states
        if state.abstained
    )
    rule_fires = Counter(
        evidence.id for state in states for evidence in state.rule_evidence
    )
    return {
        "n": len(rows),
        "answered": len(answered),
        "abstained": len(rows) - len(answered),
        "correct": correct,
        "wrong": wrong,
        "coverage": len(answered) / len(rows) if rows else None,
        "accuracy_on_answered": correct / len(answered) if answered else None,
        "routes": dict(sorted(route_counts.items())),
        "paths": dict(sorted(path_counts.items())),
        "abstention_reasons": dict(sorted(abstention_counts.items())),
        "rule_fires": dict(sorted(rule_fires.items())),
    }


def _state_signature(state) -> tuple:
    """Fields that affect observable Router behaviour and its audit trace."""
    return (
        state.abstained,
        state.decision,
        state.provisional,
        state.route.value if state.route else None,
        state.route_reason,
        state.abstention_reason.value if state.abstention_reason else None,
        tuple(e.id for e in state.rule_evidence),
        tuple(state.evidence_used),
    )


def _transitions(rows: list[dict], before: list, after: list) -> dict:
    counts = Counter()
    changed = []
    for row, old, new in zip(rows, before, after):
        a, b = _outcome(row, old), _outcome(row, new)
        counts[f"{a}->{b}"] += 1
        if _state_signature(old) != _state_signature(new):
            changed.append(
                {
                    "serial": str(row["serial"]),
                    "gold": row["label"],
                    "before_outcome": a,
                    "after_outcome": b,
                    "before_route": old.route_reason,
                    "after_route": new.route_reason,
                    "before_decision": old.decision,
                    "after_decision": new.decision,
                }
            )
    return {"counts": dict(sorted(counts.items())), "changed": changed}


def build_final_freeze(write: bool = True) -> tuple[dict, dict]:
    dev, test, corpus, frozen_clean, risk, fallback = _load_world()

    runtime_asset = project_runtime_asset(frozen_clean)
    validate_runtime_asset(runtime_asset)
    atom_kinds = {
        atom["kind"]
        for rule in runtime_asset["rules"]
        for atom in rule["atoms"]
    }
    if not atom_kinds <= RUNTIME_ATOM_KINDS:
        raise AssertionError(f"unsupported atom kinds survived: {sorted(atom_kinds)}")

    default = runtime_asset["default_label"]
    induced_transfer = rule_transfer(
        runtime_asset["rules"], default, test, matcher=fires_as_induced
    )
    router_transfer = rule_transfer(
        runtime_asset["rules"], default, test, matcher=fires_as_router
    )
    if induced_transfer != router_transfer:
        raise AssertionError("projected E6 and Router matcher disagree")

    before = _run(dev, test, corpus, frozen_clean["rules"], risk, fallback)
    after = _run(dev, test, corpus, runtime_asset["rules"], risk, fallback)
    before_profile = _profile(test, before)
    anchor_view = {key: before_profile[key] for key in C3_ANCHOR}
    if anchor_view != C3_ANCHOR:
        raise AssertionError(f"C-3 anchor drift: expected={C3_ANCHOR} actual={anchor_view}")

    transitions = _transitions(test, before, after)
    if transitions["changed"]:
        raise AssertionError(
            "capability projection changed Router behaviour: "
            f"{transitions['changed'][:3]}"
        )

    final_profile = _profile(test, after)
    final_anchor_view = {key: final_profile[key] for key in C3_ANCHOR}
    if final_anchor_view != C3_ANCHOR:
        raise AssertionError(
            f"final profile drift: expected={C3_ANCHOR} actual={final_anchor_view}"
        )

    payload = {
        "phase": "C-5 final deterministic freeze",
        "api_calls": 0,
        "profile": {
            "split": "clean-group",
            "temporal_policy": "serial proxy",
            "risk_asset": "trap_risk_clean_temporal.json",
            "rule_asset": "e6_rules_clean_runtime.json",
            "runtime_rule_atom_kinds": sorted(RUNTIME_ATOM_KINDS),
            "rule_contract": "capability projection of frozen clean E6; no re-induction",
            "s5_mode": "fail-closed qualitative safety veto; not applied to aggregate metrics",
        },
        "runtime_e6": {
            "source_rule_count": len(frozen_clean["rules"]),
            "final_rule_count": len(runtime_asset["rules"]),
            "dropped_rules": runtime_asset["dropped_rules"],
            "default_label": default,
            "atom_kinds_present": sorted(atom_kinds),
            "transfer": router_transfer,
            "behaviorally_equivalent_to_c3": True,
        },
        "c3_anchor_recomputed": before_profile,
        "final": final_profile,
        "delta": {
            "answered": final_profile["answered"] - before_profile["answered"],
            "abstained": final_profile["abstained"] - before_profile["abstained"],
            "correct": final_profile["correct"] - before_profile["correct"],
            "wrong": final_profile["wrong"] - before_profile["wrong"],
            "coverage": final_profile["coverage"] - before_profile["coverage"],
            "accuracy_on_answered": (
                final_profile["accuracy_on_answered"]
                - before_profile["accuracy_on_answered"]
            ),
        },
        "transitions": transitions,
        "limitations": [
            "T-serial is a chronology proxy, not ground-truth decision time.",
            "S5 audit is qualitative and fail-closed; safe recovery was not established.",
            "Action-label precedent evidence remains sparse.",
            "DOUBT/TRUST are inherited thresholds supported, not uniquely optimized, on clean dev.",
            "A text-only E6 re-induction was evaluated as a diagnostic and not adopted; production uses a capability projection of the frozen clean asset.",
        ],
    }

    if write:
        RULES_OUT.parent.mkdir(parents=True, exist_ok=True)
        RULES_OUT.write_text(
            json.dumps(runtime_asset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        RESULT_OUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return runtime_asset, payload


def main() -> None:
    runtime_asset, payload = build_final_freeze(write=True)
    final = payload["final"]
    print("C-5 final deterministic freeze · API 0회")
    print(
        f"runtime E6 {len(runtime_asset['rules'])} rules · "
        f"dropped={len(runtime_asset['dropped_rules'])} · "
        f"atom kinds={payload['runtime_e6']['atom_kinds_present']}"
    )
    print(
        f"final: answered {final['answered']} · abstained {final['abstained']} · "
        f"correct {final['correct']} · wrong {final['wrong']}"
    )
    print(
        f"coverage {final['coverage']:.4f} · "
        f"answered accuracy {final['accuracy_on_answered']:.4f}"
    )
    print(f"delta vs C-3: {payload['delta']}")
    print(f"transitions: {payload['transitions']['counts']}")
    print(f"-> {RULES_OUT}")
    print(f"-> {RESULT_OUT}")


if __name__ == "__main__":
    main()
