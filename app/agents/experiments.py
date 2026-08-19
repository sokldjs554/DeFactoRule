"""E8~E11a — Agent Workflow 의 각 부분이 값어치를 하는가.

## 사전 등록한 것

| | 비교 | 묻는 것 |
|---|---|---|
| E8 | naive ↔ router | 검색 결과를 무조건 믿는 것 대비 나아지는가 |
| E9 | always-precedent ↔ router | 경로 선택 자체가 값어치가 있는가 |
| E10 | router-noabstain ↔ router | 기권이 위험을 낮추는가 |
| E11a | router-novalidate ↔ router | 결정론 검증 다섯이 값어치가 있는가 |

**주 종점은 함정 구간(TRAP) 정확도**다. 설계서에 그렇게 적어 두었다.

## 주 종점이 이 실험에 맞지 않는다는 것을 여기 적어 둔다

선례를 그대로 따르는 전략의 함정 구간 정확도는 정의상 0 이다. 그런데 Router 가
함정에서 **기권**하면 그것도 정확도 0 이다. 맞히지 않았으니까. 그러면 "틀리지
않게 된 것" 이 수치에 전혀 안 나타난다.

주 종점을 바꾸지 않는다 — 결과를 보고 지표를 옮기는 것이 조작이다. 대신
함정 구간을 **셋으로 갈라** 함께 보고한다.

    맞힘 / 틀림 / 기권

`naive` 는 정의상 15건 전부 '틀림' 이다. Router 가 그중 몇 건을 '기권' 으로
옮겼는지가 이 워크플로가 실제로 한 일이고, 그것은 위 분해에만 나타난다.
"""

from __future__ import annotations

import json
from collections import Counter

from app.agents.workflow import VARIANTS, Workflow
from app.domain.labels import NON_ACTIONS
from app.domain.similarity import DOUBT
from app.evaluation.confusable import idf_table, nearest, partition
from app.evaluation.metrics import macro_f1, wilson_interval

COMPARISONS = [
    ("E8", "naive", "router", "검색 결과를 무조건 믿는 것 대비"),
    ("E9", "always-precedent", "router", "경로 선택 자체의 값어치"),
    ("E10", "router-noabstain", "router", "기권의 값어치"),
    ("E11a", "router-novalidate", "router", "결정론 검증의 값어치"),
]


def run_variant(name: str, retriever_factory, precedents, rules, risk, rows,
                corpus, fallback: str) -> list:
    """한 변형을 test 전체에 돌린다. 검색기는 변형마다 새로 적합시킨다."""
    flow = Workflow(retriever_factory().fit(precedents, corpus), precedents, rules,
                    risk, floor=DOUBT, policy=VARIANTS[name], fallback=fallback)
    return [flow.run(row) for row in rows]


def trap_keys(precedents: list[dict], rows: list[dict], corpus: list[str]) -> set:
    """E5 와 **같은 방식으로** 함정 구간을 정한다. 다시 구현하지 않는다."""
    from app.core.io import key_of

    links = nearest(rows, precedents, idf_table(corpus))
    return {key_of(link["row"]) for link in partition(links, DOUBT)["trap"]}


def decompose(rows, states, keys: set) -> dict:
    """주어진 구간에서 맞힘 / 틀림 / 기권 을 센다."""
    from app.core.io import key_of

    counts = Counter()
    for row, state in zip(rows, states):
        if key_of(row) not in keys:
            continue
        if state.abstained:
            counts["기권"] += 1
        elif state.decision == row["label"]:
            counts["맞힘"] += 1
        else:
            counts["틀림"] += 1
    total = sum(counts.values())
    lo, hi = wilson_interval(counts["맞힘"], total)
    return {
        "n": total, "correct": counts["맞힘"], "wrong": counts["틀림"],
        "abstained": counts["기권"],
        "accuracy": counts["맞힘"] / total if total else None,
        "accuracy_ci95": [lo, hi],
        "error_rate": counts["틀림"] / total if total else None,
    }


def score(rows, states) -> dict:
    """답한 것만으로 매크로 F1 · 커버리지 · 근거 없는 주장 비율."""
    answered = [(r, s) for r, s in zip(rows, states) if not s.abstained]
    pairs = [(r["label"], s.decision) for r, s in answered]
    macro = macro_f1(pairs, NON_ACTIONS)[0] if pairs else 0.0
    unsupported = sum(
        1 for _r, s in answered
        if not any(e.label == s.decision for e in s.all_evidence()))
    return {
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "answered": len(answered),
        "macro_f1_on_answered": macro,
        "accuracy_on_answered": (
            sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else None),
        "unsupported_claim_rate": unsupported / len(answered) if answered else 0.0,
        "abstention_reasons": dict(Counter(
            s.abstention_reason.value for _r, s in zip(rows, states)
            if s.abstention_reason)),
        "routes": dict(Counter(s.route.value for s in states if s.route)),
    }


def abstention_accuracy(rows, states) -> dict:
    """기권한 건 중, **답했다면 틀렸을** 비율.

    `provisional`("굳이 답한다면 무엇인가")이 정답과 다르면 그 기권은 정당했다.
    이 수치를 커버리지 없이 보면 안 된다 — 전부 기권하면 저절로 올라간다.
    그래서 반환값에 기권 건수를 함께 담고, 보고 표는 커버리지를 옆에 둔다.
    """
    justified = considered = 0
    for row, state in zip(rows, states):
        if not state.abstained:
            continue
        considered += 1
        if state.provisional != row["label"]:
            justified += 1
    lo, hi = wilson_interval(justified, considered)
    return {"abstained": considered, "justified": justified,
            "rate": justified / considered if considered else None,
            "ci95": [lo, hi]}


def report(rows, results: dict, trap: set, corpus_note: str) -> dict:
    """모든 변형을 같은 표본 위에서 정리한다."""
    out = {"n_test": len(rows), "trap_n": len(trap), "corpus": corpus_note,
           "primary_endpoint": "TRAP 정확도 (사전 등록)", "variants": {}}
    for name, states in results.items():
        out["variants"][name] = {
            **score(rows, states),
            "abstention_accuracy": abstention_accuracy(rows, states),
            "trap": decompose(rows, states, trap),
        }
    return out


def dump(path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
