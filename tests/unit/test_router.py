"""Router — 경로 선택. **fixture 를 먼저 쓰고 Router 를 거기 맞췄다.**

이 파일이 검사하는 것은 두 가지다.

    1. 극단 케이스에서 기대한 경로와 **기대한 줄**이 발화하는가
    2. 결정 표에 **죽은 줄이 없는가**

2번이 1번만큼 중요하다. 발화하지 않는 규칙은 규칙이 아니라 주석이다.
실제로 초안의 R4 는 R2 가 먼저 잡아 도달할 수 없었고, 이 검사를 쓰다 드러났다.
"""

from __future__ import annotations

import json
from pathlib import Path as FsPath

import pytest

from app.agents.calibration import band_of
from app.agents.fixtures import SCENARIOS
from app.agents.router import (
    ABSTAIN_FOR,
    MAX_YEAR_GAP,
    MIN_AGREEMENT,
    MIN_MARGIN,
    RISK_CEILING,
    route,
    signals_from,
)
from app.agents.state import Evidence, EvidenceKind, Path, RouterSignals
from app.core.paths import RESULTS
from app.domain.similarity import DOUBT, TRUST


def _risk(similarity: float) -> float:
    path = RESULTS / "trap_risk.json"
    if not path.exists():
        pytest.skip("보정표가 없습니다 — scripts/calibrate.py")
    table = json.loads(path.read_text(encoding="utf-8"))
    cell = table["by_band"][band_of(similarity)]
    return cell["risk"] if cell["risk"] is not None else table["overall_risk"]


def _evidence(scenario):
    precedents = [
        Evidence(id=f"prec:{p.source}#{p.serial}", kind=EvidenceKind.PRECEDENT,
                 label=p.label, score=p.similarity, rank=i, source=p.source,
                 serial=p.serial, year=p.year, retriever="L")
        for i, p in enumerate(sorted(scenario.precedents, key=lambda x: -x.similarity))
    ]
    rules = [
        Evidence(id=f"rule:e6#{i}", kind=EvidenceKind.RULE, label=label,
                 score=0.9, rank=i)
        for i, label in enumerate(scenario.rules_fire)
    ]
    return precedents, rules


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_extreme_cases_take_the_expected_path(scenario):
    """극단 케이스마다 **미리 적어 둔** 경로와 줄이 나오는가."""
    precedents, rules = _evidence(scenario)
    top = max((p.similarity for p in scenario.precedents), default=0.0)
    signals = signals_from(precedents, rules, _risk(top), scenario.year)
    got_path, got_rule = route(signals)

    assert got_path == scenario.expect_route, (
        f"{scenario.name}: 기대 {scenario.expect_route.value} · 실제 "
        f"{got_path.value}({got_rule}) — {scenario.what}"
    )
    assert got_rule == scenario.expect_rule, (
        f"{scenario.name}: 기대 {scenario.expect_rule} · 실제 {got_rule}"
    )
    if got_path == Path.ABSTAIN:
        assert got_rule in ABSTAIN_FOR, f"{got_rule} 에 기권 사유가 없습니다"


# ── 죽은 줄이 없는가 ──────────────────────────────────────────────
LIVE_RULES = {
    "R1": RouterSignals(rule_fired=2, rule_conflict=True),
    "R2": RouterSignals(),
    "R3": RouterSignals(evidence_count=0, rule_fired=1),
    "R5": RouterSignals(evidence_count=1, top_similarity=DOUBT + 0.01,
                        trap_risk=RISK_CEILING + 0.1, rule_fired=1),
    "R6": RouterSignals(evidence_count=3, top_similarity=TRUST + 0.1,
                        trap_risk=0.05, margin=0.2,
                        label_agreement=MIN_AGREEMENT - 0.1, source_diversity=2),
    "R7": RouterSignals(evidence_count=1, top_similarity=TRUST + 0.1, trap_risk=0.05,
                        margin=0.5, recency_gap=MAX_YEAR_GAP + 1, rule_fired=1),
    "R9": RouterSignals(evidence_count=2, top_similarity=TRUST + 0.1, trap_risk=0.05,
                        margin=MIN_MARGIN / 2, label_agreement=0.5,
                        source_diversity=1),
    "R8": RouterSignals(evidence_count=2, top_similarity=TRUST + 0.1, trap_risk=0.05,
                        margin=0.3, label_agreement=1.0, source_diversity=2),
    "R10": RouterSignals(evidence_count=1, top_similarity=TRUST - 0.1, trap_risk=0.05,
                         margin=0.3, label_agreement=1.0, source_diversity=1),
}


@pytest.mark.parametrize("rule,signals", sorted(LIVE_RULES.items()))
def test_every_rule_can_fire(rule, signals):
    """결정 표의 모든 줄이 실제로 발화할 수 있는가.

    초안의 R4("선례도 규칙도 없다")는 R2 가 먼저 잡아 **한 번도 발화할 수
    없었다.** 그것을 이 검사가 찾아냈고, 죽은 줄은 지웠다.
    """
    _, fired = route(signals)
    assert fired == rule, f"{rule} 을 노렸는데 {fired} 가 발화했습니다"


def test_decision_table_has_no_unreachable_rules():
    """소스에 적힌 줄 번호와 실제로 발화 가능한 줄이 같은가.

    첫 판은 `return` 뒤만 보는 정규식이었고, 삼항식 뒤쪽의
    `... else (Path.ABSTAIN, "R4")` 를 놓쳤다. **잡으려던 바로 그것을 놓친
    검사였다** — R4 를 되살려도 통과했다. 반환문 형태에 기대지 않고 결정
    표의 줄 번호 리터럴을 전부 센다.
    """
    import ast
    import re

    from app.agents import router

    text = FsPath(router.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    written = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "route"):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                if re.fullmatch(r"R\d+", inner.value):
                    written.add(inner.value)
    assert written, "route() 안에서 줄 번호를 하나도 찾지 못했습니다"
    assert written == set(LIVE_RULES), (
        f"소스에만 있는 줄 {sorted(written - set(LIVE_RULES))} · "
        f"검사에만 있는 줄 {sorted(set(LIVE_RULES) - written)}"
    )


def test_abstain_rules_all_have_a_reason():
    """기권으로 가는 모든 줄에 사유 코드가 붙어 있는가."""
    for _rule, signals in LIVE_RULES.items():
        path, fired = route(signals)
        if path == Path.ABSTAIN:
            assert fired in ABSTAIN_FOR, f"{fired} 로 기권했는데 사유가 없습니다"


def test_weak_single_precedent_is_not_followed_blindly():
    """약한 선례 하나를 무조건 따르지 않는가 — 이 프로젝트의 존재 이유.

    E5: 선례를 따르는 전략은 함정 구간에서 정확도 0.000 이다.
    """
    precedent = Evidence(id="prec:x#1", kind=EvidenceKind.PRECEDENT, label="비조치",
                         score=DOUBT + 0.01, rank=0, source="x", year=2024)
    rule = Evidence(id="rule:e6#0", kind=EvidenceKind.RULE, label="조치",
                    score=0.9, rank=0)
    signals = signals_from([precedent], [rule], _risk(DOUBT + 0.01), 2025)
    path, fired = route(signals)
    assert path != Path.PRECEDENT, (
        f"약한 선례({signals.top_similarity:.2f}, 위험 {signals.trap_risk:.2f})를 "
        f"그대로 따랐습니다 ({fired})"
    )
