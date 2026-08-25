"""Agent Workflow — 노드를 잇고 **지나온 길을 남긴다.**

검색 → 규칙 매칭 → 경로 선택 → 결정 → 검증 → 기권 게이트를 잇고,
각 단계가 무엇을 보고 무엇을 했는지 `execution_trace` 에 남긴다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.calibration import band_of
from app.agents.router import decide, route, signals_from
from app.agents.state import (
    AbstentionReason,
    AgentState,
    Evidence,
    EvidenceKind,
    Path,
)
from app.agents.validator import evidence_sources, validate
from app.core.audit import Discards
from app.domain.similarity import DOUBT
from app.domain.temporal import TemporalPolicy, eligible_indices
from app.retrieval.lexical import shared_span

TOP_K = 5
YEAR = re.compile(r"(20\d{2})")


def year_of(source: str | None) -> int | None:
    """출처 이름에서 연도를 읽는다. 없으면 None — 없는 값을 지어내지 않는다."""
    found = YEAR.search(source or "")
    return int(found.group(1)) if found else None


HIGH_MARGIN = 0.30
MEDIUM_MARGIN = 0.10


@dataclass(frozen=True)
class Policy:
    """워크플로 변형의 스위치. 과거 실험과 새 정책을 같은 코드에서 보존한다."""

    name: str
    use_router: bool = True
    use_rules: bool = True
    allow_abstain: bool = True
    run_validator: bool = True
    respect_floor: bool = True
    top_k: int = TOP_K
    temporal_policy: TemporalPolicy = "none"


NAIVE = Policy(
    "naive",
    use_router=False,
    use_rules=False,
    allow_abstain=False,
    run_validator=False,
    respect_floor=False,
    top_k=1,
)
ALWAYS_PRECEDENT = Policy("always-precedent", use_router=False, use_rules=False)
ROUTER = Policy("router")
ROUTER_TEMPORAL = Policy("router-temporal", temporal_policy="serial")
ROUTER_NO_ABSTAIN = Policy("router-noabstain", allow_abstain=False)
ROUTER_NO_VALIDATE = Policy("router-novalidate", run_validator=False)

VARIANTS = {
    p.name: p
    for p in (
        NAIVE,
        ALWAYS_PRECEDENT,
        ROUTER,
        ROUTER_TEMPORAL,
        ROUTER_NO_ABSTAIN,
        ROUTER_NO_VALIDATE,
    )
}


class Workflow:
    """요청 하나를 처리한다. 선례 풀과 규칙은 생성 시점에 고정된다."""

    def __init__(
        self,
        retriever,
        precedents: list[dict],
        rules: list[dict],
        risk_table: dict,
        floor: float = DOUBT,
        policy: Policy = ROUTER,
        fallback: str = "비조치",
    ) -> None:
        self.retriever = retriever
        self.precedents = precedents
        self.rules = rules if policy.use_rules else []
        self.risk_table = risk_table
        self.floor = floor if policy.respect_floor else 0.0
        self.policy = policy
        self.fallback = fallback
        self.discards = Discards("workflow")

    def _retrieve(self, state: AgentState, row: dict) -> None:
        candidates = None
        if self.policy.temporal_policy != "none":
            candidates = eligible_indices(
                self.precedents,
                row,
                self.policy.temporal_policy,
            )

        hits = self.retriever.search(
            state.request,
            self.policy.top_k,
            candidate_indices=candidates,
        )
        for rank, (index, score) in enumerate(hits):
            source = self.precedents[index]
            state.retrieved_evidence.append(
                Evidence(
                    id=f"prec:{source.get('source')}#{source.get('serial')}",
                    kind=EvidenceKind.PRECEDENT,
                    label=source["label"],
                    score=score,
                    rank=rank,
                    source=source.get("source"),
                    serial=str(source.get("serial")),
                    year=year_of(source.get("source")),
                    retriever=self.retriever.name,
                    quote=shared_span(state.request, source.get("request", "")),
                )
            )
        top = hits[0][1] if hits else 0.0
        state.precedent_score = top
        state.step(
            "retrieve",
            f"선례 {len(hits)}건",
            top_similarity=round(top, 4),
            retriever=self.retriever.name,
            temporal_policy=self.policy.temporal_policy,
            eligible_candidates=(len(candidates) if candidates is not None else len(self.precedents)),
        )

    def _match_rules(self, state: AgentState) -> None:
        for rule in self.rules:
            if rule_matches(rule, state.request):
                state.rule_evidence.append(
                    Evidence(
                        id=f"rule:e6#{rule['order']}",
                        kind=EvidenceKind.RULE,
                        label=rule["label"],
                        score=rule["dev_precision"],
                        rank=rule["order"],
                        source="e6_rules",
                        quote=rule.get("description"),
                    )
                )
        state.step(
            "rules",
            f"규칙 {len(state.rule_evidence)}건 발화",
            fired=[e.id for e in state.rule_evidence],
        )

    def _route(self, state: AgentState) -> None:
        cell = self.risk_table["by_band"].get(band_of(state.precedent_score)) or {}
        risk = cell.get("risk")
        if risk is None:
            risk = self.risk_table["overall_risk"]
        state.signals = signals_from(
            state.retrieved_evidence,
            state.rule_evidence,
            risk,
            request_year=year_of(state.request_key[0]),
        )
        if self.policy.use_router:
            state.route, state.route_reason = route(state.signals)
        else:
            usable = [e for e in state.retrieved_evidence if e.score >= self.floor]
            state.route = Path.PRECEDENT if usable else Path.ABSTAIN
            state.route_reason = "always-A" if usable else "no-evidence"
        state.step(
            "route",
            f"{state.route.value} ({state.route_reason})",
            trap_risk=round(risk, 4),
            top_similarity=round(state.signals.top_similarity, 4),
            rule_fired=state.signals.rule_fired,
        )

    def _best_guess(self, state: AgentState) -> str:
        """굳이 답한다면 무엇인가. 평가에만 쓴다."""
        usable = (
            [e for e in state.retrieved_evidence if e.score >= self.floor]
            or list(state.retrieved_evidence)
            or list(state.rule_evidence)
        )
        label, _used = decide(usable)
        return label or self.fallback

    def _decide(self, state: AgentState) -> None:
        from app.agents.router import ABSTAIN_FOR

        if state.route == Path.ABSTAIN and self.policy.allow_abstain:
            state.abstain(
                ABSTAIN_FOR.get(state.route_reason, AbstentionReason.NO_EVIDENCE),
                provisional=self._best_guess(state),
            )
            return
        if state.route == Path.ABSTAIN:
            usable = (
                [e for e in state.retrieved_evidence if e.score >= self.floor]
                or list(state.rule_evidence)
            )
            label, used = decide(usable)
            state.decision = label or self.fallback
            state.provisional = state.decision
            state.evidence_used = used
            state.confidence = "low"
            state.step("decide", f"{state.decision} (기권 없음)", evidence=used)
            return
        if state.route == Path.PRECEDENT:
            usable = [e for e in state.retrieved_evidence if e.score >= self.floor]
        else:
            usable = list(state.rule_evidence)

        label, used = decide(usable)
        if label is None:
            state.abstain(
                AbstentionReason.NO_EVIDENCE,
                "고른 경로에 근거가 없다",
                provisional=self._best_guess(state),
            )
            return
        state.decision = label
        state.provisional = label
        state.evidence_used = used
        state.confidence = grade(state.signals.margin if state.signals else 0.0)
        state.step("decide", f"{label} ({state.confidence})", evidence=used)

    def run(self, row: dict) -> AgentState:
        state = AgentState(
            request=row["request"],
            request_key=(
                row["source"],
                row["page"],
                str(row["serial"]),
                row.get("pair_index", 1),
            ),
        )
        self._retrieve(state, row)
        self._match_rules(state)
        self._route(state)
        self._decide(state)
        if not state.abstained and self.policy.run_validator:
            dropped = validate(
                state,
                evidence_sources(state.all_evidence(), self.precedents),
            )
            for record in dropped.records():
                self.discards.drop(
                    {k: v for k, v in record.items() if k != "rejected_for"},
                    record["rejected_for"],
                )
        return state


def rule_matches(rule: dict, request: str) -> bool:
    """유도 규칙의 조건이 요청문에 걸리는가. 전부 결정론."""
    from app.rules.induction import length_bucket, squeeze

    text = squeeze(request)
    for atom in rule["atoms"]:
        if atom["kind"] == "ngram":
            if atom["value"] not in text:
                return False
        elif atom["kind"] == "length":
            if length_bucket(request) != atom["value"]:
                return False
        else:
            return False
    return True


def grade(margin: float) -> str:
    """근거의 여유로 신뢰도를 매긴다. 모델은 관여하지 않는다."""
    if margin >= HIGH_MARGIN:
        return "high"
    if margin >= MEDIUM_MARGIN:
        return "medium"
    return "low"
