"""Agent 한 번의 처리 전체를 담는 상태 — **이것만으로 재현된다.**

## 왜 상태를 형으로 두는가

이 워크플로의 값어치는 답이 아니라 **어떤 근거로 그 답에 이르렀는가**에 있다.
"판단 보류" 를 돌려줬다면 왜 보류했는지, "선례를 따랐다" 면 어느 선례를 왜
믿었는지가 남아야 한다. 그것이 남지 않으면 이 프로젝트는 다시 분류기가 된다.

그래서 상태를 dict 로 흘리지 않고 형으로 못 박는다. 필드 하나가 비어 있으면
그 자리에서 드러난다.

## 재현 가능성이 형에 들어 있다

`execution_trace` 에는 **결정에 영향을 준 것만** 넣는다. 소요 시간처럼 실행마다
달라지는 값은 넣지 않는다 — 넣으면 두 실행의 trace 를 비교할 수 없고, 회귀
테스트가 못 쓴다.

`route_reason` 은 결정 표의 줄 번호(`R1`~`R10`)를 그대로 담는다. "왜 이 경로를
골랐는가" 를 사람의 서술이 아니라 데이터가 답해야 한다.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Path(str, Enum):
    """Router 가 고르는 세 경로."""

    PRECEDENT = "A"   # 선례를 믿을 만하다
    RULE = "B"        # 선례를 못 믿거나 닿지 않는다 — 명시적 규칙으로
    ABSTAIN = "C"     # 근거가 없거나 충돌한다 — 판단하지 않는다


class EvidenceKind(str, Enum):
    PRECEDENT = "precedent"
    RULE = "rule"


class AbstentionReason(str, Enum):
    """왜 판단하지 않았는가. 문자열이 아니라 코드로 남긴다 — 세어야 하므로."""

    NO_EVIDENCE = "NO_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    SURFACE_ONLY = "SURFACE_ONLY"
    AMBIGUOUS_MARGIN = "AMBIGUOUS_MARGIN"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class Evidence(BaseModel):
    """판단에 쓸 수 있는 근거 하나."""

    id: str = Field(..., description="prec:{source}#{serial} | rule:e6#{order}")
    kind: EvidenceKind
    label: str = Field(..., description="이 근거가 가리키는 결론")
    score: float = Field(..., ge=0.0, le=1.0, description="선례=유사도 · 규칙=dev 정밀도")
    rank: int = Field(..., ge=0)
    source: Optional[str] = None
    serial: Optional[str] = None
    year: Optional[int] = None
    retriever: Optional[str] = Field(None, description="L | D | H · 규칙이면 None")
    quote: Optional[str] = Field(None, description="원문에서 그대로 딴 구절")


class RouterSignals(BaseModel):
    """경로를 고르는 데 쓰는 값. **전부 결정론이고 전부 기록된다.**"""

    evidence_count: int = 0
    top_similarity: float = 0.0
    margin: float = Field(0.0, description="1등 - 2등 유사도")
    label_agreement: float = Field(1.0, description="상위 근거 중 1등 라벨과 같은 비율")
    source_diversity: int = Field(0, description="상위 근거의 서로 다른 출처 수")
    recency_gap: Optional[int] = None
    rule_fired: int = 0
    rule_conflict: bool = False
    trap_risk: float = Field(0.0, ge=0.0, le=1.0,
                             description="선례를 따르면 틀릴 확률 (dev 보정)")


class ValidationResult(BaseModel):
    """검사 하나의 결과. 통과해도 남긴다 — 아무것도 안 막는 검사를 찾으려면."""

    check: str = Field(..., description="V1~V6")
    passed: bool
    detail: str = ""
    dropped_evidence: List[str] = Field(default_factory=list)


class TraceStep(BaseModel):
    """워크플로 한 단계. 결정에 영향을 준 것만 담는다.

    소요 시간을 넣지 않는 것은 게으름이 아니다. 실행마다 달라지는 값이 들어가면
    두 실행의 trace 를 비교할 수 없고, 회귀 테스트가 이 필드를 못 쓴다.
    """

    node: str
    summary: str
    detail: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """한 요청의 처리 전체."""

    request: str
    request_key: Tuple[str, int, str, int] = Field(
        ..., description="(source, page, serial, pair_index)")

    retrieved_evidence: List[Evidence] = Field(default_factory=list)
    rule_evidence: List[Evidence] = Field(default_factory=list)
    signals: Optional[RouterSignals] = None

    route: Optional[Path] = None
    route_reason: Optional[str] = Field(None, description="발화한 결정 표의 줄 (R1~R10)")

    decision: Optional[str] = None
    confidence: Optional[str] = None
    evidence_used: List[str] = Field(default_factory=list)

    validation: List[ValidationResult] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: Optional[AbstentionReason] = None

    execution_trace: List[TraceStep] = Field(default_factory=list)

    def step(self, node: str, summary: str, **detail: Any) -> None:
        self.execution_trace.append(
            TraceStep(node=node, summary=summary, detail=detail))

    def abstain(self, reason: AbstentionReason, summary: str = "") -> None:
        """기권은 한 곳에서만 일어난다. 이유 없이 기권할 수 없다."""
        self.abstained = True
        self.abstention_reason = reason
        self.decision = None
        self.confidence = "low"
        self.step("abstain", summary or reason.value, reason=reason.value)

    def all_evidence(self) -> List[Evidence]:
        return list(self.retrieved_evidence) + list(self.rule_evidence)

    def evidence_by_id(self) -> Dict[str, Evidence]:
        return {e.id: e for e in self.all_evidence()}

    def to_prediction(self) -> Dict[str, Any]:
        """기존 평가 하네스가 읽는 예측 레코드로 옮긴다.

        **평가 계층을 고치지 않기 위해** 여기서 형을 맞춘다. 하네스는
        `predicted` `confidence` `rule` 을 본다.
        """
        source, page, serial, pair_index = self.request_key
        return {
            "source": source, "page": page, "serial": serial,
            "pair_index": pair_index,
            "predicted": self.decision,
            "confidence": self.confidence or "low",
            "rule": f"agent:{self.route.value if self.route else '-'}"
                    f":{self.route_reason or '-'}",
            "abstained": self.abstained,
            "abstention_reason": (self.abstention_reason.value
                                  if self.abstention_reason else None),
            "evidence_used": list(self.evidence_used),
        }
