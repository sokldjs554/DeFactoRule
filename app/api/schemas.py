"""API 요청 및 응답 스키마입니다."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.domain.labels import NON_ACTIONS


class Engine(str, Enum):
    """무엇으로 판정할 것인가."""

    RULE = "rule"        # 결정론적 어휘 규칙. 비용 0, 항상 사용 가능
    LLM = "llm"          # 문서 이해. API 키가 있어야 한다


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClassifyRequest(BaseModel):
    request_text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="비조치의견서의 '요청대상행위' 본문",
    )
    sector: Optional[str] = Field(
        None, description="업권. 알면 기저율 문맥에 쓰인다"
    )
    engine: Engine = Engine.RULE
    min_confidence: Confidence = Field(
        Confidence.LOW,
        description=(
            "결과를 표시할 최소 신뢰도입니다. 이 값보다 신뢰도가 낮으면 "
            "판단을 보류합니다."
        ),
    )


class Evidence(BaseModel):
    quote: str = Field("", description="판정 근거로 든 원문 구절")
    grounded: Optional[bool] = Field(
        None,
        description=(
            "인용이 원문에 글자 그대로 있는지. 규칙 엔진은 인용을 만들지 않으므로 null."
        ),
    )


class ClassifyResponse(BaseModel):
    decision: Optional[str] = Field(
        None,
        description=(
            "예측한 결론입니다. 판단을 보류하면 null입니다. "
            f"가능한 값: {', '.join(NON_ACTIONS)}"
        ),
    )
    abstained: bool = Field(..., description="판단을 보류했는지 여부")
    abstain_reason: Optional[str] = Field(
        None, description="판단을 보류한 이유입니다. 보류하지 않았으면 null입니다."
    )
    confidence: Confidence
    engine: Engine
    rule: Optional[str] = Field(None, description="규칙 엔진에서 실제로 걸린 규칙")
    evidence: Evidence = Evidence()
    base_rate_context: Optional[str] = Field(
        None, description="판정에 함께 제시된 기저율 문장 (dev 에서만 산출)"
    )


class SectorRate(BaseModel):
    sector: str
    n: int
    reliable: bool = Field(
        ..., description="표본이 충분한가. 아니면 전체 기저율로 대체된다"
    )
    rates: dict


class BaseRatesResponse(BaseModel):
    source: str = Field(..., description="반드시 'dev'. test 에서 뽑으면 정답 누출이다")
    n: int
    min_sector_n: int
    overall: dict
    sectors: List[SectorRate]


class OperatingPoint(BaseModel):
    coverage: float
    n: int
    risk: float
    accuracy: float
    macro_f1: float


class ModelCurve(BaseModel):
    name: str
    aurc: float = Field(
        ..., description="위험–커버리지 곡선을 요약한 값입니다. 낮을수록 좋습니다."
    )
    flat: bool = Field(
        ..., description="신뢰도 기준을 바꿔도 결과 지점이 하나뿐인지 여부"
    )
    points: List[OperatingPoint]


class RiskCoverageResponse(BaseModel):
    n: int = Field(..., description="공통 표본 수")
    label_set: str
    curves: List[ModelCurve]
    note: str


class FailureCase(BaseModel):
    id: str
    layer: str
    category: str
    title: str
    symptom: str
    fix: str
    status: str
    metric: Optional[dict] = None
    probe: Optional[str] = None
    probe_passed: Optional[bool] = None
    probe_detail: Optional[str] = None


class FailureReport(BaseModel):
    total: int
    by_layer: dict
    open_cases: List[str]
    cases: List[FailureCase]
