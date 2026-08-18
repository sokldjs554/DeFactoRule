"""API 계약. Pydantic 이 경계에서 형식을 강제한다.

**기권을 계약에 넣는다.** 이 프로젝트의 결론은 "LLM 이 규칙보다 정확한 것이
아니라, 자기가 틀릴 때를 안다" 는 것이다(E2). 그 성질이 값어치를 가지려면
서비스가 "모르겠다" 를 1급 응답으로 돌려줄 수 있어야 한다. 낮은 신뢰도를
억지로 라벨로 바꿔 내보내면 AURC 로 보여준 이점이 그 자리에서 사라진다.

기권 여부를 정하는 것은 **결정론적 코드**다(명세 §9). 모델은 신뢰도 등급까지만
말하고, 그것을 자를지 말지는 운영 문턱이 정한다.
"""

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
            "이 등급 미만이면 기권한다. low 면 기권하지 않는다. "
            "medium 으로 올리면 커버리지가 줄고 정확도가 오른다 — "
            "그 맞바꿈이 위험-커버리지 곡선이다."
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
        None, description=f"결론 라벨. 기권하면 null. 가능한 값: {', '.join(NON_ACTIONS)}"
    )
    abstained: bool = Field(..., description="판정을 내리지 않았는가")
    abstain_reason: Optional[str] = None
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
    aurc: float = Field(..., description="위험의 평균. 낮을수록 좋다")
    flat: bool = Field(
        ..., description="운영점이 하나뿐인가 — 기권 신호가 없다는 뜻이다"
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
