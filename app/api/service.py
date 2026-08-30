"""분류 엔진을 실행하고 최소 신뢰도 기준을 적용합니다."""

from __future__ import annotations

import json

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    Confidence,
    Engine,
    Evidence,
)
from app.core.paths import DEV_BASE_RATES
from app.domain.confidence import meets


class EngineUnavailable(RuntimeError):
    """엔진을 쓸 수 없다. 왜 못 쓰는지를 메시지에 담는다."""


def _base_rates() -> dict | None:
    if not DEV_BASE_RATES.exists():
        return None
    table = json.loads(DEV_BASE_RATES.read_text(encoding="utf-8"))
    if table.get("source") != "dev":
        # test 에서 뽑은 기저율을 프롬프트에 넣으면 정답을 흘리는 것이다.
        raise EngineUnavailable("기저율이 dev 에서 나온 것이 아닙니다. 정답 누출 위험.")
    return table


def _classify_rule(req: ClassifyRequest) -> ClassifyResponse:
    from app.rules.nonaction import classify

    label, rule, confidence = classify(req.request_text, "keyword")
    return ClassifyResponse(
        decision=label,
        abstained=False,
        confidence=Confidence(confidence),
        engine=Engine.RULE,
        rule=rule,
        evidence=Evidence(quote="", grounded=None),
    )


def _classify_llm(req: ClassifyRequest) -> ClassifyResponse:
    from app.agents.classifier import (
        NONACTION_SYSTEM,
        TASKS,
        FatalApiError,
        classify_one,
    )
    from app.domain.base_rates import describe_sector

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - 설치 여부에 달림
        raise EngineUnavailable(
            "anthropic 패키지가 없습니다: pip install anthropic"
        ) from exc

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover - 자격증명 여부에 달림
        raise EngineUnavailable(
            f"Anthropic 자격증명을 찾을 수 없습니다: {exc}"
        ) from exc

    table = _base_rates()
    # 업권을 알면 그 업권의 기저율을, 모르면 문맥 없이 간다. 기저율 파일이
    # 없으면 문맥 없는 과제로 떨어진다 — 없는 숫자를 지어내지 않는다.
    if table and req.sector:
        task = dict(TASKS["nonaction_sector"], base_rates=table)
        context = describe_sector(table, req.sector)
    elif table:
        task = dict(TASKS["nonaction"])
        context = None
    else:
        task = dict(TASKS["nonaction"])
        context = None
    task["system"] = NONACTION_SYSTEM

    row = {"request": req.request_text, "sector": req.sector}
    try:
        result = classify_one(client, row, task)
    except FatalApiError as exc:
        raise EngineUnavailable(f"계정 수준 오류: {exc}") from exc

    if "error" in result:
        raise EngineUnavailable(
            f"{result['error']}: {result.get('error_detail', '')}".strip(": ")
        )

    return ClassifyResponse(
        decision=result["predicted"],
        abstained=False,
        confidence=Confidence(result["confidence"]),
        engine=Engine.LLM,
        evidence=Evidence(
            quote=result.get("evidence", ""),
            grounded=result.get("evidence_grounded"),
        ),
        base_rate_context=context,
    )


def classify(req: ClassifyRequest) -> ClassifyResponse:
    """엔진을 고르고, 결정론적 기권 정책을 마지막에 적용한다."""
    result = _classify_rule(req) if req.engine is Engine.RULE else _classify_llm(req)

    if not meets(result.confidence.value, req.min_confidence.value):
        # 라벨을 지우고 기권으로 바꾼다. 무엇을 지웠는지는 남기지 않는다 —
        # 남기면 호출자가 그것을 답으로 쓰게 되고, 기권의 의미가 사라진다.
        return result.model_copy(
            update={
                "decision": None,
                "abstained": True,
                "abstain_reason": (
                    f"신뢰도 {result.confidence.value}가 설정한 최소 신뢰도 "
                    f"{req.min_confidence.value}보다 낮아 판단을 보류했습니다."
                ),
            }
        )
    return result
