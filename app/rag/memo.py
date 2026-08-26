"""Structured evidence memo contract and deterministic citation validation."""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.evidence import EvidenceHit

MEMO_SYSTEM = """당신은 금융업무 선례를 정리하는 evidence analyst다.
최종 결론이나 비조치/조치/기타 라벨을 결정하지 않는다.
제공된 evidence만 사용한다. 각 claim에는 evidence_id와 그 evidence 원문에
글자 그대로 존재하는 quote를 반드시 붙인다. 근거가 부족하면 uncertainty에
명시하고 handoff_recommended=true로 둔다.
"""

MEMO_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim": {"type": "string"},
                    "evidence_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["claim", "evidence_id", "quote"],
            },
        },
        "uncertainty": {"type": "string"},
        "handoff_recommended": {"type": "boolean"},
    },
    "required": ["summary", "claims", "uncertainty", "handoff_recommended"],
}


@dataclass(frozen=True)
class MemoValidation:
    valid: bool
    invalid_citations: tuple[str, ...]
    ungrounded_quotes: tuple[str, ...]
    reason: str | None = None


def build_prompt(request_text: str, context: str) -> str:
    return f"""[신규 업무 요청]
{request_text}

[검색된 선례 evidence]
{context}

위 evidence만 이용해 의사결정을 위한 evidence memo를 작성하라.
- 최종 라벨/판정은 내리지 않는다.
- claims는 신규 요청과 관련 있는 선례 사실만 적는다.
- 각 claim의 evidence_id는 위 대괄호 ID 중 하나여야 한다.
- quote는 해당 precedent_request에 실제 존재하는 연속 원문 구절이어야 한다.
- evidence가 부족하거나 서로 충돌하면 uncertainty와 handoff_recommended에 반영한다.
"""


def validate_memo(data: dict, hits: list[EvidenceHit]) -> MemoValidation:
    by_id = {hit.evidence_id: hit for hit in hits}
    invalid: list[str] = []
    ungrounded: list[str] = []

    claims = data.get("claims")
    if not isinstance(claims, list):
        return MemoValidation(False, (), (), "claims must be a list")

    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            return MemoValidation(False, (), (), f"claim[{index}] must be an object")
        citation = str(claim.get("evidence_id") or "")
        quote = str(claim.get("quote") or "").strip()
        hit = by_id.get(citation)
        if hit is None:
            invalid.append(citation or f"<empty:{index}>")
            continue
        if not quote or quote not in hit.request:
            ungrounded.append(f"{citation}:{quote[:80]}")

    valid = not invalid and not ungrounded
    reason = None if valid else "citation_or_quote_validation_failed"
    return MemoValidation(valid, tuple(invalid), tuple(ungrounded), reason)
