"""Structured extraction from OCR/native text, plus an optional LLM extractor."""
from __future__ import annotations

import re
from typing import Dict, Optional

from app.document_ai.models import ExtractedDocument
from app.extraction.casebook import NONACTION_FIELDS, RE_DECISION, RE_SERIAL, split_fields
from app.infrastructure.anthropic_client import call_structured

_DECISIONS = ("비조치", "조치", "기타")


def _clean_scalar(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t:：._-、")


def _line_value(text: str, label: str) -> Optional[str]:
    match = re.search(rf"(?m)^\s*{re.escape(label)}\s*[:：]?\s*(.+?)\s*$", text)
    return _clean_scalar(match.group(1)) if match else None


def _request_block(text: str) -> Optional[str]:
    match = re.search(r"(?ms)^\s*요청대상행위\s*[:：]?\s*\n(?P<body>.+)$", text)
    if not match:
        return None
    body = match.group("body").strip()
    return re.sub(r"\s+", " ", body) if body else None


def extract_fields(text: str) -> ExtractedDocument:
    """Deterministic baseline for OCR text and the original non-action casebook layout."""
    serial_line = _line_value(text, "일련번호")
    serial_match = RE_SERIAL.search(text) or re.search(r"\d{5,7}", serial_line or "")
    sector = _line_value(text, "업권")

    checkbox = RE_DECISION.search(text[:500])
    decision_raw = checkbox.group(0) if checkbox else (_line_value(text, "판단") or "")
    decision = checkbox.group(1) if checkbox else next(
        (label for label in _DECISIONS if label in decision_raw), None
    )

    parsed_fields, _ = split_fields(text, NONACTION_FIELDS)
    request = parsed_fields.get("요청대상행위") or _request_block(text)
    quotes: Dict[str, str] = {}
    if serial_match:
        quotes["serial"] = serial_match.group(0)
    if sector:
        quotes["sector"] = sector
    if decision_raw:
        quotes["decision"] = decision_raw
    if request:
        quotes["request"] = request
    if serial_match and serial_match.lastindex:
        serial = serial_match.group(1)
    else:
        serial = serial_match.group(0) if serial_match else None
    return ExtractedDocument(
        serial=serial,
        sector=sector,
        decision=decision,
        request=request,
        quotes=quotes,
    )


_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "serial": _NULLABLE_STRING,
        "sector": _NULLABLE_STRING,
        "decision": _NULLABLE_STRING,
        "request": _NULLABLE_STRING,
        "quotes": {
            "type": "object",
            "properties": {
                "serial": _NULLABLE_STRING,
                "sector": _NULLABLE_STRING,
                "decision": _NULLABLE_STRING,
                "request": _NULLABLE_STRING,
            },
            "required": ["serial", "sector", "decision", "request"],
            "additionalProperties": False,
        },
    },
    "required": ["serial", "sector", "decision", "request", "quotes"],
    "additionalProperties": False,
}

_SYSTEM = """You extract fields from OCR text. Never infer a missing field.
Return null when the source does not support a value. Every non-null value must carry a
verbatim supporting quote copied from the OCR text. Do not make a legal decision."""


def extract_fields_llm(text: str, client) -> ExtractedDocument:
    """Optional structured LLM extraction; deterministic validation remains authoritative."""
    prompt = "OCR TEXT\n---\n" + text[:18000]
    record = call_structured(
        client, _SYSTEM, prompt, DOCUMENT_SCHEMA, max_tokens=1800, effort="low"
    )
    if "data" not in record:
        raise RuntimeError(record.get("error") or "document extraction failed")
    data = record["data"]
    quotes = {k: v for k, v in (data.get("quotes") or {}).items() if isinstance(v, str)}
    return ExtractedDocument(
        serial=data.get("serial"),
        sector=data.get("sector"),
        decision=data.get("decision"),
        request=data.get("request"),
        quotes=quotes,
    )
