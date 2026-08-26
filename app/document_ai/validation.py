"""Fail-closed validation for structured document extraction."""
from __future__ import annotations

import re
from typing import List

from app.document_ai.models import ExtractedDocument, ValidationReport

_VALID_DECISIONS = frozenset({"비조치", "조치", "기타"})
_REQUIRED = ("serial", "sector", "decision", "request")


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def validate_extraction(source_text: str, fields: ExtractedDocument) -> ValidationReport:
    issues: List[str] = []
    if not fields.serial or not re.fullmatch(r"\d{5,7}", fields.serial):
        issues.append("invalid_or_missing_serial")
    if not fields.sector:
        issues.append("missing_sector")
    if fields.decision not in _VALID_DECISIONS:
        issues.append("invalid_or_missing_decision")
    if not fields.request:
        issues.append("missing_request")

    normalized_source = _norm(source_text)
    for name in _REQUIRED:
        value = getattr(fields, name)
        quote = fields.quotes.get(name)
        if value is None:
            continue
        if not quote:
            issues.append(f"missing_quote:{name}")
            continue
        if _norm(quote) not in normalized_source:
            issues.append(f"ungrounded_quote:{name}")
    return ValidationReport(
        valid=not issues,
        review_required=bool(issues),
        issues=issues,
    )
