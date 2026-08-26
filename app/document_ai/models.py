"""Typed records for the OCR-aware document intake layer."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    source: str  # native | ocr


@dataclass(frozen=True)
class DocumentText:
    mode: str  # native | ocr
    engine: str
    pages: list[PageText]
    text_health: float

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip()).strip()


@dataclass(frozen=True)
class ExtractedDocument:
    serial: str | None
    sector: str | None
    decision: str | None
    request: str | None
    quotes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    review_required: bool
    issues: list[str]


@dataclass(frozen=True)
class DocumentAIResult:
    document: DocumentText
    fields: ExtractedDocument
    validation: ValidationReport
    extractor: str
    input_name: str
