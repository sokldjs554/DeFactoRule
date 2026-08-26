"""Typed records for the OCR-aware document intake layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    source: str  # native | ocr
    ocr_mean_confidence: float | None = None
    ocr_low_confidence_fraction: float | None = None


@dataclass(frozen=True)
class DocumentText:
    mode: str  # native | ocr
    engine: str
    pages: list[PageText]
    text_health: float

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip()).strip()

    @property
    def ocr_mean_confidence(self) -> float | None:
        values = [
            page.ocr_mean_confidence
            for page in self.pages
            if page.ocr_mean_confidence is not None
        ]
        return mean(values) if values else None

    @property
    def ocr_low_confidence_fraction(self) -> float | None:
        values = [
            page.ocr_low_confidence_fraction
            for page in self.pages
            if page.ocr_low_confidence_fraction is not None
        ]
        # Conservative multi-page policy: one poor page is enough to trigger review.
        return max(values) if values else None


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
