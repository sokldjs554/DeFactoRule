from __future__ import annotations

from app.document_ai.extraction import extract_fields
from app.document_ai.ocr import TesseractOCR
from app.document_ai.validation import (
    OCR_LOW_CONFIDENCE_FRACTION_CEILING,
    OCR_MEAN_CONFIDENCE_FLOOR,
    validate_extraction,
)

FORM = """금융규제 업무 문서
일련번호: 240001
업권: 전자금융
판단: 비조치
요청대상행위:
금융회사가 외부 클라우드 서비스를 이용하는 경우
"""


def test_tesseract_tsv_parser_reconstructs_lines_and_confidence() -> None:
    header = [
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    ]
    tsv = "\t".join(header) + "\n"
    tsv += "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t95.0\t금융규제\n"
    tsv += "5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t55.0\t업무\n"
    tsv += "5\t1\t1\t1\t2\t1\t0\t0\t10\t10\t85.0\t문서\n"

    text, mean_confidence, low_fraction = TesseractOCR._parse_tsv(tsv)
    assert text == "금융규제 업무\n문서"
    assert mean_confidence == (95.0 + 55.0 + 85.0) / 3
    assert low_fraction == 1 / 3


def test_predeclared_mean_confidence_floor_routes_to_review() -> None:
    fields = extract_fields(FORM)
    report = validate_extraction(
        FORM,
        fields,
        ocr_mean_confidence=OCR_MEAN_CONFIDENCE_FLOOR - 0.01,
        ocr_low_confidence_fraction=0.0,
    )
    assert report.review_required
    assert "low_ocr_mean_confidence" in report.issues


def test_predeclared_low_confidence_fraction_routes_to_review() -> None:
    fields = extract_fields(FORM)
    report = validate_extraction(
        FORM,
        fields,
        ocr_mean_confidence=99.0,
        ocr_low_confidence_fraction=OCR_LOW_CONFIDENCE_FRACTION_CEILING + 0.01,
    )
    assert report.review_required
    assert "high_low_confidence_token_fraction" in report.issues


def test_quality_exactly_on_threshold_is_not_rejected() -> None:
    fields = extract_fields(FORM)
    report = validate_extraction(
        FORM,
        fields,
        ocr_mean_confidence=OCR_MEAN_CONFIDENCE_FLOOR,
        ocr_low_confidence_fraction=OCR_LOW_CONFIDENCE_FRACTION_CEILING,
    )
    assert report.valid
