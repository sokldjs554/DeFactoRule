from __future__ import annotations

import json
import shutil
from pathlib import Path

import pymupdf
import pytest

from app.document_ai.benchmark import evaluate_document_ai
from app.document_ai.extraction import DOCUMENT_SCHEMA, extract_fields, extract_fields_llm
from app.document_ai.intake import detect_mode
from app.document_ai.models import ExtractedDocument
from app.document_ai.ocr import OcrOutput, TesseractOCR
from app.document_ai.service import process_document
from app.document_ai.validation import validate_extraction
from app.infrastructure.schema_rules import check_output_schema

FORM = """금융규제 업무 문서
일련번호: 240001
업권: 전자금융
판단: 비조치
요청대상행위:
금융회사가 외부 클라우드 서비스를 이용하는 경우
"""


def _native_pdf(path: Path, text: str = FORM) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        pymupdf.Rect(45, 45, 550, 797),
        text,
        fontsize=14,
        fontname="korea",
        lineheight=1.25,
    )
    doc.save(path)


def _scan_pdf(path: Path, text: str = FORM) -> bytes:
    native = pymupdf.open()
    page = native.new_page(width=595, height=842)
    page.insert_textbox(
        pymupdf.Rect(45, 45, 550, 797),
        text,
        fontsize=14,
        fontname="korea",
        lineheight=1.25,
    )
    png = page.get_pixmap(dpi=180, alpha=False).tobytes("png")
    scan = pymupdf.open()
    target = scan.new_page(width=595, height=842)
    target.insert_image(target.rect, stream=png)
    scan.save(path)
    return png


def test_detects_native_pdf(tmp_path: Path) -> None:
    path = tmp_path / "native.pdf"
    _native_pdf(path)
    assert detect_mode(path) == "native"


def test_detects_scanned_pdf(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _scan_pdf(path)
    assert detect_mode(path) == "ocr"


def test_extracts_original_casebook_style() -> None:
    text = """비조치의견서(☑비조치□조치□기타)
(일련번호 250027)
업권: 전자금융
요청대상행위
클라우드 서비스를 이용하는 것이 가능한지 여부
판단
가능한 것으로 판단
판단이유
관련 규정 검토
"""
    fields = extract_fields(text)
    assert fields.serial == "250027"
    assert fields.sector == "전자금융"
    assert fields.decision == "비조치"
    assert fields.request == "클라우드 서비스를 이용하는 것이 가능한지 여부"


def test_validation_rejects_ungrounded_quote() -> None:
    fields = ExtractedDocument(
        serial="240001",
        sector="전자금융",
        decision="비조치",
        request="클라우드 이용",
        quotes={
            "serial": "240001",
            "sector": "전자금융",
            "decision": "비조치",
            "request": "원문에 없는 인용",
        },
    )
    report = validate_extraction(FORM, fields)
    assert not report.valid
    assert report.review_required
    assert "ungrounded_quote:request" in report.issues


def test_missing_fields_require_review() -> None:
    report = validate_extraction("빈 문서", extract_fields("빈 문서"))
    assert not report.valid
    assert report.review_required
    assert "invalid_or_missing_serial" in report.issues
    assert "missing_request" in report.issues


def test_native_service_never_needs_ocr(tmp_path: Path) -> None:
    path = tmp_path / "native.pdf"
    _native_pdf(path)

    class ExplodingOCR:
        def recognize(self, image_bytes: bytes):
            raise AssertionError("native PDF must not invoke OCR")

    result = process_document(path, ocr=ExplodingOCR())
    assert result.document.mode == "native"
    assert result.fields.serial == "240001"
    assert result.validation.valid


def test_scanned_service_uses_ocr_adapter(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _scan_pdf(path)

    class FakeOCR:
        calls = 0

        def recognize(self, image_bytes: bytes) -> OcrOutput:
            self.calls += 1
            return OcrOutput(FORM, "fake-ocr")

    ocr = FakeOCR()
    result = process_document(path, ocr=ocr)
    assert ocr.calls == 1
    assert result.document.mode == "ocr"
    assert result.fields.serial == "240001"
    assert result.validation.valid


def test_real_tesseract_reads_clean_korean_form(tmp_path: Path) -> None:
    if not shutil.which("tesseract"):
        pytest.skip("tesseract executable is not installed")
    path = tmp_path / "scan.pdf"
    png = _scan_pdf(path)
    output = TesseractOCR(language="kor", psm=6).recognize(png)
    fields = extract_fields(output.text)
    assert fields.serial == "240001"
    assert fields.decision == "비조치"
    assert fields.request and "클라우드" in fields.request


def test_document_schema_respects_known_api_contract() -> None:
    assert check_output_schema(DOCUMENT_SCHEMA) == []


class _Usage:
    input_tokens = 50
    output_tokens = 30


class _Block:
    type = "text"

    def __init__(self, payload: dict) -> None:
        self.text = json.dumps(payload, ensure_ascii=False)


class _Response:
    stop_reason = "end_turn"
    usage = _Usage()

    def __init__(self, payload: dict) -> None:
        self.content = [_Block(payload)]


class _Messages:
    def create(self, **kwargs):
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        return _Response(
            {
                "serial": "240001",
                "sector": "전자금융",
                "decision": "비조치",
                "request": "클라우드 이용",
                "quotes": {
                    "serial": "240001",
                    "sector": "전자금융",
                    "decision": "비조치",
                    "request": "클라우드 이용",
                },
            }
        )


class _Client:
    messages = _Messages()


def test_optional_llm_extractor_uses_structured_output() -> None:
    fields = extract_fields_llm(
        "일련번호 240001 전자금융 비조치 클라우드 이용",
        _Client(),
    )
    assert fields.serial == "240001"
    assert fields.quotes["request"] == "클라우드 이용"


def test_synthetic_document_ai_benchmark_has_two_profiles() -> None:
    if not shutil.which("tesseract"):
        pytest.skip("tesseract executable is not installed")
    result = evaluate_document_ai(n=3)
    assert result["benchmark"] == "synthetic_scanned_financial_requests"
    assert result["n_documents"] == 3
    assert result["ocr_language"] == "kor"
    assert len(result["profiles"]) == 2
    for profile in result["profiles"]:
        assert 0.0 <= profile["mean_cer_no_space"] <= 1.5
        assert 0.0 <= profile["scalar_field_exact"] <= 1.0
        assert profile["fully_valid"] + profile["review_required"] == 3
