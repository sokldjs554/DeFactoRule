from __future__ import annotations

import shutil

import pytest

from app.document_ai.benchmark import evaluate_document_ai


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
