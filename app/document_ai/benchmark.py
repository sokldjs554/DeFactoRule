"""Reproducible synthetic scanned-document benchmark built from clean financial requests."""
from __future__ import annotations

import re
from collections.abc import Iterable
from statistics import mean

import pymupdf

from app.core.io import load_jsonl
from app.core.paths import EVAL
from app.document_ai.extraction import extract_fields
from app.document_ai.ocr import TesseractOCR
from app.document_ai.validation import validate_extraction

BENCHMARK_N = 20
BENCHMARK_SOURCE = EVAL / "nonaction_test_clean.jsonl"
PROFILES = (
    {"name": "clean_220dpi_png", "dpi": 220, "format": "png", "quality": None},
    {"name": "degraded_120dpi_jpeg", "dpi": 120, "format": "jpeg", "quality": 45},
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _cer(reference: str, hypothesis: str) -> float:
    reference = _norm(reference)
    hypothesis = _norm(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _distance(reference, hypothesis) / len(reference)


def _select_rows(rows: Iterable[dict], n: int) -> list[dict]:
    selected = []
    for row in rows:
        request = str(row.get("request") or "")
        if row.get("masked_leaks") or not row.get("serial") or len(request) > 700:
            continue
        selected.append(row)
        if len(selected) == n:
            break
    if len(selected) < n:
        raise RuntimeError(f"benchmark needs {n} eligible rows, found {len(selected)}")
    return selected


def _ground_truth(row: dict) -> str:
    return (
        "금융규제 업무 문서\n"
        f"일련번호: {row['serial']}\n"
        f"업권: {row.get('sector') or ''}\n"
        f"판단: {row['label']}\n"
        "요청대상행위:\n"
        f"{str(row['request']).strip()}"
    )


def _render(text: str, dpi: int, image_format: str, quality: int = 45) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    remaining = page.insert_textbox(
        pymupdf.Rect(45, 45, 550, 797),
        text,
        fontsize=11.5,
        fontname="korea",
        lineheight=1.25,
    )
    if remaining < 0:
        raise RuntimeError("synthetic benchmark page overflow")
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    if image_format == "jpeg":
        return pix.tobytes("jpeg", jpg_quality=quality)
    return pix.tobytes("png")


def _scalar_accuracy(expected: dict, actual) -> tuple[int, int]:
    pairs = (
        (str(expected["serial"]), actual.serial),
        (str(expected.get("sector") or ""), actual.sector),
        (str(expected["label"]), actual.decision),
    )
    correct = sum(_norm(a) == _norm(b or "") for a, b in pairs)
    return correct, len(pairs)


def evaluate_document_ai(n: int = BENCHMARK_N) -> dict:
    rows = _select_rows(load_jsonl(BENCHMARK_SOURCE), n)
    ocr = TesseractOCR(language="kor", psm=6)
    profile_results = []
    for profile in PROFILES:
        char_errors: list[float] = []
        request_errors: list[float] = []
        scalar_correct = scalar_total = 0
        valid = 0
        review = 0
        for row in rows:
            truth = _ground_truth(row)
            image = _render(
                truth,
                dpi=int(profile["dpi"]),
                image_format=str(profile["format"]),
                quality=int(profile["quality"] or 45),
            )
            output = ocr.recognize(image)
            fields = extract_fields(output.text)
            report = validate_extraction(output.text, fields)
            char_errors.append(_cer(truth, output.text))
            request_errors.append(_cer(str(row["request"]), fields.request or ""))
            c, t = _scalar_accuracy(row, fields)
            scalar_correct += c
            scalar_total += t
            valid += int(report.valid)
            review += int(report.review_required)
        profile_results.append(
            {
                "profile": profile["name"],
                "n": n,
                "mean_cer_no_space": mean(char_errors),
                "mean_request_cer_no_space": mean(request_errors),
                "scalar_field_exact": scalar_correct / scalar_total,
                "fully_valid": valid,
                "review_required": review,
            }
        )
    return {
        "schema_version": 1,
        "benchmark": "synthetic_scanned_financial_requests",
        "source": str(BENCHMARK_SOURCE.relative_to(EVAL.parent.parent)),
        "selection": "first 20 rows with masked_leaks=0, serial present, request<=700 chars",
        "n_documents": n,
        "profiles": profile_results,
        "ocr_engine": ocr.version(),
        "ocr_language": ocr.language,
        "note": (
            "Synthetic rasterization/degradation benchmark; not a claim about real scanned "
            "customer documents. No LLM calls are used."
        ),
    }
