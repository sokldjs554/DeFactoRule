"""Reproducible synthetic scanned-document benchmark built from clean financial requests.

The benchmark intentionally evaluates more than OCR character accuracy. It measures
field-level exact extraction, document-level exactness, and whether the fail-closed
validator actually routes ground-truth extraction errors to review.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from statistics import mean

import pymupdf

from app.core.io import load_jsonl
from app.core.paths import EVAL
from app.document_ai.extraction import extract_fields
from app.document_ai.ocr import TesseractOCR
from app.document_ai.validation import (
    OCR_LOW_CONFIDENCE_FRACTION_CEILING,
    OCR_MEAN_CONFIDENCE_FLOOR,
    validate_extraction,
)

BENCHMARK_N = 60
BENCHMARK_SOURCE = EVAL / "nonaction_test_clean.jsonl"
FIELD_NAMES = ("serial", "sector", "decision", "request")
PROFILES = (
    {
        "name": "clean_220dpi_png",
        "dpi": 220,
        "format": "png",
        "quality": None,
        "font_size": 11.5,
        "gray": 0.0,
        "skew_deg": 0.0,
    },
    {
        "name": "standard_170dpi_jpeg",
        "dpi": 170,
        "format": "jpeg",
        "quality": 70,
        "font_size": 10.5,
        "gray": 0.12,
        "skew_deg": 0.6,
    },
    {
        "name": "degraded_120dpi_jpeg",
        "dpi": 120,
        "format": "jpeg",
        "quality": 40,
        "font_size": 9.5,
        "gray": 0.30,
        "skew_deg": 1.5,
    },
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
    eligible = []
    for row in rows:
        request = str(row.get("request") or "")
        if row.get("masked_leaks") or not row.get("serial") or len(request) > 700:
            continue
        eligible.append(row)
    if len(eligible) < n:
        raise RuntimeError(f"benchmark needs {n} eligible rows, found {len(eligible)}")

    # Do not cherry-pick the easiest first N rows. Spread deterministically across
    # the eligible clean split so the benchmark samples the corpus more broadly.
    indices = [int(i * len(eligible) / n) for i in range(n)]
    return [eligible[index] for index in indices]


def _ground_truth(row: dict) -> str:
    return (
        "금융규제 업무 문서\n"
        f"일련번호: {row['serial']}\n"
        f"업권: {row.get('sector') or ''}\n"
        f"판단: {row['label']}\n"
        "요청대상행위:\n"
        f"{str(row['request']).strip()}"
    )


def _expected_fields(row: dict) -> dict[str, str]:
    return {
        "serial": str(row["serial"]),
        "sector": str(row.get("sector") or ""),
        "decision": str(row["label"]),
        "request": str(row["request"]).strip(),
    }


def _actual_fields(fields) -> dict[str, str]:
    return {
        "serial": str(fields.serial or ""),
        "sector": str(fields.sector or ""),
        "decision": str(fields.decision or ""),
        "request": str(fields.request or ""),
    }


def _render(
    text: str,
    *,
    dpi: int,
    image_format: str,
    quality: int = 45,
    font_size: float = 11.5,
    gray: float = 0.0,
    skew_deg: float = 0.0,
) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    remaining = page.insert_textbox(
        pymupdf.Rect(45, 45, 550, 797),
        text,
        fontsize=font_size,
        fontname="korea",
        lineheight=1.25,
        color=(gray, gray, gray),
    )
    if remaining < 0:
        raise RuntimeError("synthetic benchmark page overflow")
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale).prerotate(skew_deg)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    if image_format == "jpeg":
        return pix.tobytes("jpeg", jpg_quality=quality)
    return pix.tobytes("png")


def _prf_counts(expected: dict[str, str], actual: dict[str, str]) -> tuple[int, int, int, int]:
    tp = fp = fn = exact = 0
    for name in FIELD_NAMES:
        exp = _norm(expected[name])
        got = _norm(actual[name])
        if got == exp:
            exact += 1
            tp += 1
            continue
        fn += 1
        if got:
            fp += 1
    return tp, fp, fn, exact


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def evaluate_document_ai(n: int = BENCHMARK_N) -> dict:
    rows = _select_rows(load_jsonl(BENCHMARK_SOURCE), n)
    ocr = TesseractOCR(language="kor", psm=6)
    profile_results = []

    for profile in PROFILES:
        char_errors: list[float] = []
        request_errors: list[float] = []
        ocr_confidences: list[float] = []
        low_confidence_fractions: list[float] = []
        tp = fp = fn = field_exact = 0
        document_exact = 0
        valid = 0
        review = 0
        erroneous_documents = 0
        review_on_error = 0
        review_on_correct = 0

        for row in rows:
            truth = _ground_truth(row)
            image = _render(
                truth,
                dpi=int(profile["dpi"]),
                image_format=str(profile["format"]),
                quality=int(profile["quality"] or 45),
                font_size=float(profile["font_size"]),
                gray=float(profile["gray"]),
                skew_deg=float(profile["skew_deg"]),
            )
            output = ocr.recognize(image)
            fields = extract_fields(output.text)
            report = validate_extraction(
                output.text,
                fields,
                ocr_mean_confidence=output.mean_confidence,
                ocr_low_confidence_fraction=output.low_confidence_fraction,
            )
            expected = _expected_fields(row)
            actual = _actual_fields(fields)

            char_errors.append(_cer(truth, output.text))
            request_errors.append(_cer(expected["request"], actual["request"]))
            if output.mean_confidence is not None:
                ocr_confidences.append(output.mean_confidence)
            if output.low_confidence_fraction is not None:
                low_confidence_fractions.append(output.low_confidence_fraction)

            row_tp, row_fp, row_fn, row_exact = _prf_counts(expected, actual)
            tp += row_tp
            fp += row_fp
            fn += row_fn
            field_exact += row_exact

            row_document_exact = row_exact == len(FIELD_NAMES)
            document_exact += int(row_document_exact)
            valid += int(report.valid)
            review += int(report.review_required)

            if not row_document_exact:
                erroneous_documents += 1
                review_on_error += int(report.review_required)
            elif report.review_required:
                review_on_correct += 1

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        mean_request_cer = mean(request_errors)

        profile_results.append(
            {
                "profile": profile["name"],
                "n": n,
                "render": {
                    "dpi": profile["dpi"],
                    "format": profile["format"],
                    "quality": profile["quality"],
                    "font_size": profile["font_size"],
                    "gray": profile["gray"],
                    "skew_deg": profile["skew_deg"],
                },
                "mean_cer_no_space": mean(char_errors),
                "mean_request_cer_no_space": mean_request_cer,
                "mean_request_char_accuracy": max(0.0, 1.0 - mean_request_cer),
                "mean_ocr_word_confidence": mean(ocr_confidences) if ocr_confidences else None,
                "mean_low_confidence_token_fraction": (
                    mean(low_confidence_fractions) if low_confidence_fractions else None
                ),
                "field_precision_exact": precision,
                "field_recall_exact": recall,
                "field_f1_exact": f1,
                "field_exact_rate": field_exact / (n * len(FIELD_NAMES)),
                "document_exact_match": document_exact / n,
                "fully_valid": valid,
                "auto_accept_rate": valid / n,
                "review_required": review,
                "review_rate": review / n,
                "ground_truth_error_documents": erroneous_documents,
                "review_on_ground_truth_error": review_on_error,
                "error_detection_recall": _safe_div(review_on_error, erroneous_documents),
                "review_on_ground_truth_correct": review_on_correct,
            }
        )

    return {
        "schema_version": 3,
        "benchmark": "synthetic_scanned_financial_requests",
        "source": str(BENCHMARK_SOURCE.relative_to(EVAL.parent.parent)),
        "selection": (
            f"{n} rows deterministically spread across eligible clean-test rows; "
            "masked_leaks=0, serial present, request<=700 chars"
        ),
        "n_documents": n,
        "fields": list(FIELD_NAMES),
        "ocr_quality_gate": {
            "mean_confidence_floor": OCR_MEAN_CONFIDENCE_FLOOR,
            "low_confidence_fraction_ceiling": OCR_LOW_CONFIDENCE_FRACTION_CEILING,
            "low_confidence_token_threshold": 60.0,
            "policy": "review when mean confidence < floor OR low-confidence fraction > ceiling",
            "preregistered_before_post_gate_benchmark": True,
        },
        "profiles": profile_results,
        "ocr_engine": ocr.version(),
        "ocr_language": ocr.language,
        "note": (
            "Synthetic rasterization/degradation benchmark; not a claim about real scanned "
            "customer documents. No LLM calls are used. Exact field/document metrics compare "
            "against ground truth; review metrics show how often structural, grounding, and "
            "predeclared OCR-confidence checks catch those extraction errors."
        ),
    }
