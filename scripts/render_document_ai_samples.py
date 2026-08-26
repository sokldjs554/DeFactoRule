#!/usr/bin/env python3
# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.io import load_jsonl  # noqa: E402
from app.document_ai.benchmark import (  # noqa: E402
    BENCHMARK_N,
    BENCHMARK_SOURCE,
    PROFILES,
    _actual_fields,
    _expected_fields,
    _ground_truth,
    _render,
    _select_rows,
)
from app.document_ai.extraction import extract_fields  # noqa: E402
from app.document_ai.ocr import TesseractOCR  # noqa: E402
from app.document_ai.validation import validate_extraction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible Document AI images/JSON for portfolio screenshots"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/document_ai_capture"))
    parser.add_argument(
        "--sample",
        type=int,
        default=12,
        help=f"0-based sample index within the fixed {BENCHMARK_N}-document benchmark",
    )
    args = parser.parse_args()

    rows = _select_rows(load_jsonl(BENCHMARK_SOURCE), BENCHMARK_N)
    if args.sample < 0 or args.sample >= len(rows):
        raise SystemExit(f"--sample must be between 0 and {len(rows) - 1}")
    row = rows[args.sample]
    truth = _ground_truth(row)
    expected = _expected_fields(row)
    ocr = TesseractOCR(language="kor", psm=6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "00_ground_truth.txt").write_text(truth + "\n", encoding="utf-8")

    manifest = {
        "sample_index": args.sample,
        "serial": str(row["serial"]),
        "expected": expected,
        "ocr_engine": ocr.version(),
        "files": [],
    }

    for order, profile in enumerate(PROFILES, 1):
        ext = "jpg" if profile["format"] == "jpeg" else "png"
        stem = f"{order:02d}_{profile['name']}"
        image_path = args.output_dir / f"{stem}.{ext}"
        json_path = args.output_dir / f"{stem}_extraction.json"

        image = _render(
            truth,
            dpi=int(profile["dpi"]),
            image_format=str(profile["format"]),
            quality=int(profile["quality"] or 45),
            font_size=float(profile["font_size"]),
            gray=float(profile["gray"]),
            skew_deg=float(profile["skew_deg"]),
        )
        image_path.write_bytes(image)

        output = ocr.recognize(image)
        fields = extract_fields(output.text)
        validation = validate_extraction(output.text, fields)
        payload = {
            "profile": profile["name"],
            "render": profile,
            "expected": expected,
            "ocr_text": output.text,
            "actual": _actual_fields(fields),
            "quotes": fields.quotes,
            "validation": asdict(validation),
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["files"].append({"image": str(image_path), "extraction": str(json_path)})

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"capture set written to {args.output_dir}")
    for item in manifest["files"]:
        print(f"- {item['image']}")
        print(f"  {item['extraction']}")


if __name__ == "__main__":
    main()
