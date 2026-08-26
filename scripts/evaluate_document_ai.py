#!/usr/bin/env python3
# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.document_ai.benchmark import evaluate_document_ai  # noqa: E402


def _run_checks() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "checks/document_ai", "-q"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="skip the dedicated Document AI checks before the benchmark",
    )
    args = parser.parse_args()
    if not args.skip_checks:
        _run_checks()
    result = evaluate_document_ai(n=args.n)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
