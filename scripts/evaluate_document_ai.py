#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.document_ai.benchmark import evaluate_document_ai


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    result = evaluate_document_ai(n=args.n)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
