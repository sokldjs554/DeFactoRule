#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.document_ai.service import process_document
from app.infrastructure.anthropic_client import connect


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR-aware document intake and field extraction")
    parser.add_argument("path", type=Path)
    parser.add_argument("--llm", action="store_true", help="use opt-in Anthropic structured extraction")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    client = connect() if args.llm else None
    result = process_document(args.path, client=client, use_llm=args.llm, dpi=args.dpi)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
