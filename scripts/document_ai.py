#!/usr/bin/env python3
# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.document_ai.rag_bridge import process_document_with_rag  # noqa: E402
from app.document_ai.service import process_document  # noqa: E402
from app.infrastructure.anthropic_client import connect  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR-aware document intake and field extraction")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="use opt-in Anthropic structured extraction",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="forward validated extraction into temporal Evidence RAG",
    )
    parser.add_argument(
        "--rag-memo",
        action="store_true",
        help="generate an opt-in grounded Evidence RAG memo (implies --rag)",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    needs_client = args.llm or args.rag_memo
    client = connect() if needs_client else None
    if args.rag or args.rag_memo:
        result = process_document_with_rag(
            args.path,
            client=client,
            use_llm_extraction=args.llm,
            generate_memo=args.rag_memo,
            dpi=args.dpi,
        )
    else:
        result = process_document(args.path, client=client, use_llm=args.llm, dpi=args.dpi)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
