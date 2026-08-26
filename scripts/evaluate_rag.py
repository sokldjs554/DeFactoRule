#!/usr/bin/env python3
"""Print the API-free Evidence RAG retrieval evaluation as JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.evaluation import evaluate_retrieval  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(evaluate_retrieval(k=5), ensure_ascii=False, indent=2))
