#!/usr/bin/env python3
"""Print the API-free Evidence RAG retrieval evaluation as JSON."""

import json

from app.rag.evaluation import evaluate_retrieval


if __name__ == "__main__":
    print(json.dumps(evaluate_retrieval(k=5), ensure_ascii=False, indent=2))
