#!/usr/bin/env python3
"""C-5 final deterministic freeze CLI. API/LLM 호출 없음."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.final_freeze import main  # noqa: E402

if __name__ == "__main__":
    main()
