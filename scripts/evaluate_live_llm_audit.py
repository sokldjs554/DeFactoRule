#!/usr/bin/env python3
"""live LLM contract audit — CLI 진입점.

구현은 `app.evaluation.live_llm_audit` 에 있다. 기본은 dry-run(API 0회)이고
`--go` 에서만 실제로 호출한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.live_llm_audit import main  # noqa: E402

if __name__ == "__main__":
    main()
