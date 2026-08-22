#!/usr/bin/env python3
"""clean 평가 규약 감사 — CLI 진입점.

구현은 `app.evaluation.protocol_audit` 에 있다. 문턱을 하나도 바꾸지 않는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.protocol_audit import main  # noqa: E402

if __name__ == "__main__":
    main()
