#!/usr/bin/env python3
"""실패 케이스 레지스트리 보고 — CLI 진입점.

구현은 `app.evaluation.failure_report` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.failure_report import main  # noqa: E402

if __name__ == "__main__":
    main()
