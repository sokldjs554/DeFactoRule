#!/usr/bin/env python3
"""정답 누출 감사 — CLI 진입점.

구현은 `app.evaluation.leakage` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.leakage import main  # noqa: E402

if __name__ == "__main__":
    main()
