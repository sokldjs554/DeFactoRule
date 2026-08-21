#!/usr/bin/env python3
"""E11b 실측 — 사전 등록한 5건만, 각 1회씩. CLI 진입점.

    python3 scripts/applicability.py            # 계획만 (호출 없음)
    python3 scripts/applicability.py --go       # 실제 5회 호출

구현은 `app.agents.applicability_run` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.applicability_run import main  # noqa: E402

if __name__ == "__main__":
    main()
