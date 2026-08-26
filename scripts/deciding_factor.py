#!/usr/bin/env python3
"""C-4 S5 Deciding-Factor 5건 검증 CLI.

    python3 scripts/deciding_factor.py       # dry-run, API 0회
    python3 scripts/deciding_factor.py --go  # 고정 5건 각 1회
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.deciding_factor_run import main  # noqa: E402

if __name__ == "__main__":
    main()
