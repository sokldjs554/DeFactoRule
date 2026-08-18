#!/usr/bin/env python3
"""회답 근거 구조화 — CLI 진입점.

    python3 scripts/criteria.py extract --dry-run ...
    python3 scripts/criteria.py apply   --dry-run ...

구현은 `app.agents.criteria` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.criteria import main  # noqa: E402

if __name__ == "__main__":
    main()
