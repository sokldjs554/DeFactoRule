#!/usr/bin/env python3
"""낡은 본문으로 만들어진 예측 제거 — CLI 진입점.

구현은 `app.evaluation.prune_stale` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.prune_stale import main  # noqa: E402

if __name__ == "__main__":
    main()
