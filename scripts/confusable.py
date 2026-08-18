#!/usr/bin/env python3
"""표면동형 구분율(CPA) — CLI 진입점.

구현은 `app.evaluation.confusable` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.confusable import main  # noqa: E402

if __name__ == "__main__":
    main()
