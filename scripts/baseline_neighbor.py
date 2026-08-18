#!/usr/bin/env python3
"""최근접 선례 기준선 — CLI 진입점.

구현은 `app.retrieval.neighbor` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.neighbor import main  # noqa: E402

if __name__ == "__main__":
    main()
