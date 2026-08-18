#!/usr/bin/env python3
"""사실상 규칙 역추출 — CLI 진입점.

구현은 `app.rules.induction` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rules.induction import main  # noqa: E402

if __name__ == "__main__":
    main()
