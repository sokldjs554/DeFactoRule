#!/usr/bin/env python3
"""선례 신뢰도 보정 — CLI 진입점. 구현은 `app.agents.calibration` 에 있다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.calibration import main  # noqa: E402

if __name__ == "__main__":
    main()
