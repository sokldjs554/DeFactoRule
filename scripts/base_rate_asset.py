#!/usr/bin/env python3
"""출처를 못 박은 기저율 표 — CLI 진입점.

구현은 `app.evaluation.base_rate_asset` 에 있다. legacy 파일을 덮어쓰지 않고,
test 파일을 입력으로 받지 않는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.base_rate_asset import main  # noqa: E402

if __name__ == "__main__":
    main()
