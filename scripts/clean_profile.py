#!/usr/bin/env python3
"""clean split 결정론 프로파일 — CLI 진입점.

구현은 `app.evaluation.clean_profile` 에 있다. API 를 부르지 않는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.clean_profile import main  # noqa: E402

if __name__ == "__main__":
    main()
