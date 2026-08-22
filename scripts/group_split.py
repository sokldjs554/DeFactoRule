#!/usr/bin/env python3
"""사안 무리 단위 분할 — CLI 진입점.

    python3 scripts/group_split.py            # 요약만 (파일 안 씀)
    python3 scripts/group_split.py --write    # clean split 파일 생성

구현은 `app.evaluation.group_split` 에 있다. legacy 파일은 덮어쓰지 않는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.group_split import main  # noqa: E402

if __name__ == "__main__":
    main()
