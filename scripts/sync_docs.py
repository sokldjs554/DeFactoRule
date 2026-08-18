#!/usr/bin/env python3
"""문서의 파생 수치를 산출물에서 다시 써 넣는다 — CLI 진입점.

구현은 `app.evaluation.doc_sync` 에 있다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.doc_sync import main  # noqa: E402

if __name__ == "__main__":
    main()
