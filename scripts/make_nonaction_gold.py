#!/usr/bin/env python3
"""비조치 gold set 생성 — CLI 진입점.

구현은 `app.evaluation.gold_nonaction` 에 있다. 이 파일은 저장소 루트를 임포트 경로에 얹고
그 `main()` 을 부르는 것 외에 아무 일도 하지 않는다.

`python scripts/make_nonaction_gold.py` 로 실행하면 sys.path[0] 이 scripts/ 가 되어
`app` 패키지가 보이지 않는다. 그래서 루트를 직접 넣는다. 설치 없이 돌아가는
편이 이 프로젝트에서는 중요하다 — 실행 문턱을 낮추기 위해 Python 3.9 까지
내려온 것과 같은 이유다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.gold_nonaction import main  # noqa: E402

if __name__ == "__main__":
    main()
