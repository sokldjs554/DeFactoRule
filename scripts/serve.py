#!/usr/bin/env python3
"""API 서버 실행 — CLI 진입점.

    python3 scripts/serve.py            # http://127.0.0.1:8000/docs
    python3 scripts/serve.py --port 9000

uvicorn 을 직접 써도 된다: `uvicorn app.api.main:app --reload`
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn 이 없습니다: pip3 install -r requirements.txt")

    print(f"문서: http://{args.host}:{args.port}/docs")
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
