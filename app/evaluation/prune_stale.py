"""낡은 본문으로 만들어진 예측을 걷어낸다. 일회성 정리 도구다.

파서를 고쳐 gold 의 요청문이 바뀌었는데(EX-05), 레코드 키는 그대로라 낡은
예측이 새 gold 와 조용히 짝지어진다. 지문(`input_sha`) 이 도입되기 전에 만든
예측에는 그 사실을 알아낼 단서가 없으므로, 바뀐 키 목록을 파일로 못박아 두고
그것만 걷어낸다.

걷어낸 뒤 같은 분류 명령에 `--resume` 을 붙이면 그 건들만 다시 부른다.

    python scripts/prune_stale.py --pred data/processed/pred_nonaction_sector.jsonl
    python scripts/prune_stale.py --pred data/processed/*.jsonl --split test

지문이 도입된 뒤로는 `--resume` 이 스스로 대조하므로 이 도구가 필요 없다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.io import key_of, load_jsonl, write_jsonl
from app.core.paths import EVAL

DEFAULT_KEYS = EVAL / "stale_after_field_fix.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--keys", default=str(DEFAULT_KEYS))
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(Path(args.keys).read_text(encoding="utf-8"))
    stale = {tuple(k) for k in spec[args.split]}
    print(f"{args.split} 에서 본문이 바뀐 키 {len(stale)}개\n")

    for name in args.pred:
        path = Path(name)
        if not path.exists():
            print(f"  {path.name}: 파일 없음 — 건너뜀")
            continue
        rows = load_jsonl(path)
        keep = [r for r in rows if key_of(r) not in stale]
        dropped = len(rows) - len(keep)
        if dropped and not args.dry_run:
            write_jsonl(path, keep)
        verb = "제거 예정" if args.dry_run else "제거"
        print(f"  {path.name}: {len(rows)} → {len(keep)} (낡은 {dropped}건 {verb})")

    if not args.dry_run:
        print("\n같은 분류 명령에 --resume 을 붙이면 걷어낸 건만 다시 부릅니다.")


if __name__ == "__main__":
    main()
