"""JSONL 입출력과 레코드 키.

파이프라인의 모든 중간 산출물은 JSONL 이다. 한 줄이 한 사례이고, 줄 단위라
중간에 끊겨도 앞부분이 살아남는다 — `--resume` 이 성립하는 이유다.

**레코드 키를 여기 두는 이유**는 gold 와 예측을 맞추는 기준이 한 곳에만
있어야 하기 때문이다. 채점 스크립트마다 따로 정의하면, 키가 어긋나도 예외가
나지 않고 그냥 매칭 건수가 조용히 줄어든다. 실제로 30건 예측을 170건 기준선과
비교하면서 커버리지 17.6% 를 못 보고 지나친 적이 있다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

# 모듈 수준 별칭은 런타임에 평가되므로 `X | None` 을 쓰면 3.10 미만에서 깨진다.
# 함수 시그니처 안의 같은 표기는 `from __future__ import annotations` 덕에 안전하다.
Key = tuple[str, int, Optional[str], int]  # noqa: UP007


def key_of(row: dict) -> Key:
    """사례를 가리키는 유일 키.

    일련번호(`serial`)만으로는 부족하다. 서로 다른 사례집에서 같은 번호가
    쓰이고, 한 사례가 여러 질의로 쪼개지기도 한다. 출처·쪽·번호·쌍 번호를
    모두 써야 유일해진다.
    """
    return (row["source"], row["page"], row["serial"], row["pair_index"])


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """부모 디렉터리를 만들고 줄 단위로 쓴다. 쓴 줄 수를 돌려준다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
