"""사안 무리 단위 분할 — **행이 아니라 사안으로 자른다.**

## 왜 이것이 필요한가

기존 분할(`gold_nonaction.build`)은 정렬 후 `i % 3` 으로 **행**을 잘랐다.
난수가 없어 재현은 완벽했지만, **같은 사안이 여러 행일 수 있다는 것을 몰랐다.**
그래서 test 27건이 dev 에 요청문이 같은 선례를 갖게 됐다 — 시험 문제가
참고서에 실린 것이다(`docs/22`).

대부분은 부동산PF·저축은행 한시 규제유예처럼 **같은 요청이 기한을 갱신하며
해마다 다시 실린** 것이다. 행을 지우면 이 코퍼스의 가장 큰 사안 무리가
통째로 사라지므로, **지우지 않고 무리째 한쪽으로 보낸다.**

## 무엇을 바꾸고 무엇을 그대로 두는가

    바꾸는 것   자르는 단위 — 행 -> 사안 무리
    그대로     난수 없음 · 결정론 · 요청문 텍스트 · 라벨 · 누출 마스킹

**기존 분할 파일을 덮어쓰지 않는다.** 새 이름으로 낸다(`*_clean.jsonl`).
legacy 는 비교 기준으로 남아야 한다.

## 행의 식별키

`serial` 하나로는 행을 가릴 수 없다. 일련번호 `230014` 가 **같은 사례집
61쪽과 67쪽에 두 번** 있다(둘 다 `기타`). 그래서 식별키는
**`(source, page, serial)`** 이다. 정렬·조회·중복 검사 전부 이것을 쓴다.

## 무리 정의 — G3

후보 다섯을 재보고 골랐다(`docs/23`).

    G1 원문 exact          27쌍 중 10 포착 — 부족
    G2 정규화 exact        26 포착 — 1쌍이 남는다
    G3 G2 + 날짜·일련번호   **27/27** ← 이것
    G4 G3 + 숫자 정규화     G3 과 무리 수가 같다. 효과 0 인 규칙은 넣지 않는다
    G5 유사도 ≥ TRUST      과병합. 일부러 남긴 표면유사 쌍까지 지운다

G2 로 안 잡히던 1쌍(`250058/240090`)은 **요청문 안에 기한이 적혀 있어서**
날짜를 정규화해야 붙는다.

## 층화

`조치` 는 255건 중 22건뿐이라 무리 하나가 어긋나면 비율이 흔들린다. 무리의
**가장 희소한 라벨**로 층화해 그것을 막는다. 층화 없이 자르면 test 의 `조치`
가 13건이 되고, 층화하면 14건으로 기존과 같아진다.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.core.text import normalize_for_match

# 무리를 지을 때 지우는 표기. `docs/20 §3.2` 의 선언 목록과 같은 성격이다.
SERIAL_MARK = re.compile(r"일련번호\s*[:：]?\s*\d{6}")
DATE_MARKS = (
    re.compile(r"20\d{2}\s*[.\-년]\s*\d{1,2}\s*[.\-월]?\s*(?:말)?\s*\d{0,2}\s*[.일]?"),
    re.compile(r"[''’‘]\s?\d{2}\s*\.\s*\d{1,2}\s*\.\s*(?:\d{1,2}\s*\.)?"),
    re.compile(r"\d{1,2}\s*월\s*말"),
)

# 희소한 것부터. 층화가 이 순서로 배분한다.
LABEL_RANK = {"조치": 0, "기타": 1, "비조치": 2}

DEV_EVERY = 3      # 기존 분할과 같은 주기. 바꾸는 것은 단위이지 비율이 아니다


def row_key(row: dict) -> tuple:
    """행의 식별키. **`serial` 하나로는 안 된다** — 230014 가 두 번 있다."""
    return (str(row.get("source") or ""), str(row.get("page") or ""),
            str(row.get("serial") or ""))


def group_key(row: dict) -> str:
    """같은 사안인가. 일련번호·날짜 표기를 지운 뒤 정규화해 비교한다."""
    text = SERIAL_MARK.sub(" ", row.get("request") or "")
    for pattern in DATE_MARKS:
        text = pattern.sub(" 〈기한〉 ", text)
    return normalize_for_match(text)


def build_groups(rows: list[dict]) -> dict[str, list[dict]]:
    """무리를 짓는다. 무리 안의 행은 식별키로 정렬해 순서를 고정한다."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    for members in groups.values():
        members.sort(key=row_key)
    return dict(groups)


def stratum(members: list[dict]) -> int:
    """무리의 층 — 그 안에 있는 **가장 희소한 라벨**."""
    return min(LABEL_RANK.get(m.get("label"), len(LABEL_RANK)) for m in members)


def split(rows: list[dict], every: int = DEV_EVERY) -> tuple[list[dict], list[dict]]:
    """무리 단위로 나눈다. **한 무리는 반드시 한쪽에만 있다.**

    난수를 쓰지 않는다. 같은 입력이면 같은 결과다 — 기존 분할이 잘한 유일한
    점이고 그대로 가져간다.
    """
    groups = build_groups(rows)
    buckets: dict[int, list[list[dict]]] = defaultdict(list)
    for members in groups.values():
        buckets[stratum(members)].append(members)

    dev: list[dict] = []
    test: list[dict] = []
    for level in sorted(buckets):
        ordered = sorted(buckets[level], key=lambda ms: row_key(ms[0]))
        for index, members in enumerate(ordered):
            (dev if index % every == 0 else test).extend(members)
    dev.sort(key=row_key)
    test.sort(key=row_key)
    return dev, test


def shared_groups(dev: list[dict], test: list[dict]) -> list[str]:
    """양쪽에 걸친 무리. **비어 있어야 한다** — 이 분할의 불변식이다."""
    return sorted({group_key(r) for r in dev} & {group_key(r) for r in test})


def main() -> None:
    import argparse
    from pathlib import Path

    from app.core.io import load_jsonl
    from app.core.paths import EVAL

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(EVAL / "nonaction_dev.jsonl"))
    ap.add_argument("--test", default=str(EVAL / "nonaction_test.jsonl"))
    ap.add_argument("--out-dev", default=str(EVAL / "nonaction_dev_clean.jsonl"))
    ap.add_argument("--out-test", default=str(EVAL / "nonaction_test_clean.jsonl"))
    ap.add_argument("--write", action="store_true",
                    help="파일을 쓴다. 없으면 요약만 보여 준다.")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(Path(args.dev)) + load_jsonl(Path(args.test))
            if r.get("label")]
    dev, test = split(rows)
    groups = build_groups(rows)
    crossing = shared_groups(dev, test)

    print(f"입력 {len(rows)}건 (legacy dev+test) · 무리 {len(groups)}개")
    print(f"clean dev {len(dev)} · clean test {len(test)}")
    print(f"양쪽에 걸친 무리 {len(crossing)}개  (0 이어야 한다)")
    if crossing:
        raise SystemExit("불변식 위반 — 무리가 갈렸습니다. 쓰지 않았습니다.")

    if not args.write:
        print("\n아직 쓰지 않았습니다. --write 를 붙이면 씁니다.")
        return

    for path_text, out in ((args.out_dev, dev), (args.out_test, test)):
        path = Path(path_text)
        if path.name in {"nonaction_dev.jsonl", "nonaction_test.jsonl"}:
            raise SystemExit(f"legacy 파일을 덮어쓰려 했습니다: {path.name}")
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in out:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"-> {path}  ({len(out)}건)")
