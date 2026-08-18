"""평가용 gold set 표본을 뽑는다.

무엇을 만들든 정답이 없으면 성능을 말할 수 없다. 비조치의견서는 체크박스가
정답이지만, 법령해석 회신문은 사람이 읽고 붙여야 한다.

표본은 **층화 추출**한다. 무작위로 뽑으면 다수 업권(공통·자본시장)이 표본을
독식하고, 소수 업권에서의 성능을 전혀 알 수 없게 된다. 연도도 섞는다 —
서식과 문체가 해마다 달랐다는 것을 이미 확인했기 때문이다.

난수를 쓰지 않는다. 층 안에서 결정론적으로 고르므로 몇 번을 돌려도 같은 표본이
나오고, 라벨링 작업이 헛돌지 않는다.

    python scripts/make_gold_set.py --input data/processed/qa_pairs.jsonl \
        --output data/eval/gold_unlabeled.jsonl --size 120
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from labels import GUIDELINE, VERDICTS


def year_of(source: str) -> str:
    m = re.search(r"20\d\d", source)
    return m.group() if m else "unknown"


def stratify(rows: list[dict], size: int) -> list[dict]:
    """(연도 × 업권) 층에서 고르게 뽑는다.

    각 층에서 뽑는 수는 층 크기에 비례하되 최소 1건은 보장한다. 소수 업권이
    표본에서 통째로 빠지면 그 업권의 성능을 영원히 모른다.
    """
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        strata[(year_of(r["source"]), r.get("sector") or "미분류")].append(r)

    total = len(rows)
    picked: list[dict] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: (r["serial"] or "", r["pair_index"]))
        want = max(1, round(size * len(group) / total))
        # 층 안에서 균등 간격으로 — 앞쪽(=문서 앞부분)만 뽑히지 않게 한다
        step = max(1, len(group) // want)
        picked.extend(group[::step][:want])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--doc-type", default="interpretation")
    ap.add_argument("--size", type=int, default=120)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets = [
        r
        for r in rows
        if r["doc_type"] == args.doc_type
        and r.get("answer", "").strip()
        and r.get("question", "").strip()
    ]
    picked = stratify(targets, args.size)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(
                json.dumps(
                    {
                        "source": r["source"],
                        "serial": r["serial"],
                        "page": r["page"],
                        "pair_index": r["pair_index"],
                        "sector": r.get("sector"),
                        "question": r["question"],
                        "answer": r["answer"],
                        "label": None,  # ← 사람이 채운다
                        "labeler": None,
                        "note": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    guide = out.with_name("LABELING.md")
    guide.write_text(
        "# 라벨링 지침\n\n"
        f"`{out.name}` 의 각 행에서 `label` 을 다음 중 하나로 채운다.\n\n"
        + "".join(f"- `{v}`\n" for v in VERDICTS)
        + "\n`labeler` 에 라벨을 붙인 사람을 적고, 애매했던 이유는 `note` 에 남긴다.\n"
        "애매했던 사례의 메모가 나중에 지침을 고칠 근거가 된다.\n\n"
        "## 판정 기준\n\n" + GUIDELINE,
        encoding="utf-8",
    )

    from collections import Counter

    print(f"{len(targets)}건 중 {len(picked)}건 추출 -> {out}")
    print(f"지침 -> {guide}\n")
    print("연도")
    for k, n in sorted(Counter(year_of(r["source"]) for r in picked).items()):
        print(f"  {k}: {n}")
    print("업권")
    for k, n in Counter(r.get("sector") or "미분류" for r in picked).most_common():
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
