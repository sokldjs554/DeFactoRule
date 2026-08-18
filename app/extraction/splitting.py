"""사례를 질의–회답 쌍 단위로 쪼갠다.

한 사례가 ①②③ 복수 질의를 담는 경우가 있다. 학습·평가의 단위는 사례가
아니라 질의–회답 쌍이므로 여기서 분리한다.

**질의요지와 회답 양쪽에 순번이 맞아떨어질 때만 쪼갠다.**

회답에만 순번이 있다고 쪼개면 안 된다. 실제 사례를 열어 보면 이 문서들의
①②는 압도적으로 질의 구분이 아니라 요건·조건을 열거하는 표지다.

    "① 의결권 있는 발행주식 10% 이상을 소유하거나
     ② 혼자서 또는 다른 주주와 합의·계약 등으로 대표이사를 선임하는
        등의 사정이 없는 이상 대주주에 해당하지 않는 것으로 판단됩니다."

이것은 한 문장이다. 순번마다 쪼개면 결론절이 통째로 떨어져 나가고, 남은
조각은 결론이 없는 요건 나열이 된다. 회답 단독 분할을 켰을 때 81쌍이
생겼는데 표본 검토에서 전부 이런 오분할이었다.

역설적으로 이 ①② 요건 열거는 이 프로젝트가 찾으려는 판단 기준 그 자체다.
쪼개서 흩을 대상이 아니라 구조를 보존해야 할 대상이다.

    python scripts/split_queries.py --input data/processed --output data/processed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# 순번 문자 → 정수. 발간물마다 다른 글자체를 쓴다.
CIRCLED: dict[str, int] = {}
for _row in (
    "①②③④⑤⑥⑦⑧⑨⑩",
    "➀➁➂➃➄➅➆➇➈➉",
    "❶❷❸❹❺❻❼❽❾❿",
    "➊➋➌➍➎➏➐➑➒➓",
    "⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽",
):
    for _i, _ch in enumerate(_row):
        CIRCLED[_ch] = _i + 1

# "(질의1)", "질의 2." 처럼 글자로 쓰는 서식도 있다
RE_TEXT_SEQ = re.compile(r"[(<\[]?\s*질\s*의\s*(\d)\s*[.):>\]]")


def find_marks(text: str) -> list[tuple[int, int]]:
    """(문자 위치, 순번) 목록을 순번이 증가하는 구간만 남겨 돌려준다.

    본문 인용구 안의 "①" 같은 것이 섞이면 순번이 뒤로 갔다가 돌아온다.
    1부터 시작해 단조증가하는 첫 사슬만 유효한 질의 구분으로 본다.
    """
    raw: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if ch in CIRCLED:
            raw.append((i, CIRCLED[ch]))
    for m in RE_TEXT_SEQ.finditer(text):
        raw.append((m.start(), int(m.group(1))))
    raw.sort()

    chain: list[tuple[int, int]] = []
    expected = 1
    for pos, num in raw:
        if num == expected:
            chain.append((pos, num))
            expected += 1
    return chain if len(chain) >= 2 else []


def slice_by_marks(text: str, marks: list[tuple[int, int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for idx, (pos, num) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        out[num] = text[pos:end].strip()
    return out


def split_case(case: dict) -> list[dict]:
    fields = case.get("fields", {})
    question = fields.get("질의요지") or fields.get("요청대상행위") or ""
    answer = fields.get("회답") or fields.get("판단") or ""

    q_marks = find_marks(question)
    a_marks = find_marks(answer)

    base = {
        "source": case["source"],
        "doc_type": case["doc_type"],
        "serial": case["serial"],
        "sector": case.get("sector"),
        "subsector": case.get("subsector"),
        "page": case["page"],
        "decision": case.get("decision"),
        "reason": fields.get("이유") or fields.get("판단이유") or "",
        "case_warnings": case.get("warnings", []),
    }

    # 양쪽 순번이 **정확히 같을 때만** 짝을 지어 자른다.
    #
    # 처음에는 교집합이 2개 이상이면 잘랐다. 그러자 질의 ➊~➎ · 회답 󰊱󰊲 인
    # 사례에서 {1,2} 로 짝을 지었는데, 질의의 ➊~➎ 는 다섯 개 질문이 아니라
    # **한 질문 안의 행위 열거**였고 회답의 󰊱󰊲 는 그것을 두 묶음으로 나눈
    # 답변이었다. 질의① "보관" 과 회답① "➌현금을 영수" 를 붙인 셈이다.
    # 짝이 어긋난 데다 질의 ➌➍➎ 는 경고도 없이 사라졌다.
    #
    # 회답에만 순번이 있을 때 자르지 않기로 한 것과 같은 이유다 — 순번은
    # 질의 구분일 수도 있고 열거일 수도 있으며, 양쪽이 정확히 대응할 때만
    # 구분이라고 믿을 수 있다.
    if q_marks and a_marks:
        q_parts, a_parts = slice_by_marks(question, q_marks), slice_by_marks(answer, a_marks)
        shared = sorted(set(q_parts) & set(a_parts))
        aligned = set(q_parts) == set(a_parts)
        if aligned and len(shared) >= 2:
            return [
                {
                    **base,
                    "pair_index": n,
                    "pair_count": len(shared),
                    "split_mode": "paired",
                    "has_enumeration": True,
                    "question": q_parts[n],
                    "answer": a_parts[n],
                }
                for n in shared
            ]

    # 회답에만 순번이 있는 경우는 쪼개지 않는다. 요건 열거일 가능성이 압도적이다.
    # 다만 그 사실을 표시해 두어 다운스트림에서 열거 구조를 살릴 수 있게 한다.
    enumerated = bool(a_marks) or bool(q_marks)

    warnings = list(base["case_warnings"])
    if q_marks and a_marks and {n for _, n in q_marks} != {n for _, n in a_marks}:
        # 조용히 넘어가지 않는다. 양쪽에 순번이 있는데 대응하지 않는다는 것은
        # 한쪽이 열거라는 뜻이고, 그 사실 자체가 기록할 가치가 있다.
        warnings.append("mark_mismatch")

    return [
        {
            **base,
            "case_warnings": warnings,
            "pair_index": 1,
            "pair_count": 1,
            "split_mode": "single",
            "has_enumeration": enumerated,
            "question": question,
            "answer": answer,
        }
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    for path in sorted(in_dir.glob("cases_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            cases.extend(json.loads(line) for line in fh if line.strip())

    pairs: list[dict] = []
    for case in cases:
        pairs.extend(split_case(case))

    out = out_dir / "qa_pairs.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter

    modes = Counter(p["split_mode"] for p in pairs)
    gained = len(pairs) - len(cases)
    print(f"사례 {len(cases)} → 질의–회답 쌍 {len(pairs)}  (+{gained})")
    for k, v in modes.most_common():
        print(f"  {k}: {v}")
    empty_q = sum(1 for p in pairs if not p["question"].strip())
    empty_a = sum(1 for p in pairs if not p["answer"].strip())
    print(f"  빈 질의 {empty_q} · 빈 회답 {empty_a}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
