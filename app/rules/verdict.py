"""규칙 기반 baseline 분류기.

LLM 을 붙이기 전에 넘어야 할 선을 만든다. 이 baseline 이 이미 충분히 잘한다면
LLM 을 쓸 이유를 설명할 수 없고, 못한다면 그 격차가 LLM 의 기여분이 된다.

정규식 몇 개로 만든 것이지만 진지하게 만든다. 허수아비 baseline 을 세워 놓고
이기는 것은 아무것도 증명하지 못한다.

    python scripts/baseline_rules.py --input data/processed/qa_pairs.jsonl \
        --output data/processed/pred_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.domain.labels import Verdict

# baseline 전용 결과값. 규칙이 아무것도 잡지 못한 상태를 `판단유보` 로 적으면
# "당국이 판단을 유보한 것" 과 "규칙이 못 읽은 것" 이 한 라벨에 섞인다.
# 그 둘은 평가에서 전혀 다르게 다뤄야 하므로 분리한다.
UNKNOWN = "미분류"

# 결론절은 회답 끝쪽에 온다. 앞쪽 인용문에 낚이지 않도록 뒤에서부터 본다.
TAIL_CHARS = 300

# 판단유보 — 가장 먼저 본다. 다른 표현과 겹쳐 나와도 유보가 우선한다.
RE_ABSTAIN = re.compile(
    r"(개별적으로\s*판단|구체적\s*사실관계|사실관계에\s*따라|일률적으로\s*(말하기|판단하기)|"
    r"답변\s*(드리기\s*)?어렵|소관\s*(사항\s*)?이\s*아니|판단할\s*사항이\s*아니|"
    r"해당\s*기관에\s*문의|별도\s*검토가\s*필요)"
)

# 조건부 — 요건이 붙은 허용
RE_CONDITIONAL = re.compile(
    r"((?:요건|조건)을?\s*(?:갖춘|충족(?:하는|한)|모두\s*충족)[^.]{0,40}?"
    r"(?:경우에\s*한(?:하여|해)|한(?:하여|해)|경우에는)|"
    r"다만[^.]{0,60}?(?:경우에\s*한|한(?:하여|해)\s*(?:가능|허용))|"
    r"(?:하는|한)\s*경우에\s*한(?:하여|해)\s*(?:가능|허용|인정))"
)

RE_DENY = re.compile(
    r"(해당하지\s*(?:않는|아니)|포함되지\s*(?:않는|아니)|볼\s*수\s*없|"
    r"허용되지\s*(?:않|아니)|가능하지\s*(?:않|아니)|할\s*수\s*없|"
    r"위반(?:에\s*해당|됩니다|입니다)|불가(?:능)?(?:합니다|하다|한\s*것))"
)

RE_AFFIRM = re.compile(
    r"(해당(?:하는\s*것으로\s*(?:보|판단)|합니다|됩니다|한다고\s*보)|"
    r"포함(?:되는\s*것으로|됩니다)|가능(?:합니다|하다고|한\s*것으로)|"
    r"허용(?:됩니다|되는\s*것으로|할\s*수)|볼\s*수\s*있|"
    r"할\s*수\s*있(?:습니다|는\s*것으로)|무방(?:합니다|하다))"
)


def classify(answer: str) -> tuple[str, str]:
    """(라벨, 근거가 된 패턴 이름) 을 돌려준다.

    근거를 함께 남기는 것은 오류 분석 때문이다. 어떤 규칙이 어떤 오류를
    만들었는지 모르면 규칙을 고칠 수 없다.
    """
    text = " ".join(answer.split())
    if not text:
        return UNKNOWN, "empty"

    tail = text[-TAIL_CHARS:]

    # 유보가 최우선. "가능하나 개별 사실관계에 따라 다르다" 는 유보다.
    if RE_ABSTAIN.search(tail):
        return Verdict.ABSTAIN.value, "abstain"
    if RE_CONDITIONAL.search(tail):
        return Verdict.CONDITIONAL.value, "conditional"

    deny = RE_DENY.search(tail)
    affirm = RE_AFFIRM.search(tail)
    if deny and affirm:
        # 둘 다 걸리면 더 뒤에 나온 쪽이 결론절일 가능성이 높다.
        return (
            (Verdict.DENY.value, "deny>affirm")
            if deny.start() > affirm.start()
            else (Verdict.AFFIRM.value, "affirm>deny")
        )
    if deny:
        return Verdict.DENY.value, "deny"
    if affirm:
        return Verdict.AFFIRM.value, "affirm"
    return UNKNOWN, "no_match"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="qa_pairs.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--doc-type",
        default="interpretation",
        help="분류 대상 문서 종류 (비조치는 라벨이 이미 있다)",
    )
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets = [r for r in rows if r["doc_type"] == args.doc_type]

    from collections import Counter

    preds = []
    for r in targets:
        label, rule = classify(r.get("answer", ""))
        preds.append(
            {
                "source": r["source"],
                "serial": r["serial"],
                "page": r["page"],
                "pair_index": r["pair_index"],
                "predicted": label,
                "rule": rule,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = Counter(p["predicted"] for p in preds)
    covered = len(preds) - counts[UNKNOWN]
    print(f"{len(preds)}건 분류 -> {out}")
    print(f"규칙이 판정한 것 {covered} ({covered / len(preds):.1%}), "
          f"미분류 {counts[UNKNOWN]} ({counts[UNKNOWN] / len(preds):.1%})\n")
    for label, n in counts.most_common():
        print(f"  {label}: {n} ({n / len(preds):.1%})")
    print("\n적용된 규칙")
    for rule, n in Counter(p["rule"] for p in preds).most_common():
        print(f"  {rule}: {n}")


if __name__ == "__main__":
    main()
