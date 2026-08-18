"""비조치의견서 결론 예측 baseline.

두 개를 만든다. 둘째보다 첫째가 중요하다.

  majority   무조건 다수 클래스(비조치)를 찍는다
  keyword    dev 에서 뽑은 어휘 규칙

`majority` 를 반드시 먼저 세운다. test 의 74.1%가 비조치이므로 아무 생각 없이
찍어도 정확도 74%가 나온다. 이 선을 명시하지 않으면 나중에 LLM 이 75%를 내고도
"괜찮다"고 착각하게 된다.

그래서 대표 지표는 정확도가 아니라 **매크로 F1** 이다. 다수 클래스만 맞히는
분류기의 매크로 F1 은 0.28 근처로 주저앉는다.

    python scripts/baseline_nonaction.py --gold data/eval/nonaction_test.jsonl \
        --output data/processed/pred_nonaction_majority.jsonl --strategy majority
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from labels import NonAction

MAJORITY = NonAction.NO_ACTION.value

# dev 85건을 보고 뽑은 어휘. test 는 열지 않았다.
#
# 솔직히 말해 신호가 약하다. dev 에서 클래스 전용 어휘를 뽑아 보면 `조치` 쪽은
# 3회짜리 낱말 두 개가 전부다. 세 클래스의 요청문은 형태가 거의 같고
# ("~하는 것이 ~에 해당하는지 여부"), 결론은 요청문의 표면이 아니라 법적 분석의
# 결과다. 그래도 있는 대로 만들어서 그 약함을 수치로 남긴다.
RULES: list[tuple[str, re.Pattern]] = [
    # 망분리·클라우드 기술 적합성 질의는 조치로 간 사례가 몰려 있었다
    (
        NonAction.ACTION.value,
        re.compile(r"(망연계|망분리|VDI|Active\s*Directory|정보처리시스템.{0,20}연결)"),
    ),
    # 적용 범위·해당 여부를 묻는 순수 해석 질의는 기타로 처리된 경우가 있었다
    (
        NonAction.OTHER.value,
        re.compile(r"(적용\s*(?:여부|되는지)|해당\s*(?:되는지|하는지)\s*여부.{0,30}시행세칙)"),
    ),
]


def classify(request: str, strategy: str) -> tuple[str, str]:
    if strategy == "majority":
        return MAJORITY, "majority"
    text = " ".join(request.split())
    for label, pattern in RULES:
        if pattern.search(text):
            return label, f"rule:{label}"
    return MAJORITY, "fallback:majority"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strategy", choices=["majority", "keyword"], default="majority")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.gold).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    preds = []
    for r in rows:
        label, rule = classify(r["request"], args.strategy)
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

    print(f"{args.strategy}: {len(preds)}건 -> {out}")
    for label, n in Counter(p["predicted"] for p in preds).most_common():
        print(f"  {label}: {n} ({n / len(preds):.1%})")


if __name__ == "__main__":
    main()
