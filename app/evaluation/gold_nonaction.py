"""비조치의견서 평가셋을 만든다.

이 평가셋의 값어치는 **정답을 모델이 만들지 않았다**는 데 있다. 라벨은 금융감독원이
문서 헤더의 체크박스에 직접 찍은 것이다. LLM 이 만든 정답으로 LLM 을 채점하는
순환을 피할 수 있는 유일한 자산이다.

과제 정의 — 요청대상행위만 보고 결론을 맞힌다.

    입력  요청대상행위 (신청인이 무엇을 하려는가)
    출력  비조치 / 조치 / 기타

`판단` 과 `판단이유` 는 입력에서 뺀다. 실측 결과 `판단이유` 는 252건 중 145건
(58%)이 결론 문구를 그대로 담고 있다. 그걸 넣고 정확도를 재면 모델 성능이 아니라
문자열 복사 능력을 재게 된다.

`요청대상행위` 에도 17건에 결론 문구가 섞여 있어 가린다. 6.7% 는 무시할 만해
보이지만, 소수 클래스가 22건뿐이라 몇 건만 새도 성능이 통째로 달라진다.

    python scripts/make_nonaction_gold.py --input data/processed/cases_nonaction.jsonl \
        --output data/eval
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.domain.labels import NON_ACTIONS

# 결론을 직접 드러내는 표현. 입력에서 가린다.
#
# 처음에는 "비조치의견을 제시" 같은 구를 잡았는데, dev 의 클래스 특징어를 뽑아
# 보니 비조치 클래스 상위에 낱말 "비조치를" 이 6회로 남아 있었다. 요청문에
# 등장하는 "비조치" 라는 낱말 자체가 정답과 상관을 갖는다.
#
# 신청인은 어느 건이든 비조치를 요청하므로 이 낱말은 원리적으로 결론을 가릴
# 정보를 주지 못한다. 그런데도 클래스별 출현이 갈린다면 그것은 신호가 아니라
# 누출이다. 낱말 단위로 통째로 가린다.
# 뒤따르는 조사까지 함께 삼킨다. "비조치를 요청" 에서 낱말만 지우면 "를요청"
# 이라는 흔적이 남고, 그 흔적이 다시 클래스 신호가 된다.
LEAK = re.compile(
    r"(비조치|조치하지\s*않|제재하지\s*않|"
    r"조치(?:를)?\s*(?:취하|하)(?:지\s*않|기로))\s*(?:를|을|이|가|의|에|은|는)?\s*"
)
# 표시를 남기지 않고 지운다.
#
# 처음에는 "[결론표현]" 으로 치환했는데, 그러자 그 토큰이 비조치 클래스에서만
# 23회 나타나 마스크 자체가 새 누출이 됐다. 무엇을 가렸다는 사실을 남기면
# "이 요청문에 그 낱말이 있었다" 는 정보가 그대로 보존된다.
MASK = " "

# 조판 잔재와 항목명 머리글자
JUNK = re.compile("[\\x00-\\x08\\x0b-\\x1f\\x7f\\u00ad\\u200b-\\u200f\\u2244\\ufeff]")
LEADING_FIELD = re.compile(r"^\s*행위\s*\n")

# dev 는 규칙을 쓰면서 들여다봐도 되는 몫, test 는 끝까지 건드리지 않는다.
DEV_EVERY = 3


def clean(text: str) -> str:
    text = JUNK.sub("", text or "")
    text = LEADING_FIELD.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def build(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    labeled = [r for r in rows if r.get("decision") in NON_ACTIONS]
    # 정렬을 고정해 난수 없이 재현되게 한다
    labeled.sort(key=lambda r: (r["source"], r["page"], r["serial"] or ""))

    items = []
    for r in labeled:
        text = clean(r["fields"].get("요청대상행위", ""))
        masked, n_leaks = LEAK.subn(MASK, text)
        items.append(
            {
                "source": r["source"],
                "serial": r["serial"],
                "page": r["page"],
                "pair_index": 1,
                "sector": r.get("sector"),
                "request": masked,
                "label": r["decision"],
                "label_source": "document_checkbox",
                "masked_leaks": n_leaks,
            }
        )

    dev = [x for i, x in enumerate(items) if i % DEV_EVERY == 0]
    test = [x for i, x in enumerate(items) if i % DEV_EVERY != 0]
    return dev, test


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict]) -> None:
    dist = Counter(r["label"] for r in rows)
    leaks = sum(r["masked_leaks"] for r in rows)
    empty = sum(1 for r in rows if not r["request"])
    print(f"\n{name} {len(rows)}건 · 가린 결론표현 {leaks} · 빈 요청 {empty}")
    for label in NON_ACTIONS:
        n = dist[label]
        print(f"  {label}: {n} ({n / len(rows):.1%})" if rows else f"  {label}: 0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="디렉토리")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev, test = build(rows)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "nonaction_dev.jsonl", dev)
    write(out / "nonaction_test.jsonl", test)

    summarize("dev  (규칙 작성용)", dev)
    summarize("test (최종 보고용 — 건드리지 않는다)", test)
    print(f"\n-> {out / 'nonaction_dev.jsonl'}")
    print(f"-> {out / 'nonaction_test.jsonl'}")


if __name__ == "__main__":
    main()
