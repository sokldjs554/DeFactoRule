"""dev 에서 기저율을 계산해 파일로 굳힌다.

E3 에서 모델이 소수 클래스를 과잉 예측한다는 것을 확인했다. `조치` 라벨은
전자금융과 공통에만 존재하고 나머지 여섯 업권에는 한 건도 없는데, 모델이 그것을
모른 채 모든 업권에서 세 라벨을 고려한다.

E4 는 그 정보를 프롬프트에 넣으면 나아지는지 본다.

**기저율은 반드시 dev 에서만 뽑는다.** test 에서 뽑아 프롬프트에 넣으면 정답을
흘리는 것이고, 그러면 실험 자체가 무의미해진다. dev(85건)와 test(170건)의 분포가
서로 다를 수 있다는 점까지 포함해서 실험이다 — 실측 차이는 비조치 기준 약 6%p 다.

표본이 적은 업권은 업권별 값 대신 전체 값을 쓴다. 3건짜리 분포를 100% 라고 적어
주면 잡음을 신호로 위장하는 셈이 된다.

    python scripts/base_rates.py --dev data/eval/nonaction_dev.jsonl \\
        --output data/eval/dev_base_rates.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from app.core.io import load_jsonl, write_json
from app.domain.labels import NON_ACTIONS

# 이보다 적으면 업권별 값을 쓰지 않는다
MIN_SECTOR_N = 5


def compute(rows: list[dict]) -> dict:
    overall = Counter(r["label"] for r in rows)
    total = len(rows)

    by_sector: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_sector[r.get("sector") or "미분류"].append(r["label"])

    sectors = {}
    for sector, labs in by_sector.items():
        counts = Counter(labs)
        sectors[sector] = {
            "n": len(labs),
            "reliable": len(labs) >= MIN_SECTOR_N,
            "rates": {lab: counts.get(lab, 0) / len(labs) for lab in NON_ACTIONS},
        }

    return {
        "source": "dev",
        "n": total,
        "min_sector_n": MIN_SECTOR_N,
        "overall": {lab: overall.get(lab, 0) / total for lab in NON_ACTIONS},
        "sectors": sectors,
    }


def describe_overall(table: dict) -> str:
    """프롬프트에 넣을 전체 기저율 문장."""
    parts = ", ".join(
        f"{lab} {table['overall'][lab]:.0%}" for lab in NON_ACTIONS
    )
    return (
        f"참고 — 과거 유사 사례 {table['n']}건에서 결론 분포는 {parts} 였습니다. "
        "이 분포는 참고용이며, 개별 사안의 내용이 분포보다 우선합니다."
    )


def describe_sector(table: dict, sector: str | None) -> str:
    """프롬프트에 넣을 업권별 기저율 문장.

    표본이 적은 업권은 전체 값으로 대체한다. 그 사실도 함께 적어 모델이
    숫자의 신뢰도를 알 수 있게 한다.
    """
    info = table["sectors"].get(sector or "")
    if not info or not info["reliable"]:
        return describe_overall(table)

    rates = info["rates"]
    parts = ", ".join(f"{lab} {rates[lab]:.0%}" for lab in NON_ACTIONS)
    absent = [lab for lab in NON_ACTIONS if rates[lab] == 0]
    note = ""
    if absent:
        note = f" 과거 사례에서 {'·'.join(absent)} 결론은 나온 적이 없습니다."
    return (
        f"참고 — 이 사안은 '{sector}' 분야입니다. 같은 분야 과거 사례 "
        f"{info['n']}건에서 결론 분포는 {parts} 였습니다.{note} "
        "이 분포는 참고용이며, 개별 사안의 내용이 분포보다 우선합니다."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.dev))
    table = compute(rows)

    out = Path(args.output)
    write_json(out, table)

    print(f"dev {table['n']}건에서 기저율 산출 -> {out}\n")
    print("전체: " + ", ".join(f"{k} {v:.1%}" for k, v in table["overall"].items()))
    print(f"\n{'업권':>12}  {'건수':>4}  {'사용':>4}  분포")
    for sector, info in sorted(
        table["sectors"].items(), key=lambda kv: -kv[1]["n"]
    ):
        use = "업권" if info["reliable"] else "전체"
        dist = " · ".join(f"{k} {v:.0%}" for k, v in info["rates"].items() if v)
        print(f"{sector:>12}  {info['n']:>4}  {use:>4}  {dist}")

    print("\n프롬프트 문장 예시:")
    print("  [전체]   " + describe_overall(table))
    print("  [업권별] " + describe_sector(table, "여신전문금융업"))


if __name__ == "__main__":
    main()
