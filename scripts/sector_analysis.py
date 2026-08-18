"""업권별 성능을 다수 클래스 기준선 대비로 본다.

E1 에서 "전자금융 62.2%로 가장 낮다, 이 업권만 따로 프롬프트를 손볼 여지가 있다"
고 적었다. **그 진단이 틀렸다.**

업권마다 정답 분포가 다르다. 다수 클래스 비율이 51%인 업권과 80%인 업권에서
같은 정확도 숫자는 전혀 다른 의미다. 절대 정확도만 줄세우면 "어려운 구간"과
"모델이 못하는 구간"을 혼동하게 된다.

그래서 업권마다 **그 업권의 majority baseline**을 따로 세우고, 그 위로 얼마나
올렸는지(lift)를 본다. 그것이 모델이 실제로 기여한 몫이다.

    python scripts/sector_analysis.py --gold data/eval/nonaction_test.jsonl \\
        --pred data/processed/pred_nonaction_llm.jsonl --labels nonaction
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate import LABEL_SETS, key_of, load, macro_f1

MIN_N = 8  # 이보다 작은 업권은 숫자가 흔들려 줄세우기에 쓸 수 없다


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", choices=sorted(LABEL_SETS), default="verdict")
    ap.add_argument("--report")
    args = ap.parse_args()

    labels = LABEL_SETS[args.labels]
    gold = {key_of(r): r for r in load(Path(args.gold)) if r.get("label")}
    pred = {key_of(r): r for r in load(Path(args.pred))}
    rows = [(gold[k], pred[k]) for k in gold if k in pred]

    by_sector: dict[str, list] = defaultdict(list)
    for g, p in rows:
        by_sector[g.get("sector") or "미분류"].append((g, p))

    stats = []
    for sector, items in by_sector.items():
        n = len(items)
        dist = Counter(g["label"] for g, _ in items)
        base_label, base_n = dist.most_common(1)[0]
        base = base_n / n  # 이 업권의 majority baseline
        acc = sum(1 for g, p in items if g["label"] == p.get("predicted")) / n
        macro, _ = macro_f1(
            [(g["label"], p.get("predicted")) for g, p in items], labels
        )
        # 남은 여지 대비 얼마나 메웠는가. 정규화 lift 는 base 가 달라도 비교된다.
        headroom = 1 - base
        normalized = (acc - base) / headroom if headroom > 0 else 0.0
        # 정규화 lift 는 작은 표본에서 극적으로 보인다. n=12 에서 한 건만 틀려도
        # -100% 가 찍힌다. 실제 건수 차이를 함께 적어 그 착시를 막는다.
        correct = round(acc * n)
        stats.append(
            {
                "sector": sector,
                "n": n,
                "correct": correct,
                "majority_correct": base_n,
                "items_vs_majority": correct - base_n,
                "base_label": base_label,
                "base_rate": base,
                "accuracy": acc,
                "lift": acc - base,
                "normalized_lift": normalized,
                "macro_f1": macro,
                "distribution": dict(dist),
            }
        )

    big = [s for s in stats if s["n"] >= MIN_N]
    small = [s for s in stats if s["n"] < MIN_N]

    print(f"업권 {len(stats)}개 · 표본 {MIN_N}건 이상인 {len(big)}개만 줄세운다\n")
    print(
        f"{'업권':>12}  {'건수':>4}  {'다수비율':>7}  {'정확도':>7}  "
        f"{'정규화 lift':>10}  {'majority 대비':>12}"
    )
    for s in sorted(big, key=lambda x: -x["normalized_lift"]):
        delta = s["items_vs_majority"]
        mark = f"{delta:+d}건"
        print(
            f"{s['sector']:>12}  {s['n']:>4}  {s['base_rate']:>7.1%}  "
            f"{s['accuracy']:>7.1%}  {s['normalized_lift']:>+10.1%}  {mark:>12}"
        )
    if small:
        names = ", ".join(f"{s['sector']}({s['n']})" for s in small)
        print(f"\n  표본 부족으로 제외: {names}")

    print("\n  다수비율 = 그 업권에서 무조건 다수 클래스를 찍었을 때의 정확도")
    print("  정규화 lift = (정확도 − 다수비율) / (1 − 다수비율)")
    print("  majority 대비 = 다수 클래스만 찍었을 때보다 몇 건 더/덜 맞혔는가")
    print("\n  ⚠ 정규화 lift 는 작은 표본에서 과장된다. n=12 에서 한 건만 틀려도")
    print("     -100% 가 찍힌다. 반드시 오른쪽 건수와 함께 읽을 것.")

    # ── 정답 분포 ───────────────────────────────────────────────
    print(f"\n{'─' * 68}\n업권별 정답 분포 — 왜 절대 정확도로 줄세우면 안 되는가\n")
    for s in sorted(big, key=lambda x: -x["n"]):
        d = s["distribution"]
        parts = " · ".join(
            f"{lab} {d.get(lab, 0):>2}({d.get(lab, 0) / s['n']:>3.0%})" for lab in labels
        )
        print(f"  {s['sector']:>12}  {parts}")

    # ── 선택적 성능 ─────────────────────────────────────────────
    print(f"\n{'─' * 68}\n업권별 신뢰도 — 어려운 구간을 모델이 알아보는가\n")
    print(f"{'업권':>12}  {'high+medium':>12}  {'그 정확도':>9}  {'low':>5}  {'그 정확도':>9}")
    for s in sorted(big, key=lambda x: -x["n"]):
        items = by_sector[s["sector"]]
        hi = [(g, p) for g, p in items if p.get("confidence") in ("high", "medium")]
        lo = [(g, p) for g, p in items if p.get("confidence") == "low"]
        hi_acc = (
            sum(1 for g, p in hi if g["label"] == p["predicted"]) / len(hi) if hi else 0
        )
        lo_acc = (
            sum(1 for g, p in lo if g["label"] == p["predicted"]) / len(lo) if lo else 0
        )
        print(
            f"{s['sector']:>12}  {len(hi):>12}  {hi_acc:>9.1%}  "
            f"{len(lo):>5}  {lo_acc:>9.1%}" + ("" if lo else "  (low 없음)")
        )

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"min_n": MIN_N, "sectors": stats}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
