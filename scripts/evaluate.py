"""예측을 gold set 과 대조해 점수를 낸다.

baseline 과 LLM 을 **같은 하네스**로 잰다. 서로 다른 스크립트로 잰 숫자를
나란히 놓고 비교하면 그 비교는 믿을 수 없다.

`미분류`(규칙이 아무것도 잡지 못한 상태)는 오답으로 세지 않고 **커버리지**로
따로 뺀다. 규칙 baseline 의 정직한 성적표는 "판정한 것 중 얼마나 맞혔나"와
"얼마나 판정했나" 두 숫자다. 하나로 뭉치면 둘 다 거짓말이 된다.

    python scripts/evaluate.py --gold data/eval/gold_labeled.jsonl \
        --pred data/processed/pred_baseline.jsonl --name baseline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from labels import VERDICTS

UNKNOWN = "미분류"

Key = tuple[str, int, str | None, int]


def key_of(row: dict) -> Key:
    return (row["source"], row["page"], row["serial"], row["pair_index"])


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def macro_f1(pairs: list[tuple[str, str]]) -> tuple[float, dict[str, dict]]:
    """라벨별 P/R/F1 과 매크로 평균. 클래스 불균형이 심하므로 매크로로 본다."""
    per: dict[str, dict] = {}
    for label in VERDICTS:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[label] = {
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    present = [v for v in VERDICTS if per[v]["support"]]
    macro = sum(per[v]["f1"] for v in present) / len(present) if present else 0.0
    return macro, per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--report", help="결과를 JSON 으로 저장할 경로")
    args = ap.parse_args()

    gold_rows = [r for r in load(Path(args.gold)) if r.get("label")]
    if not gold_rows:
        raise SystemExit(
            f"{args.gold} 에 라벨이 없습니다. data/eval/LABELING.md 를 보고 "
            "`label` 을 채운 뒤 다시 실행하세요."
        )
    gold = {key_of(r): r for r in gold_rows}
    pred = {key_of(r): r for r in load(Path(args.pred))}

    missing = [k for k in gold if k not in pred]
    matched = [k for k in gold if k in pred]

    unknown = [k for k in matched if pred[k].get("predicted") == UNKNOWN]
    errored = [k for k in matched if "error" in pred[k]]
    scored = [
        k for k in matched if k not in set(unknown) | set(errored)
    ]

    pairs = [(gold[k]["label"], pred[k]["predicted"]) for k in scored]
    accuracy = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0
    macro, per = macro_f1(pairs)
    coverage = len(scored) / len(gold) if gold else 0.0

    print(f"=== {args.name} ===")
    print(f"gold {len(gold)}건 · 예측 매칭 {len(matched)} · 미매칭 {len(missing)}")
    print(f"커버리지 {coverage:.1%}  (미분류 {len(unknown)} · 오류 {len(errored)})")
    print(f"판정한 것 중 정확도 {accuracy:.1%} · 매크로 F1 {macro:.3f}\n")

    print(f"{'라벨':>8}  {'지원':>4}  {'정밀도':>7}  {'재현율':>7}  {'F1':>7}")
    for label in VERDICTS:
        m = per[label]
        if not m["support"]:
            continue
        print(
            f"{label:>8}  {m['support']:>4}  "
            f"{m['precision']:>7.3f}  {m['recall']:>7.3f}  {m['f1']:>7.3f}"
        )

    confusion: dict[str, Counter] = defaultdict(Counter)
    for g, p in pairs:
        confusion[g][p] += 1
    wrong = [(g, p, n) for g, c in confusion.items() for p, n in c.items() if g != p]
    if wrong:
        print("\n주요 혼동 (정답 → 예측)")
        for g, p, n in sorted(wrong, key=lambda x: -x[2])[:8]:
            print(f"  {g} → {p}: {n}")

    # LLM 예측이면 인용 대조 결과도 본다
    ungrounded = [k for k in scored if pred[k].get("evidence_grounded") is False]
    if ungrounded:
        print(
            f"\n인용 미대조 {len(ungrounded)} ({len(ungrounded) / len(scored):.1%}) "
            "— 원문에 없는 구절을 근거로 든 경우"
        )

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "name": args.name,
                    "gold_size": len(gold),
                    "matched": len(matched),
                    "coverage": coverage,
                    "unknown": len(unknown),
                    "errors": len(errored),
                    "accuracy_on_scored": accuracy,
                    "macro_f1": macro,
                    "per_label": per,
                    "ungrounded_evidence": len(ungrounded),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
