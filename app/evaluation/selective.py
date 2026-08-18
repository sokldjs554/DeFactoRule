"""위험-커버리지 곡선 — 기권을 허용했을 때의 공정한 비교.

E1 에서 "LLM 은 커버리지 72.9%에서 매크로 F1 0.819" 와 "규칙은 커버리지 100%에서
0.494" 를 나란히 적었다. **그 비교는 성립하지 않는다.** 답하기 쉬운 것만 골라
답하면 점수는 당연히 오른다. 커버리지가 다르면 점수를 비교할 수 없다.

바로잡는 방법은 두 가지다.

  같은 커버리지에서 비교   각 모델이 같은 비율만 답하도록 맞춘 뒤 성능을 본다
  곡선 전체를 비교         커버리지를 0%에서 100%까지 훑으며 그린 곡선을 겹쳐 본다

곡선 하나로 요약한 값이 **AURC**(Area Under the Risk-Coverage curve)다. 위험(오류율)의
평균이므로 **낮을수록 좋다.** 기권 신호가 전혀 없는 모델은 곡선이 평평해지고
AURC 가 전체 오류율과 같아진다 — 그 평평함 자체가 "자기가 틀릴 때를 모른다"는
진단이다.

    python scripts/risk_coverage.py --gold data/eval/nonaction_test.jsonl \\
        --labels nonaction \\
        --pred keyword=data/processed/pred_nonaction_keyword.jsonl \\
        --pred llm=data/processed/pred_nonaction_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.core.io import key_of, load_jsonl
from app.domain.labels import LABEL_SETS
from app.evaluation.metrics import macro_f1

# 신뢰도 문자열의 순서. 높을수록 먼저 채택된다.
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "?": 0}


def rank_of(pred: dict) -> int:
    return CONFIDENCE_RANK.get(pred.get("confidence", "?"), 0)


def operating_points(
    items: list[tuple[str, str, int]], labels: tuple[str, ...]
) -> list[dict]:
    """신뢰도 내림차순으로 채택 범위를 넓히며 (커버리지, 위험, F1) 을 만든다.

    같은 신뢰도끼리는 쪼갤 수 없으므로 등급 단위로만 점이 생긴다. 세 등급이면
    점이 셋뿐이고, 그건 이 신호가 거친 탓이지 계산의 한계가 아니다.
    """
    ranks = sorted({r for _, _, r in items}, reverse=True)
    total = len(items)
    points = []
    accepted: list[tuple[str, str]] = []
    for rank in ranks:
        accepted += [(g, p) for g, p, r in items if r == rank]
        errors = sum(1 for g, p in accepted if g != p)
        points.append(
            {
                "min_confidence_rank": rank,
                "coverage": len(accepted) / total,
                "n": len(accepted),
                "risk": errors / len(accepted),
                "accuracy": 1 - errors / len(accepted),
                "macro_f1": macro_f1(accepted, labels)[0],
            }
        )
    return points


def aurc(points: list[dict]) -> float:
    """커버리지에 대한 위험의 가중 평균. 낮을수록 좋다.

    각 구간의 위험을 그 구간이 차지하는 커버리지 폭으로 가중한다.
    """
    area = 0.0
    prev_cov = 0.0
    for pt in points:
        area += pt["risk"] * (pt["coverage"] - prev_cov)
        prev_cov = pt["coverage"]
    return area / prev_cov if prev_cov else 0.0


def bar(value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--labels", choices=sorted(LABEL_SETS), default="verdict")
    ap.add_argument("--report")
    args = ap.parse_args()

    labels = LABEL_SETS[args.labels]
    gold = {key_of(r): r for r in load_jsonl(Path(args.gold)) if r.get("label")}

    models: dict[str, dict] = {}
    for spec in args.pred:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"형식은 NAME=PATH 입니다: {spec}")
        models[name] = {key_of(r): r for r in load_jsonl(Path(path))}

    keys = [k for k in gold if all(k in m for m in models.values())]
    print(f"공통 표본 {len(keys)}건\n")

    curves: dict[str, list[dict]] = {}
    for name, pred in models.items():
        items = [
            (gold[k]["label"], pred[k].get("predicted", ""), rank_of(pred[k]))
            for k in keys
        ]
        curves[name] = operating_points(items, labels)

    for name, points in curves.items():
        score = aurc(points)
        flat = len(points) == 1
        note = "  ← 기권 신호 없음 (곡선이 한 점)" if flat else ""
        print(f"■ {name}   AURC {score:.3f}{note}")
        print(f"  {'커버리지':>8}  {'건수':>4}  {'위험':>6}  {'정확도':>7}  {'매크로 F1':>9}")
        for pt in points:
            print(
                f"  {pt['coverage']:>8.1%}  {pt['n']:>4}  {pt['risk']:>6.1%}  "
                f"{pt['accuracy']:>7.1%}  {pt['macro_f1']:>9.3f}  {bar(pt['accuracy'])}"
            )
        print()

    # ── 같은 커버리지에서 비교 ────────────────────────────────────
    print("─" * 62)
    print("같은 커버리지에서의 비교\n")
    print("각 모델이 도달 가능한 커버리지 중 서로 가장 가까운 점끼리 맞춰 본다.")
    print("커버리지가 다르면 정확도를 비교할 수 없다 — 쉬운 것만 골라 답하면")
    print("점수는 저절로 오르기 때문이다.\n")

    names = list(curves)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            print(f"▸ {a} vs {b}")
            # 한쪽의 운영점마다 다른 쪽에서 가장 가까운 점을 찾는다.
            # 가장 가까운 한 쌍만 보면 100% 지점끼리 맞아떨어져, 정작 흥미로운
            # 저커버리지 구간을 통째로 놓친다.
            seen: set[tuple[int, int]] = set()
            for source, other, flip in ((a, b, False), (b, a, True)):
                for ps in curves[source]:
                    po = min(
                        curves[other],
                        key=lambda q: abs(q["coverage"] - ps["coverage"]),
                    )
                    pair = (
                        (po["n"], ps["n"]) if flip else (ps["n"], po["n"])
                    )
                    if pair in seen:
                        continue
                    seen.add(pair)
                    left, right = (po, ps) if flip else (ps, po)
                    gap = abs(left["coverage"] - right["coverage"])
                    flag = "  ⚠ 커버리지 차이 큼" if gap > 0.10 else ""
                    print(
                        f"    {a:>9} {left['coverage']:>6.1%} 위험 {left['risk']:>5.1%}"
                        f"  │  {b:>9} {right['coverage']:>6.1%} 위험 {right['risk']:>5.1%}"
                        f"{flag}"
                    )
            print()

    # ── AURC 차이 검정 ───────────────────────────────────────────
    if len(names) >= 2:
        print("─" * 62)
        print("AURC 차이 — 대응표본 부트스트랩 (2,000회)\n")
        rng = random.Random(0)
        n = len(keys)
        draws = [[rng.randrange(n) for _ in range(n)] for _ in range(2000)]
        item_cache = {
            name: [
                (gold[k]["label"], models[name][k].get("predicted", ""), rank_of(models[name][k]))
                for k in keys
            ]
            for name in names
        }
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                diffs = []
                for idx in draws:
                    sa = [item_cache[a][j] for j in idx]
                    sb = [item_cache[b][j] for j in idx]
                    diffs.append(
                        aurc(operating_points(sa, labels))
                        - aurc(operating_points(sb, labels))
                    )
                diffs.sort()
                lo = diffs[int(0.025 * len(diffs))]
                hi = diffs[int(0.975 * len(diffs)) - 1]
                observed = aurc(curves[a]) - aurc(curves[b])
                verdict = "유의" if lo > 0 or hi < 0 else "판정 보류"
                print(
                    f"  {a} - {b}: {observed:+.3f}  "
                    f"[95% CI {lo:+.3f}–{hi:+.3f}]  {verdict}"
                )
        print("\n  AURC 는 위험의 평균이므로 음수일수록 앞쪽 모델이 낫다.\n")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n": len(keys),
                    "aurc": {name: aurc(pts) for name, pts in curves.items()},
                    "curves": curves,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"-> {out}")


if __name__ == "__main__":
    main()
