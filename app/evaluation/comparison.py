"""두 모델의 성능 차이를 대응표본 부트스트랩으로 검정한다.

각 모델의 신뢰구간을 따로 그려 놓고 "겹치니까 차이 없다"고 말하는 것은 틀렸다.
두 모델은 **같은 사례**를 채점하므로 오차가 상관되어 있고, 그 상관을 무시하면
검정력을 크게 잃는다. 같은 부트스트랩 표본에서 두 모델을 동시에 채점하고
**차이의 분포**를 본다.

모델이 늘면 비교 쌍이 제곱으로 는다. 7개면 21쌍이고, 유의수준 5%에서
21번 검정하면 **아무 차이가 없어도** 하나쯤은 유의하게 나온다(1-0.95^21 ≈ 66%).
그래서 Holm-Bonferroni 보정을 함께 낸다. Holm 은 Bonferroni 보다 검정력을
덜 잃으면서 family-wise 오류율을 같게 지킨다.

보정 전 p값도 나란히 남긴다 — 숨기면 나중에 "왜 이 결과만 사라졌지" 를
되짚을 수 없다.

    python scripts/compare_models.py --gold data/eval/nonaction_test.jsonl \\
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

ROUNDS = 5000
SEED = 0


def holm(pvalues: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Holm-Bonferroni. (보정된 p, 기각 여부) 를 원래 순서로 돌려준다.

    p 를 오름차순으로 놓고 i 번째에 (m - i) 를 곱한다. 앞선 것보다 작아지지
    않도록 누적 최대를 취하고(단조성), 처음 alpha 를 넘는 지점부터 전부
    기각하지 않는다.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return [(adjusted[i], adjusted[i] < alpha) for i in range(m)]


def aligned_pairs(
    gold: dict, preds: dict[str, dict]
) -> tuple[list, dict[str, list[str]], list[str]]:
    """모든 모델이 예측한 사례만 남긴다. 표본이 다르면 비교가 성립하지 않는다."""
    keys = [k for k in gold if all(k in p for p in preds.values())]
    truth = [gold[k]["label"] for k in keys]
    by_model = {
        name: [p[k].get("predicted", "") for k in keys] for name, p in preds.items()
    }
    return keys, by_model, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument(
        "--pred",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="모델 이름과 예측 파일. 두 번 이상 지정한다.",
    )
    ap.add_argument("--labels", choices=sorted(LABEL_SETS), default="verdict")
    ap.add_argument("--report")
    args = ap.parse_args()

    if len(args.pred) < 2:
        raise SystemExit("--pred 를 두 개 이상 지정하세요.")

    gold = {
        key_of(r): r for r in load_jsonl(Path(args.gold)) if r.get("label")
    }
    preds = {}
    for spec in args.pred:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"형식은 NAME=PATH 입니다: {spec}")
        preds[name] = {key_of(r): r for r in load_jsonl(Path(path))}

    keys, by_model, truth = aligned_pairs(gold, preds)
    if not keys:
        raise SystemExit("모든 모델이 공통으로 예측한 사례가 없습니다.")

    labels = LABEL_SETS[args.labels]
    n = len(keys)
    print(f"공통 표본 {n}건 (gold {len(gold)}건 중)\n")

    point = {
        name: macro_f1(list(zip(truth, p)), labels)[0] for name, p in by_model.items()
    }
    for name, score in sorted(point.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: 매크로 F1 {score:.3f}")

    names = list(by_model)
    rng = random.Random(SEED)
    # 같은 재표본에서 모든 모델을 함께 채점한다 — 그래야 차이가 짝을 이룬다.
    draws = [[rng.randrange(n) for _ in range(n)] for _ in range(ROUNDS)]

    results = []
    print("\n대응표본 부트스트랩 (매크로 F1 차이)")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diffs = []
            for idx in draws:
                pa = [(truth[j], by_model[a][j]) for j in idx]
                pb = [(truth[j], by_model[b][j]) for j in idx]
                diffs.append(macro_f1(pa, labels)[0] - macro_f1(pb, labels)[0])
            diffs.sort()
            lo = diffs[int(0.025 * ROUNDS)]
            hi = diffs[int(0.975 * ROUNDS) - 1]
            observed = point[a] - point[b]
            # 부호가 뒤집히는 비율 — 양측 p값의 부트스트랩 근사
            worse = sum(1 for d in diffs if d <= 0) / ROUNDS
            p = 2 * min(worse, 1 - worse)
            results.append(
                {
                    "a": a,
                    "b": b,
                    "diff": observed,
                    "ci95": [lo, hi],
                    "p_approx": p,
                    "significant_raw": bool(lo > 0 or hi < 0),
                }
            )

    # ── 다중비교 보정 ────────────────────────────────────────────
    adjusted = holm([r["p_approx"] for r in results])
    for r, (p_adj, keep) in zip(results, adjusted):
        r["p_holm"] = p_adj
        r["significant_holm"] = bool(keep)

    width = max(len(f"{r['a']} - {r['b']}") for r in results)
    print(f"  {'비교':<{width}}  {'차이':>7}  {'95% CI':>18}  {'p':>7}  {'p(Holm)':>8}  판정")
    for r in sorted(results, key=lambda x: x["p_approx"]):
        lo, hi = r["ci95"]
        if r["significant_holm"]:
            verdict = "유의"
        elif r["significant_raw"]:
            verdict = "보정 후 탈락"
        else:
            verdict = "판정 보류"
        print(
            f"  {r['a'] + ' - ' + r['b']:<{width}}  {r['diff']:>+7.3f}  "
            f"[{lo:>+7.3f}, {hi:>+7.3f}]  {r['p_approx']:>7.3f}  "
            f"{r['p_holm']:>8.3f}  {verdict}"
        )

    survived = sum(1 for r in results if r["significant_holm"])
    dropped = sum(1 for r in results if r["significant_raw"] and not r["significant_holm"])
    print(f"\n  {len(results)}쌍 중 보정 후 유의 {survived}쌍" +
          (f" · 보정에서 탈락 {dropped}쌍" if dropped else ""))
    if len(results) >= 10:
        naive = 1 - 0.95 ** len(results)
        print(f"  보정 없이 {len(results)}번 검정하면 아무 차이가 없어도 "
              f"{naive:.0%} 확률로 하나쯤 유의하게 나온다.")

    print(
        "\n주변 신뢰구간이 겹쳐도 대응표본 차이는 유의할 수 있습니다. "
        "두 모델이 같은 사례에서 함께 틀리기 때문입니다."
    )

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n": n,
                    "rounds": ROUNDS,
                    "correction": "holm-bonferroni",
                    "alpha": 0.05,
                    "point": point,
                    "comparisons": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"-> {out}")


if __name__ == "__main__":
    main()
