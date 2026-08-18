"""점수 계산. 어떤 모델도 자기 점수를 매기지 않도록 여기 한 곳에만 둔다.

대표 지표를 정확도가 아니라 **매크로 F1** 으로 잡은 이유는 클래스 불균형이다.
test 170건의 74.1%가 비조치라, 아무 생각 없이 다수 클래스만 찍어도 정확도
74%가 나온다. 그 선을 모르면 LLM 이 75%를 내고도 잘한 줄 알게 된다.
"""

from __future__ import annotations

import random


def macro_f1(
    pairs: list[tuple[str, str]], labels: tuple[str, ...]
) -> tuple[float, dict[str, dict]]:
    """라벨별 P/R/F1 과 매크로 평균. 클래스 불균형이 심하므로 매크로로 본다."""
    per: dict[str, dict] = {}
    for label in labels:
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
    present = [v for v in labels if per[v]["support"]]
    macro = sum(per[v]["f1"] for v in present) / len(present) if present else 0.0
    return macro, per


def bootstrap_macro_f1(
    pairs: list[tuple[str, str]],
    labels: tuple[str, ...],
    rounds: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """매크로 F1 의 95% 부트스트랩 구간.

    표본이 작으면 점추정 하나만 보고 순위를 매기게 된다. 소수 클래스가 3건인
    상태에서 재현율은 0, 1/3, 2/3, 1 네 값밖에 못 가지므로, 한 건이 뒤집히면
    F1 이 0.1 이상 움직인다. 구간을 함께 적어 그 불확실성을 드러낸다.

    난수는 고정 시드로 재현한다.
    """
    if len(pairs) < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(pairs)
    scores = []
    for _ in range(rounds):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        scores.append(macro_f1(sample, labels)[0])
    scores.sort()
    lo = scores[int(0.025 * rounds)]
    hi = scores[int(0.975 * rounds) - 1]
    return lo, hi
