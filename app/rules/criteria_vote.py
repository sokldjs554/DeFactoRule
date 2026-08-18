"""기준별 답을 라벨로 바꾸는 산술. **모델은 여기 관여하지 않는다.**

명세 §9 의 분리를 끝까지 지키는 자리다. 모델은 "이 요청이 그 기준에 해당하는가"
까지만 답하고, 그 답들을 결론으로 바꾸는 것은 dev 에서 정한 가중치의 산술이다.

모델에게 최종 라벨을 물으면 기준은 장식이 된다. 답이 왜 그렇게 나왔는지
되짚을 수 없고, 기준 하나를 빼면 무엇이 달라지는지도 알 수 없다.

## 가중치

기준 c 에 'yes' 라고 답한 dev 사례들에서 라벨 분포를 보고, 기저율 대비
로그 승산으로 가중치를 만든다.

    w(c, label) = log( P(label | c=yes) / P(label) )

기저율보다 그 라벨이 흔해지면 양수, 드물어지면 음수다. 기저율을 나눠 주므로
다수 클래스가 저절로 이기지 않는다.

'unknown' 은 0 으로 둔다 — 모르는 것은 증거가 아니다.

## 신뢰도

1등과 2등 점수의 차이(margin)로 정한다. 문턱은 dev 에서만 정한다.
"""

from __future__ import annotations

import math
from collections import Counter

from app.domain.labels import NON_ACTIONS

# 라플라스 평활. dev 표본이 작아 0 이 나오면 로그가 발산한다.
SMOOTH = 1.0


def fit(
    answers_by_key: dict,
    rows: list[dict],
    n_criteria: int,
    labels: tuple = NON_ACTIONS,
) -> dict:
    """dev 에서 기준별 가중치를 뽑는다. test 는 열지 않는다."""
    from app.core.io import key_of

    labeled = [r for r in rows if r.get("label")]
    base = Counter(r["label"] for r in labeled)
    total = len(labeled)
    priors = {lab: (base[lab] + SMOOTH) / (total + SMOOTH * len(labels)) for lab in labels}

    weights = []
    for j in range(n_criteria):
        yes_rows = [
            r for r in labeled
            if (answers_by_key.get(key_of(r)) or [None] * n_criteria)[j] == "yes"
        ]
        dist = Counter(r["label"] for r in yes_rows)
        n = len(yes_rows)
        w = {}
        for lab in labels:
            p = (dist[lab] + SMOOTH) / (n + SMOOTH * len(labels))
            w[lab] = math.log(p / priors[lab])
        weights.append({"index": j, "n_yes": n, "weights": w,
                        "distribution": dict(dist)})
    return {"priors": priors, "criteria": weights, "n_criteria": n_criteria}


def score(model: dict, answers: list[str], labels: tuple = NON_ACTIONS) -> dict:
    """답 목록을 점수로 바꾼다. 'yes' 만 증거로 센다."""
    totals = {lab: 0.0 for lab in labels}
    fired = []
    for j, answer in enumerate(answers[: model["n_criteria"]]):
        if answer != "yes":
            continue
        fired.append(j)
        for lab, w in model["criteria"][j]["weights"].items():
            totals[lab] += w
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "predicted": ranked[0][0],
        "scores": totals,
        "margin": margin,
        "fired": fired,
    }


def confidence(result: dict, high: float, medium: float) -> str:
    """근거가 하나도 없으면 low. 문턱은 dev 에서 정한다."""
    if not result["fired"]:
        return "low"
    if result["margin"] >= high:
        return "high"
    if result["margin"] >= medium:
        return "medium"
    return "low"
