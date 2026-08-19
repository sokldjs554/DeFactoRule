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

## 평활은 기저율 쪽으로 한다

처음에는 라벨마다 +1 을 더하는 라플라스 평활을 썼다. 그것은 **균등 사전분포**를
가정하는 것과 같은데, 그 결과를 다시 치우친 실제 기저율로 나누면 희귀 클래스가
증거 없이 부풀려진다. 실제로 이런 일이 벌어졌다.

    증거: 비조치 5건, 조치 0건
    가중치: 비조치 +0.112, 조치 +0.201   <- 증거가 없는 쪽이 더 크다

그래서 기저율 쪽으로 평활한다(m-estimate).

    P(label | c=yes) = (n_label + m * P(label)) / (n_yes + m)

증거가 하나도 없으면 P = 기저율이 되어 가중치가 정확히 0 이다. **모르는 것은
증거가 아니다** 를 평활에서도 지키는 것이다.

기저율 자체는 평활하지 않는다. 양쪽 평활이 어긋나면 "모두에게 yes 인 기준" 이
0 이 되지 않는다 — 그런 기준은 아무것도 가르지 못하므로 반드시 0 이어야 한다.
dev 에 한 번도 없는 라벨은 기저율이 0 이고, 그 라벨에 대해서는 **아무것도 알 수
없으므로** 가중치를 0 으로 둔다. 큰 음수를 주는 것은 없는 정보를 지어내는 것이다.

## 신뢰도

1등과 2등 점수의 차이(margin)로 정한다. 문턱은 dev 에서만 정한다.
"""

from __future__ import annotations

import math
from collections import Counter

from app.domain.labels import NON_ACTIONS

# 기저율 쪽 평활의 세기(가상 표본 수). dev 표본이 작아 0 이 나오면 로그가
# 발산하므로 평활은 필요하다. 다만 **어느 쪽으로** 평활하는지가 중요하다.
PRIOR_STRENGTH = 3.0

# 기저율은 평활하지 않는다. 위 문서의 이유.
SMOOTH = 0.0


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
    priors = {lab: (base[lab] / total if total else 0.0) for lab in labels}

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
            if priors[lab] <= 0.0:
                # dev 에 한 번도 없는 라벨. 있다 없다를 말할 근거가 없다.
                w[lab] = 0.0
                continue
            p = (dist[lab] + PRIOR_STRENGTH * priors[lab]) / (n + PRIOR_STRENGTH)
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
    # 아무 기준도 발화하지 않으면 모든 점수가 0 이다. 그때 이름순으로 이기게
    # 두면 '기타' 가 선택되는데, 그것은 판단이 아니라 자모 순서다. 증거가
    # 없을 때는 기저율이 가장 높은 결론으로 돌아가고, 신뢰도는 low 가 된다.
    priors = model.get("priors") or {}
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], -priors.get(kv[0], 0.0), kv[0]))
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
