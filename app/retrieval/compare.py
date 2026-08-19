"""검색기 비교 — **Recall@K 가 아니라 함정 구간으로 잰다.**

## 왜 이 지표인가

이 프로젝트의 물음은 "닮은 것을 잘 찾는가" 가 아니라 **"찾은 것이 실제로 판단
근거가 되는가"** 다. 두 물음은 다르다. 더 많이 찾는 검색기가 더 많이 틀릴 수
있다.

그래서 검색기마다 test 사례를 E5 와 같은 방식으로 가른다.

    순응(AGREE)      최근접 선례의 결론 = 정답    따라가면 맞는다
    함정(TRAP)       최근접 선례의 결론 ≠ 정답    따라가면 틀린다
    선례 없음         문턱 미만                   따라갈 것이 없다

선례를 그대로 따르는 전략의 함정 구간 정확도는 **정의상 0** 이다. 그러므로
검색기를 비교하는 수는 함정 구간의 **크기**다.

    함정 비율 = 함정 / (순응 + 함정)

이 값이 낮을수록 "찾았을 때 믿을 만한" 검색기다. 그리고 클래스별로 따로 본다 —
E5 가 보인 대로 `조치` 에서 검색이 눈이 멀기 때문이다.
"""

from __future__ import annotations

from collections import defaultdict

from app.domain.labels import NON_ACTIONS
from app.domain.similarity import DOUBT


def partition_by_retriever(
    retriever, precedents: list[dict], rows: list[dict], floor: float = DOUBT
) -> dict:
    """한 검색기로 test 를 순응/함정/선례없음 으로 가른다."""
    buckets = {"agree": [], "trap": [], "unanchored": []}
    for row in rows:
        hits = retriever.search(row["request"], 1)
        if not hits or hits[0][1] < floor:
            buckets["unanchored"].append({"row": row, "neighbor": None,
                                          "similarity": hits[0][1] if hits else 0.0})
            continue
        index, score = hits[0]
        neighbor = precedents[index]
        key = "agree" if neighbor["label"] == row["label"] else "trap"
        buckets[key].append({"row": row, "neighbor": neighbor, "similarity": score})
    return buckets


def summarise(buckets: dict) -> dict:
    """전체와 클래스별 함정 비율."""
    anchored = len(buckets["agree"]) + len(buckets["trap"])
    per_label: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "anchored": 0, "trap": 0})

    for key in ("agree", "trap", "unanchored"):
        for link in buckets[key]:
            label = link["row"]["label"]
            per_label[label]["n"] += 1
            if key != "unanchored":
                per_label[label]["anchored"] += 1
            if key == "trap":
                per_label[label]["trap"] += 1

    for stats in per_label.values():
        stats["anchor_rate"] = stats["anchored"] / stats["n"] if stats["n"] else 0.0
        stats["trap_rate"] = (stats["trap"] / stats["anchored"]
                              if stats["anchored"] else None)

    return {
        "agree": len(buckets["agree"]),
        "trap": len(buckets["trap"]),
        "unanchored": len(buckets["unanchored"]),
        "anchored": anchored,
        "trap_rate": len(buckets["trap"]) / anchored if anchored else None,
        "by_label": {label: dict(per_label[label]) for label in NON_ACTIONS
                     if label in per_label},
    }


def compare(retrievers, precedents: list[dict], rows: list[dict],
            corpus: list[str], floor: float = DOUBT) -> dict:
    """여러 검색기를 같은 표본 위에서 비교한다."""
    out = {}
    for retriever in retrievers:
        retriever.fit(precedents, corpus)
        out[retriever.name] = summarise(
            partition_by_retriever(retriever, precedents, rows, floor))
    return out
