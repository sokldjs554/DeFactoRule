"""검색기 공통 계약 — 세 방식을 **같은 자리에서** 비교하기 위해.

## 무엇을 비교하는가

검색 성능만 보지 않는다. 이 프로젝트의 물음은 다르다.

> **표면적으로 비슷한 선례가 실제로 판단 근거가 되는가?**

E5 가 이미 답의 절반을 줬다 — 문자 4-gram 코사인으로 뽑은 최근접 선례는
순응 구간에서 정확도 1.000, 함정 구간에서 0.000 이다. 그러므로 검색기를
바꿔서 함정 구간이 나아지는지가 물어야 할 것이고, Recall@K 는 그 물음에
답하지 못한다.

그래서 검색기의 평가 지표는 **TRAP 구간의 크기와 그 위에서의 정확도**다.
"""

from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    """선례 풀에서 닮은 것을 찾는다."""

    name: str

    def fit(self, precedents: list[dict], corpus: list[str]) -> Retriever:
        """선례 풀과 (표현을 만들 때 쓸) 코퍼스를 받는다."""
        ...

    def search(
        self,
        request: str,
        k: int = 5,
        candidate_indices: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """(선례 인덱스, 점수)를 내림차순으로.

        `candidate_indices`가 주어지면 **순위를 매기기 전에** 그 후보만 본다.
        temporal eligibility처럼 top-k 이전에 적용해야 하는 제약을 위한 계약이다.
        """
        ...
