"""H — 두 검색기의 순위를 섞는다 (Reciprocal Rank Fusion)."""

from __future__ import annotations

RRF_K = 60


class HybridRetriever:
    name = "H"

    def __init__(self, first, second, k: int = RRF_K) -> None:
        self.first, self.second, self.k = first, second, k
        self.name = f"H({first.name}+{second.name})"

    def fit(self, precedents: list[dict], corpus: list[str]) -> HybridRetriever:
        self.first.fit(precedents, corpus)
        self.second.fit(precedents, corpus)
        return self

    def search(
        self,
        request: str,
        k: int = 5,
        candidate_indices: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        pool = max(k * 4, 20)
        left = self.first.search(request, pool, candidate_indices)
        right = self.second.search(request, pool, candidate_indices)

        fused: dict[int, float] = {}
        for ranking in (left, right):
            for rank, (index, _score) in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.k + rank + 1)

        # 척도는 첫 번째 검색기(L)의 유사도를 쓴다 — 문턱이 그 척도로 보정돼 있다.
        scale = dict(left)
        order = sorted(fused, key=lambda i: (-fused[i], i))
        return [(i, scale.get(i, 0.0)) for i in order[:k]]
