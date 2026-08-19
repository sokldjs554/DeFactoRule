"""H — 두 검색기의 순위를 섞는다 (Reciprocal Rank Fusion).

점수를 더하지 않고 **순위**를 섞는다. L 의 코사인과 D 의 잠재 코사인은 척도가
다르고 분포도 다르다. 척도를 맞추려면 정규화를 해야 하는데, 그 정규화 방식이
또 하나의 자유 변수가 된다. RRF 는 점수를 아예 쓰지 않으므로 그 변수가 없다.

    RRF(d) = Σ_r  1 / (K + rank_r(d))

K 는 상위권의 영향력을 조절한다. 60 은 이 방법을 제안한 논문의 값이고, 이
프로젝트에서 따로 조정하지 않는다 — dev 85건에서 조정할 만한 값이 아니다.

**주의**: RRF 점수는 유사도가 아니다. Router 는 `top_similarity` 를 문턱과
비교하므로, H 를 쓸 때는 융합 순위로 고른 뒤 **원래 검색기의 유사도**를
그대로 넘긴다. 순위를 섞었다고 척도까지 섞으면 문턱이 뜻을 잃는다.
"""

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

    def search(self, request: str, k: int = 5) -> list[tuple[int, float]]:
        pool = max(k * 4, 20)
        left = self.first.search(request, pool)
        right = self.second.search(request, pool)

        fused: dict[int, float] = {}
        for ranking in (left, right):
            for rank, (index, _score) in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.k + rank + 1)

        # 척도는 첫 번째 검색기(L)의 유사도를 쓴다 — 문턱이 그 척도로 보정돼 있다
        scale = dict(left)
        order = sorted(fused, key=lambda i: (-fused[i], i))
        return [(i, scale.get(i, 0.0)) for i in order[:k]]
