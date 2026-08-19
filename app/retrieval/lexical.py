"""L — 문자 4-gram IDF 코사인. **E5 가 쓴 바로 그것이다.**

새로 만들지 않는다. `app/evaluation/confusable.py` 의 기계를 검색기 계약에
맞춰 감싸기만 한다. 그래야 E8 의 기준선이 E5 의 기준선과 같은 것임을 말할 수
있다 — 다시 구현하면 미묘하게 다른 것을 비교하게 된다.
"""

from __future__ import annotations

from app.evaluation.confusable import cosine, idf_table, weighted_vector


class LexicalRetriever:
    name = "L"

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []

    def fit(self, precedents: list[dict], corpus: list[str]) -> LexicalRetriever:
        self._idf = idf_table(corpus)
        self._vectors = [weighted_vector(p["request"], self._idf) for p in precedents]
        return self

    def search(self, request: str, k: int = 5) -> list[tuple[int, float]]:
        query = weighted_vector(request, self._idf)
        scored = [(i, cosine(query, v)) for i, v in enumerate(self._vectors)]
        # 동점은 앞선 선례가 이긴다 — 난수 없이 재현된다
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
