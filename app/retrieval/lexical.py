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

    def search(
        self,
        request: str,
        k: int = 5,
        candidate_indices: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        query = weighted_vector(request, self._idf)
        indices = (
            range(len(self._vectors))
            if candidate_indices is None
            else candidate_indices
        )
        scored = [(i, cosine(query, self._vectors[i])) for i in indices]
        # 동점은 앞선 선례가 이긴다 — 난수 없이 재현된다
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]


def shared_span(query: str, precedent: str, minimum: int = 6) -> str | None:
    """두 글 사이에서 **가장 긴 공통 구절**을 찾는다.

    검색 결과에 "왜 닮았다고 봤는가" 를 붙이기 위한 것이다. 이 구절은
    Validator V3 가 원문과 글자 단위로 대조한다 — 근거에 인용이 없으면 V3 는
    검사할 것이 없고, 그러면 있으나 마나 한 검사가 된다.

    공통 4-gram 에서 시작해 양쪽으로 늘린다. 문서가 수백 자라 전체 DP 는
    필요 없다.
    """
    if not query or not precedent:
        return None
    best = ""
    seen: set[int] = set()
    for i in range(len(query) - 3):
        if i in seen:
            continue
        seed = query[i:i + 4]
        j = precedent.find(seed)
        if j < 0:
            continue
        start_q, start_p = i, j
        while start_q > 0 and start_p > 0 and query[start_q - 1] == precedent[start_p - 1]:
            start_q -= 1
            start_p -= 1
        end_q, end_p = i + 4, j + 4
        while (end_q < len(query) and end_p < len(precedent)
               and query[end_q] == precedent[end_p]):
            end_q += 1
            end_p += 1
        seen.update(range(start_q, end_q - 3))
        if end_q - start_q > len(best):
            best = query[start_q:end_q]
    return best.strip() if len(best.strip()) >= minimum else None
