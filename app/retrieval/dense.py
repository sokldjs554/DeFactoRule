"""D — 잠재의미(LSA) 검색. **모델 내려받기 없이 dense 를 만든다.**

## 왜 사전학습 인코더가 아닌가

`sentence-transformers` + 한국어 인코더가 더 강한 비교였을 것이다. 쓰지 않은
이유는 둘이다.

    1. 이 환경에서 torch 와 400MB 모델 다운로드를 확인하지 못했다.
    2. 저장소를 받은 사람이 다운로드 없이 같은 결과를 재현할 수 있어야 한다.

대신 **코퍼스 자체에서** 잠재 공간을 만든다. 문자 4-gram × 문서 행렬을 절단
특이값분해(SVD)해 얻은 좌표는 dense 표현이고, 공기(共起) 구조를 담는다.
"""

from __future__ import annotations

from app.evaluation.confusable import idf_table, ngrams, normalize

COMPONENTS = 128


class DenseRetriever:
    name = "D"

    def __init__(self, components: int = COMPONENTS) -> None:
        self.components = components
        self._terms: list[str] = []
        self._index: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._projection = None
        self._vectors = None

    def _row(self, text: str):
        import numpy as np

        row = np.zeros(len(self._terms), dtype=np.float64)
        counts = ngrams(normalize(text))
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            j = self._index.get(term)
            if j is not None:
                row[j] = (count / total) * self._idf.get(term, 0.0)
        return row

    def _latent(self, text: str):
        import numpy as np

        vec = self._row(text) @ self._projection
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def fit(self, precedents: list[dict], corpus: list[str]) -> DenseRetriever:
        import numpy as np

        self._idf = idf_table(corpus)
        self._terms = sorted(self._idf)
        self._index = {term: i for i, term in enumerate(self._terms)}
        matrix = np.vstack([self._row(text) for text in corpus])
        k = min(self.components, min(matrix.shape))
        _u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        keep = [i for i in range(k) if s[i] > 1e-12]
        if not keep:
            raise ValueError("코퍼스에서 잠재 축을 하나도 찾지 못했습니다")
        self._projection = vt[keep].T / s[keep]
        self._vectors = (
            np.vstack([self._latent(p["request"]) for p in precedents])
            if precedents
            else np.zeros((0, len(keep)))
        )
        return self

    def search(
        self,
        request: str,
        k: int = 5,
        candidate_indices: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        import numpy as np

        if self._vectors is None or not len(self._vectors):
            return []
        query = self._latent(request)
        sims = np.clip(self._vectors @ query, 0.0, 1.0)
        indices = (
            range(len(sims))
            if candidate_indices is None
            else candidate_indices
        )
        order = sorted(indices, key=lambda i: (-sims[i], i))
        return [(i, float(sims[i])) for i in order[:k]]
