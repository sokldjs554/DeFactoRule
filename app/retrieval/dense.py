"""D — 잠재의미(LSA) 검색. **모델 내려받기 없이 dense 를 만든다.**

## 왜 사전학습 인코더가 아닌가

`sentence-transformers` + 한국어 인코더가 더 강한 비교였을 것이다. 쓰지 않은
이유는 둘이다.

    1. 이 환경에서 torch 와 400MB 모델 다운로드를 확인하지 못했다.
    2. 저장소를 받은 사람이 다운로드 없이 같은 결과를 재현할 수 있어야 한다.

대신 **코퍼스 자체에서** 잠재 공간을 만든다. 문자 4-gram × 문서 행렬을 절단
특이값분해(SVD)해 얻은 좌표는 dense 표현이고, 공기(共起) 구조를 담는다 —
문자열이 겹치지 않아도 같은 맥락에 나오는 표현끼리 가까워진다.

**한계는 분명하다.** L 도 D 도 결국 이 코퍼스의 표면 통계다. 둘 다 TRAP 을 못
깰 가능성이 있고, 그렇다면 그것이야말로 "검색을 바꿔 풀 문제가 아니다" 의
증명이 된다. 그 결과도 그대로 적는다.

의존성은 numpy 하나다.
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
        self._projection = None      # V_k · S_k^-1  (항 -> 잠재 좌표)
        self._vectors = None         # 선례들의 잠재 좌표 (정규화됨)

    # ── 내부 ────────────────────────────────────────────────────
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

    # ── 계약 ────────────────────────────────────────────────────
    def fit(self, precedents: list[dict], corpus: list[str]) -> DenseRetriever:
        import numpy as np

        self._idf = idf_table(corpus)
        # 코퍼스 전체에 한 번만 나오는 4-gram 은 잠재 구조에 기여하지 않고
        # 행렬만 키운다. IDF 가 만들어진 어휘를 그대로 쓴다.
        self._terms = sorted(self._idf)
        self._index = {term: i for i, term in enumerate(self._terms)}

        matrix = np.vstack([self._row(text) for text in corpus])
        # 성분 수를 랭크보다 **하나 적게** 잡던 판이 있었다. 코퍼스가 작으면
        # 그 하나를 버리는 것만으로 서로 다른 문서가 같은 좌표로 겹친다 —
        # 자기 자신을 찾지 못하는 검색기가 된다. 랭크까지 다 쓴다.
        k = min(self.components, min(matrix.shape))
        _u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        # 특이값이 0 이면 그 축은 정보가 없다 — 나누지 않는다
        keep = [i for i in range(k) if s[i] > 1e-12]
        if not keep:
            raise ValueError("코퍼스에서 잠재 축을 하나도 찾지 못했습니다")
        self._projection = vt[keep].T / s[keep]

        self._vectors = (np.vstack([self._latent(p["request"]) for p in precedents])
                         if precedents else np.zeros((0, len(keep))))
        return self

    def search(self, request: str, k: int = 5) -> list[tuple[int, float]]:
        import numpy as np

        if self._vectors is None or not len(self._vectors):
            return []
        query = self._latent(request)
        sims = self._vectors @ query
        # 코사인은 [-1, 1] 이다. 검색기 계약이 [0, 1] 이므로 음수는 0 으로 둔다
        # — "반대 방향" 과 "무관" 을 구분해 봐야 순위에 쓸 데가 없다.
        sims = np.clip(sims, 0.0, 1.0)
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
        return [(i, float(sims[i])) for i in order[:k]]
