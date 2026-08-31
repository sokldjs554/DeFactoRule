"""FastAPI router for the optional Evidence RAG layer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.rag.schemas import RAGRequest, RAGResponse
from app.rag.service import RAGUnavailable, run_rag

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/evidence",
    response_model=RAGResponse,
    summary="Temporal hybrid retrieval + grounded evidence memo",
)
def _evidence_rag(req: RAGRequest) -> RAGResponse:
    """Retrieve traceable precedent evidence and optionally generate a memo.

    `generate_memo=false` is deterministic/API-free. Generation never returns a
    trusted memo unless every citation ID exists and every quoted span is grounded
    in its cited precedent request.
    """
    try:
        return run_rag(req)
    except RAGUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# --- 화면에서 고를 사례 목록 ------------------------------------------------
# 근거 검색은 시간 순서를 거르기 위해 request_serial 이 필요하다. 화면에서 사람이
# 일련번호를 외워 입력할 수는 없으므로 커밋된 clean test 에서 골라 준다.
#
# 고르는 기준이 중요하다. 처음에는 업권마다 맨 앞 건을 집었는데, 일련번호가 이른
# 건은 그보다 **앞선 선례 자체가 없어서** 6건 중 5건이 빈 결과였다. 화면만 보면
# 검색이 고장 난 것처럼 보인다.
#
# 실제로 유사도 기준을 넘는 선례가 나오는 것은 clean test 168건 중 88건이다
# (Evidence coverage 52.38%). 그래서 늦은 일련번호부터 훑으며 업권마다 실제로
# 근거가 나오는 건을 하나씩 고른다. 훑는 순서가 고정이라 결과도 고정이다.

_SAMPLE_SECTORS = ("전자금융", "공통", "보험", "자본시장", "상호저축은행업", "여신전문금융업")
_SAMPLE_TOP_K = 3
_samples_cache: dict | None = None


def _clean(text: str) -> str:
    """PDF 에서 딸려 온 사용자 정의 영역 문자와 줄바꿈을 화면용으로 정리한다."""
    out = "".join(" " if "\ue000" <= ch <= "\uf8ff" else ch for ch in text)
    return " ".join(out.split())


def _pick_samples() -> dict:
    from app.core.io import load_jsonl
    from app.core.paths import EVAL
    from app.rag.schemas import RAGRequest

    rows = load_jsonl(EVAL / "nonaction_test_clean.jsonl")
    want = set(_SAMPLE_SECTORS)
    found: dict[str, dict] = {}

    # 늦은 일련번호일수록 쓸 수 있는 선례가 많다. 뒤에서부터 훑는다.
    for row in sorted(rows, key=lambda r: r.get("serial", ""), reverse=True):
        sector = row.get("sector")
        if sector not in want or sector in found:
            continue
        text = _clean(row.get("request", ""))
        if not 40 <= len(text) <= 260:
            continue
        try:
            result = run_rag(RAGRequest(
                request_text=row["request"], request_serial=row["serial"],
                top_k=_SAMPLE_TOP_K, generate_memo=False,
            ))
        except RAGUnavailable:
            break
        if not result.evidence:
            continue
        found[sector] = {
            "serial": row["serial"], "sector": sector,
            "source": row.get("source"), "page": row.get("page"), "request": text,
        }
        if len(found) == len(want):
            break

    return {
        "cases": [found[s] for s in _SAMPLE_SECTORS if s in found],
        "note": (
            "커밋된 clean test에서 업권마다 한 건씩 골랐습니다. "
            "전체 168건 가운데 유사도 기준을 넘는 선례가 나오는 것은 88건(52.4%)이며, "
            "나머지는 근거 없이 답하지 않습니다."
        ),
    }


@router.get("/samples", summary="근거 검색에 쓸 사례 목록")
def _samples() -> dict:
    """clean test에서 업권마다 한 건씩 뽑아 돌려줍니다.

    실제로 선례가 나오는 건으로 고릅니다 — 화면에서 눌렀을 때 빈 결과만 나오면
    검색이 고장 난 것처럼 보이기 때문입니다. 정답 라벨은 담지 않습니다: 이 절이
    보여주는 것은 결론 예측이 아니라 **어떤 선례를 근거로 찾아오는가**입니다.
    """
    global _samples_cache
    if _samples_cache is None:
        _samples_cache = _pick_samples()
    return _samples_cache
