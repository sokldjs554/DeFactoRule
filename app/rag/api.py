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
# 근거 검색은 시간 순서를 거르기 위해 request_serial 이 필요하다. 화면에서
# 사람이 일련번호를 외워 입력할 수는 없으므로, 커밋된 clean test 에서 업권마다
# 하나씩 뽑아 고를 수 있게 한다. 여기서도 산출물만 읽는다 — 새로 계산하지 않는다.

_SAMPLE_SECTORS = ("전자금융", "보험", "자본시장", "상호저축은행업", "여신전문금융업", "공통")


def _clean(text: str) -> str:
    """PDF 에서 딸려 온 사용자 정의 영역 문자와 줄바꿈을 화면용으로 정리한다."""
    out = "".join(" " if "\ue000" <= ch <= "\uf8ff" else ch for ch in text)
    return " ".join(out.split())


@router.get("/samples", summary="근거 검색에 쓸 사례 목록")
def _samples() -> dict:
    """clean test 에서 업권마다 한 건씩 뽑아 돌려줍니다.

    화면이 사례를 고를 수 있게 하기 위한 목록입니다. 정답 라벨은 담지 않습니다 —
    이 절이 보여주는 것은 결론 예측이 아니라 **어떤 선례를 근거로 찾아오는가**입니다.
    """
    from app.core.io import load_jsonl
    from app.core.paths import EVAL

    rows = load_jsonl(EVAL / "nonaction_test_clean.jsonl")
    picked: list[dict] = []
    for sector in _SAMPLE_SECTORS:
        for row in rows:
            if row.get("sector") != sector:
                continue
            text = _clean(row.get("request", ""))
            if not 40 <= len(text) <= 220:
                continue
            picked.append({
                "serial": row["serial"],
                "sector": sector,
                "source": row.get("source"),
                "page": row.get("page"),
                "request": text,
            })
            break
    return {
        "cases": picked,
        "note": (
            "커밋된 clean test 세트에서 업권마다 한 건씩 뽑았습니다. "
            "검색은 요청보다 앞선 선례만 후보로 씁니다."
        ),
    }
