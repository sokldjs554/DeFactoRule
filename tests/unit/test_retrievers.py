"""검색기 세 종 — 계약과 결정론.

이 프로젝트에서 검색기의 값어치는 "더 많이 찾는가" 가 아니라 **"찾은 것이
판단 근거가 되는가"** 로 잰다. 그 비교는 `test_retrieval_comparison` 이 하고,
여기서는 계약이 지켜지는지를 본다.
"""

from __future__ import annotations

import pytest

from app.domain.temporal import eligible_indices, precedent_is_eligible, serial_time
from app.retrieval.compare import partition_by_retriever, summarise
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import LexicalRetriever

PRECEDENTS = [
    {"request": "내부망과 외부망을 물리적으로 분리하여 운영하는지 질의합니다",
     "label": "비조치"},
    {"request": "클라우드컴퓨팅서비스를 이용하여 인사정보를 처리하려 합니다",
     "label": "비조치"},
    {"request": "보험설계사 단말기가 내부통신망에 연결되는 구조입니다", "label": "조치"},
    {"request": "전자지급결제대행업 등록 대상인지 문의드립니다", "label": "기타"},
]
CORPUS = [p["request"] for p in PRECEDENTS] + [
    "전자금융거래법상 등록 의무에 관한 질의입니다",
    "망분리 대체정보보호통제를 적용하는 경우에 관하여",
]


def _fitted():
    return [
        LexicalRetriever().fit(PRECEDENTS, CORPUS),
        DenseRetriever().fit(PRECEDENTS, CORPUS),
        HybridRetriever(LexicalRetriever(), DenseRetriever()).fit(PRECEDENTS, CORPUS),
    ]


@pytest.mark.parametrize("retriever", _fitted(), ids=lambda r: r.name)
def test_scores_are_in_range_and_descending(retriever):
    hits = retriever.search("내부망과 외부망 분리에 관한 질의", 4)
    assert hits, f"{retriever.name} 이 아무것도 못 찾았습니다"
    scores = [s for _, s in hits]
    assert all(0.0 <= s <= 1.0 for s in scores), f"{retriever.name}: {scores}"
    assert scores == sorted(scores, reverse=True), f"{retriever.name}: 내림차순이 아닙니다"


@pytest.mark.parametrize("retriever", _fitted(), ids=lambda r: r.name)
def test_search_is_deterministic(retriever):
    """같은 입력에 같은 결과인가 — 난수가 섞이면 실험을 다시 돌릴 수 없다."""
    query = "클라우드 이용에 관한 질의입니다"
    assert retriever.search(query, 3) == retriever.search(query, 3)


@pytest.mark.parametrize("retriever", _fitted(), ids=lambda r: r.name)
def test_k_larger_than_pool_is_safe(retriever):
    assert len(retriever.search("아무 질의", 99)) <= len(PRECEDENTS)


@pytest.mark.parametrize("retriever", _fitted(), ids=lambda r: r.name)
def test_every_precedent_finds_itself(retriever):
    """선례와 글자가 같은 질의는 **그 선례**를 1등으로 찾아야 한다.

    잠재 검색기에서 성분 수를 랭크보다 하나 적게 잡았더니 서로 다른 선례가
    같은 좌표로 겹쳤고, 자기 자신 대신 엉뚱한 것을 1등으로 내놨다. 자기를
    못 찾는 검색기는 무엇도 제대로 못 찾는다.
    """
    for target, precedent in enumerate(PRECEDENTS):
        hits = retriever.search(precedent["request"], 1)
        assert hits and hits[0][0] == target, (
            f"{retriever.name}: 선례 {target} 가 자기를 못 찾았습니다 -> {hits}"
        )


def test_empty_pool_returns_nothing():
    for retriever in (LexicalRetriever(), DenseRetriever()):
        retriever.fit([], CORPUS)
        assert retriever.search("질의", 3) == []

    # T-serial 계약도 이 검색기 계약 테스트에 함께 묶는다. 새 테스트 수를
    # 늘리지 않고도 "미래 후보 제거가 ranking 전에 일어난다"는 회귀를 지킨다.
    def temporal_row(serial: str, request: str = "같은 요청") -> dict:
        return {"serial": serial, "request": request, "label": "비조치"}

    assert serial_time("230041") == (2023, 41)
    request = temporal_row("230041", "스트리밍 방식은 아님")
    assert precedent_is_eligible(temporal_row("220055"), request)
    assert not precedent_is_eligible(temporal_row("230058"), request)
    assert not precedent_is_eligible(temporal_row("unknown"), request)
    assert precedent_is_eligible(
        temporal_row("250999"), temporal_row("220001"), policy="none"
    )

    precedents = [
        temporal_row("230058", "스트리밍 방식은 아님"),
        temporal_row("220055", "스트리밍 방식은 아님 일부"),
    ]
    corpus = [p["request"] for p in precedents] + [request["request"]]
    retriever = LexicalRetriever().fit(precedents, corpus)
    assert retriever.search(request["request"], 1)[0][0] == 0
    candidates = eligible_indices(precedents, request)
    assert candidates == [1]
    assert retriever.search(request["request"], 1, candidates)[0][0] == 1


def test_hybrid_keeps_the_lexical_scale():
    """융합은 순위만 섞고 **척도는 L 의 것을 쓴다.**

    Router 는 `top_similarity` 를 도메인 문턱과 비교한다. 그 문턱은 L 의
    코사인 위에서 보정됐으므로, 융합 점수(RRF)를 그대로 넘기면 문턱이 뜻을
    잃는다.
    """
    lexical = LexicalRetriever().fit(PRECEDENTS, CORPUS)
    hybrid = HybridRetriever(LexicalRetriever(), DenseRetriever()).fit(
        PRECEDENTS, CORPUS)
    query = "내부망과 외부망을 분리하여 운영합니다"
    lex_scores = dict(lexical.search(query, 99))
    for index, score in hybrid.search(query, 4):
        assert score == pytest.approx(lex_scores.get(index, 0.0)), (
            f"선례 {index}: 융합 {score} · L {lex_scores.get(index)}"
        )


# ── 비교 지표 ────────────────────────────────────────────────────
def test_trap_rate_counts_what_it_says():
    """함정 비율이 '선례를 찾았을 때 틀릴 비율' 인가."""
    rows = [
        {"request": PRECEDENTS[0]["request"], "label": "비조치"},   # 순응
        {"request": PRECEDENTS[2]["request"], "label": "비조치"},   # 함정
        {"request": "완전히 무관한 다른 이야기입니다", "label": "기타"},  # 선례 없음
    ]
    retriever = LexicalRetriever().fit(PRECEDENTS, CORPUS)
    stats = summarise(partition_by_retriever(retriever, PRECEDENTS, rows))
    assert stats["agree"] + stats["trap"] == stats["anchored"]
    assert stats["agree"] + stats["trap"] + stats["unanchored"] == len(rows)
    if stats["anchored"]:
        assert stats["trap_rate"] == pytest.approx(
            stats["trap"] / stats["anchored"])


def test_unanchored_rows_are_not_counted_as_correct():
    """선례를 못 찾은 것을 '맞았다' 로 세지 않는가.

    커버리지가 낮은 검색기가 함정 비율만으로 좋아 보이는 것을 막으려면
    선례 없음 건수를 항상 함께 봐야 한다.
    """
    rows = [{"request": "무관한 이야기", "label": "조치"} for _ in range(5)]
    retriever = LexicalRetriever().fit(PRECEDENTS, CORPUS)
    stats = summarise(partition_by_retriever(retriever, PRECEDENTS, rows))
    assert stats["unanchored"] == 5
    assert stats["agree"] == 0 and stats["trap"] == 0
    assert stats["by_label"]["조치"]["anchor_rate"] == 0.0
