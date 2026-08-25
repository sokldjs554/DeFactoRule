from __future__ import annotations

from app.domain.temporal import eligible_indices, precedent_is_eligible, serial_time
from app.retrieval.lexical import LexicalRetriever


def row(serial: str, request: str = "같은 요청") -> dict:
    return {"serial": serial, "request": request, "label": "비조치"}


def test_serial_time_parses_year_and_within_year_order() -> None:
    assert serial_time("230041") == (2023, 41)
    assert serial_time("240006") == (2024, 6)


def test_future_same_year_precedent_is_rejected() -> None:
    request = row("230041")
    assert precedent_is_eligible(row("220055"), request)
    assert not precedent_is_eligible(row("230058"), request)


def test_missing_or_malformed_time_is_not_invented() -> None:
    request = row("230041")
    assert not precedent_is_eligible(row("unknown"), request)
    assert not precedent_is_eligible(row("230040"), {"serial": None})


def test_none_policy_preserves_legacy_behavior() -> None:
    assert precedent_is_eligible(row("250999"), row("220001"), policy="none")


def test_eligible_indices_are_determined_before_ranking() -> None:
    precedents = [
        row("230058", "스트리밍 방식은 아님"),
        row("220055", "스트리밍 방식은 아님 일부"),
    ]
    request = row("230041", "스트리밍 방식은 아님")
    corpus = [p["request"] for p in precedents] + [request["request"]]
    retriever = LexicalRetriever().fit(precedents, corpus)

    assert retriever.search(request["request"], 1)[0][0] == 0
    candidates = eligible_indices(precedents, request)
    assert candidates == [1]
    assert retriever.search(request["request"], 1, candidates)[0][0] == 1
