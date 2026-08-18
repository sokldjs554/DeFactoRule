"""질의–회답 분할.

이 모듈에는 값비싼 실수가 하나 있었다. "회답에만 순번이 있으면 회답만 자른다"
는 폴백을 두었고 그것으로 81쌍이 생겼는데 **전부 오분할**이었다. 회답 안의
순번은 질의 구분이 아니라 요건 열거였기 때문이다.

산출물 회귀 테스트가 분할 모드를 감시하고 있지만, 그 안쪽 함수들은 여태
직접 검사된 적이 없었다.
"""

from __future__ import annotations

from app.extraction.splitting import find_marks, slice_by_marks, split_case


def case(question: str, answer: str, **extra) -> dict:
    return {
        "source": "t", "doc_type": "interpretation", "serial": "1",
        "page": 1, "sector": "공통", "warnings": [],
        "fields": {"질의요지": question, "회답": answer, "이유": "이유 본문"},
        **extra,
    }


# ── 순번 찾기 ────────────────────────────────────────────────────
def test_finds_a_monotonic_chain():
    marks = find_marks("①첫째 질의 ②둘째 질의")
    assert [n for _, n in marks] == [1, 2]


def test_single_mark_is_not_a_split():
    """하나뿐이면 구분이 아니다. 자를 곳이 없다."""
    assert find_marks("①오직 하나뿐") == []


def test_chain_must_start_at_one():
    assert find_marks("②둘째부터 ③셋째") == []


def test_quoted_numbers_that_break_the_chain_are_ignored():
    """본문 인용구 안의 ① 이 섞이면 순번이 뒤로 갔다가 돌아온다."""
    marks = find_marks("①첫째 (인용: ①어쩌고) ②둘째")
    assert [n for _, n in marks] == [1, 2]
    assert marks[1][0] > marks[0][0]


def test_textual_form_is_recognised():
    marks = find_marks("질의1. 무엇인가 질의2. 또 무엇인가")
    assert [n for _, n in marks] == [1, 2]


def test_slices_run_to_the_next_mark():
    text = "①첫째 내용 ②둘째 내용"
    sliced = slice_by_marks(text, find_marks(text))
    assert sliced[1].startswith("①첫째")
    assert "둘째" not in sliced[1]
    assert sliced[2].startswith("②둘째")


# ── 사례 분할 ────────────────────────────────────────────────────
def test_paired_marks_produce_paired_split():
    rows = split_case(case("①갑이 가능한가 ②을이 가능한가",
                           "①갑은 가능하다 ②을은 불가하다"))
    assert len(rows) == 2
    assert {r["split_mode"] for r in rows} == {"paired"}
    assert [r["pair_index"] for r in rows] == [1, 2]
    assert all(r["pair_count"] == 2 for r in rows)
    assert "갑" in rows[0]["question"] and "갑" in rows[0]["answer"]


def test_answer_only_marks_do_not_split():
    """회답 안의 순번은 요건 열거이지 질의 구분이 아니다.

    이 폴백이 81쌍을 만들었고 전부 오분할이었다. 되살아나면 여기서 걸린다.
    """
    rows = split_case(case("하나의 질의입니다",
                           "①첫째 요건을 갖추고 ②둘째 요건도 갖추어야 한다"))
    assert len(rows) == 1
    assert rows[0]["split_mode"] == "single"


def test_question_only_marks_do_not_split():
    rows = split_case(case("①갑 ②을", "일괄하여 가능합니다"))
    assert len(rows) == 1
    assert rows[0]["split_mode"] == "single"


def test_mismatched_counts_fall_back_to_single():
    """질의 3개 회답 2개처럼 짝이 안 맞으면 자르지 않는다."""
    rows = split_case(case("①갑 ②을 ③병", "①갑은 가능 ②을은 불가"))
    assert len(rows) == 1
    assert rows[0]["split_mode"] == "single"


def test_metadata_is_carried_to_every_pair():
    rows = split_case(case("①갑 ②을", "①가능 ②불가", decision="비조치"))
    for row in rows:
        assert row["source"] == "t"
        assert row["serial"] == "1"
        assert row["decision"] == "비조치"
        assert row["reason"] == "이유 본문"


def test_nonaction_fields_are_used_when_present():
    """비조치의견서는 항목 이름이 다르다 — 요청대상행위 / 판단."""
    row = split_case({
        "source": "t", "doc_type": "nonaction", "serial": "1", "page": 1,
        "sector": "공통", "warnings": [],
        "fields": {"요청대상행위": "갑 행위", "판단": "비조치", "판단이유": "근거"},
    })[0]
    assert row["question"] == "갑 행위"
    assert row["answer"] == "비조치"
    assert row["reason"] == "근거"
