"""문서 수치 검사가 **자기가 지킨다고 말한 것**을 실제로 잡는가.

README 코퍼스 검사는 오랫동안 "그 숫자가 어딘가 있는가" 만 물었다. 그래서
위쪽 표에 1,122쌍, 아래쪽 데이터 표에 1,124쌍이 동시에 적힌 채로 통과했다.
가드가 있다는 사실은 가드가 잡는다는 뜻이 아니다.
"""

from __future__ import annotations

from app.evaluation.doc_check import corpus_restatements
from app.evaluation.doc_sync import corpus_counts


def test_conflicting_pair_counts_are_caught(tmp_path):
    """한 문서 안에 서로 다른 쌍 수가 있으면 잡는가 — EV-17 의 반례."""
    cases, pairs = corpus_counts()
    if not cases:
        import pytest
        pytest.skip("코퍼스 산출물이 없습니다")

    (tmp_path / "README.md").write_text(
        f"| 1b | 질의–회답 분할 | {pairs:,}쌍 |\n"
        f"| Track A | {cases:,} 사례 -> {pairs + 2:,} 질의–회답 쌍 |\n",
        encoding="utf-8",
    )
    problems = corpus_restatements(tmp_path)
    assert problems, "밀린 쌍 수를 놓쳤습니다 — 존재 검사로 되돌아갔습니다"
    assert str(pairs + 2) in problems[0].replace(",", "")


def test_agreeing_restatements_pass(tmp_path):
    """모든 자리가 맞으면 통과하는가 — 거짓 경보를 내지 않는다."""
    cases, pairs = corpus_counts()
    if not cases:
        import pytest
        pytest.skip("코퍼스 산출물이 없습니다")

    (tmp_path / "README.md").write_text(
        f"코퍼스 {cases:,} 사례 · {pairs:,}쌍. 다시 적어도 {pairs:,}쌍.\n",
        encoding="utf-8",
    )
    assert not corpus_restatements(tmp_path)


def test_unrelated_counts_are_not_flagged(tmp_path):
    """'실패 케이스 58건' 같은 다른 단위의 수는 건드리지 않는가."""
    cases, pairs = corpus_counts()
    if not cases:
        import pytest
        pytest.skip("코퍼스 산출물이 없습니다")

    (tmp_path / "README.md").write_text(
        f"실패 케이스 58건 · 테스트 341개 · {pairs:,}쌍 · {cases:,} 사례\n",
        encoding="utf-8",
    )
    assert not corpus_restatements(tmp_path)
