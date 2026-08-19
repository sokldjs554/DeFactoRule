"""라벨을 찾는 정규식이 고정돼 있는가.

`조치` 는 `비조치` 의 부분문자열이다. 고정하지 않으면 결과 표에서 조치 행을
찾는 정규식이 비조치 행을 읽고, **검사는 거짓으로 통과한다.** 이 방향의 오류는
스스로 드러나지 않는다.
"""

from __future__ import annotations

from app.evaluation.label_patterns import (
    find_unanchored_label_patterns,
    unanchored_labels,
)


def test_repository_has_no_unanchored_label_pattern():
    problems = find_unanchored_label_patterns()
    assert not problems, "\n".join(problems)


def test_the_pattern_that_actually_fooled_us_is_caught():
    """실제로 우리를 속인 그 정규식을 지금은 잡는가 — 반례.

    처음 만든 판은 "맨 앞은 고정된 것으로 본다" 는 규칙이었고, 그래서 우리를
    속인 바로 그 패턴을 놓쳤다. 규칙을 거꾸로 알고 있었던 것이다.
    """
    fooled = unanchored_labels(r"조치\s+\d+\s+([\d.]+)")
    assert fooled, "패턴 맨 앞의 라벨이 가장 위험한 자리입니다"
    assert fooled[0] == ("조치", 0)

    for risky in (r"\s*조치", r"[^?]{0,24}조치", r".*조치"):
        assert unanchored_labels(risky), f"수량자 뒤를 놓쳤습니다: {risky}"


def test_anchored_forms_pass():
    """제대로 고정한 형태는 통과하는가 — 거짓 경보를 내지 않는다.

    `비조치` 안의 `조치` 앞에는 언제나 글자 `비` 가 있다. 그러므로 어떤
    글자든 리터럴로 앞에 있으면 그것이 곧 고정이다.
    """
    for pattern in (r"^  조치\s+\d+", r"(비조치|조치|기타)", r"\b조치\b", r"a|조치",
                    r"결과:조치", r"^조치"):
        assert not unanchored_labels(pattern), f"고정된 형태를 잡았습니다: {pattern}"


def test_only_labels_that_end_another_label_are_checked():
    """헷갈릴 수 없는 라벨까지 잡지 않는가.

    `비조치` 와 `기타` 를 품는 라벨은 없다. 고정하지 않아도 다른 것으로
    잘못 읽힐 수 없으므로 잡으면 그것은 오탐이다.
    """
    from app.evaluation.label_patterns import AMBIGUOUS

    assert AMBIGUOUS == ("조치",), f"헷갈리는 라벨 목록이 바뀌었습니다: {AMBIGUOUS}"
    assert not unanchored_labels(r"비조치의견서\s*\(")
    assert not unanchored_labels(r"기타\s+\d+")


def test_label_inside_a_longer_label_is_not_a_separate_attempt():
    """'비조치의견서' 안의 '조치' 는 별도의 매칭 시도가 아니다."""
    assert not unanchored_labels(r"비조치의견서\s*\(")


def test_docstrings_are_not_scanned(tmp_path):
    """docstring 에 라벨이 있다고 잡지 않는가.

    처음 만든 판은 docstring 까지 훑어 오탐이 82건 나왔다. 매번 짖는 검사는
    읽히지 않는다.
    """
    (tmp_path / "m.py").write_text(
        '"""이 모듈은 조치 여부를 다룬다. 표기는 \\\\s+ 처럼 보일 수 있다."""\n',
        encoding="utf-8",
    )
    assert not find_unanchored_label_patterns(tmp_path)


def test_scanner_reads_f_string_patterns(tmp_path):
    """f-string 으로 만든 패턴도 보는가 — 실제 사고가 그 모양이었다."""
    (tmp_path / "m.py").write_text(
        "import re\n"
        "n = 3\n"
        "re.search(f'조치{n}\\\\s+', '')\n",
        encoding="utf-8",
    )
    assert find_unanchored_label_patterns(tmp_path)
