"""경계 라벨 생성 — 이 프로젝트에서 가장 비쌌던 버그.

`labels[i]` 는 `chars[i]` 와 `chars[i+1]` **사이**의 공백이어야 하는데, 처음
구현은 공백을 만난 시점에 `labels[-1]` 을 세웠다. 그건 다가올 경계가 아니라
직전 경계다. 모든 라벨이 한 칸 왼쪽으로 밀렸다.

학습과 평가가 같은 함수를 썼기 때문에 지표는 버그와 일관되어 F1 0.79 로
멀쩡해 보였다. 출력을 눈으로 보고서야 드러났다 — "후유증으 로", "것 이".

지표가 검출하지 못한 버그이므로 지표가 아니라 **불변조건**으로 못박는다.
"""

from __future__ import annotations

from app.extraction.spacing import iter_boundaries, restore_line


def test_boundary_labels_are_not_shifted():
    chars, labels = iter_boundaries("금융위원회 및 금융감독원은")
    assert chars == "금융위원회및금융감독원은"
    assert len(labels) == len(chars) - 1
    spaced = {(chars[i], chars[i + 1]) for i, v in enumerate(labels) if v}
    assert spaced == {("회", "및"), ("및", "금")}


def test_leading_space_is_ignored():
    chars, labels = iter_boundaries("  가나 다")
    assert chars == "가나다"
    assert labels == [0, 1]


def test_trailing_space_produces_no_label():
    """마지막 글자 뒤에는 경계가 없다. 라벨 개수는 항상 글자 수 − 1 이다."""
    chars, labels = iter_boundaries("가나다   ")
    assert chars == "가나다"
    assert labels == [0, 0]


def test_roundtrip_labels_reconstruct_original():
    original = "금융회사 등이 대통령령으로 정하는 경우"
    chars, labels = iter_boundaries(original)
    rebuilt = chars[0] + "".join(
        (" " if labels[i] else "") + chars[i + 1] for i in range(len(labels))
    )
    assert rebuilt == original


def test_single_character_line_has_no_boundaries():
    chars, labels = iter_boundaries("가")
    assert (chars, labels) == ("가", [])


def test_restore_never_removes_existing_spaces():
    """복원기는 공백을 넣기만 한다. 원문의 공백은 모델보다 신뢰도가 높다."""
    model = {"contexts": {}, "prior": -99.0}  # 아무것도 삽입하지 않는 모델
    line = "코로나19 펜데믹의후유증으로"
    assert restore_line(model, line, threshold=0.0) == line
