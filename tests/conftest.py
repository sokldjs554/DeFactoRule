"""Repository-wide pytest guards that do not add standalone test cases."""

from app.agents.deciding_factor import Factor, evaluate_diff_coverage


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    """C-4 deciding-factor gate 핵심 계약을 매 test session 시작 시 검증한다."""
    # AG-13 / clean 230041형: 공통 조건과 양쪽의 결정적 차이를 절 단위로 나눈다.
    request = (
        "외부 시스템에서 자료를 전달받음.\n"
        "실시간 데이터 연계가 가능한 스트리밍 방식은 아님."
    )
    precedent = (
        "외부 시스템에서 자료를 전달받음.\n"
        "망연계솔루션(스트리밍 방식)을 통해 내부메일시스템으로 전송함."
    )
    shared = [Factor(id="S1", text="외부 시스템에서 자료를 전달받음", side="both")]
    decisive = [
        Factor(
            id="F1",
            text="스트리밍 방식은 아님",
            side="request",
            axis="스트리밍 방식 여부",
            value_in_request="아님",
            value_in_precedent="사용",
            decisive=True,
        ),
        Factor(
            id="F2",
            text="망연계솔루션(스트리밍 방식)을 통해 내부메일시스템으로 전송함",
            side="precedent",
            axis="스트리밍 방식 여부",
            value_in_request="아님",
            value_in_precedent="사용",
            decisive=True,
        ),
    ]
    result = evaluate_diff_coverage(request, precedent, shared, decisive)
    assert result.basis == "decisive_difference"
    assert result.fired_rule == "G4"
    assert result.decisive_confirmed_ids == ("F1", "F2")

    # 표면 공통문장만 내놓고 실제 차이를 분석하지 않으면 applies 금지.
    surface_only = evaluate_diff_coverage(request, precedent, shared, [])
    assert surface_only.basis == "incomplete_analysis"
    assert surface_only.fired_rule == "G2"
    assert surface_only.uncovered_differences

    # 공통 factor도 양쪽 원문에 실제로 존재해야 한다. 지어낸 shared로 G1 우회 금지.
    fake_shared = [Factor(id="S2", text="양쪽에 없는 공통 조건", side="both")]
    no_real_shared = evaluate_diff_coverage(request, precedent, fake_shared, decisive)
    assert no_real_shared.basis == "incomplete_analysis"
    assert no_real_shared.fired_rule == "G1"
    assert "S2" in no_real_shared.rejected_factor_ids

    # 원문에 없는 decisive factor는 접지 실패로 폐기되고 applies로 빠지지 않는다.
    phantom = [
        Factor(
            id="F3",
            text="원문에 존재하지 않는 결정적 조건",
            side="request",
            axis="가공 조건",
            value_in_request="있음",
            value_in_precedent="없음",
            decisive=True,
        )
    ]
    rejected = evaluate_diff_coverage(request, precedent, shared, phantom)
    assert rejected.basis == "incomplete_analysis"
    assert rejected.fired_rule == "G2"
    assert rejected.rejected_factor_ids == ("F3",)

    # 실질 차이가 0이어도 선례 적격성 A1~A4를 확인하지 않으면 applies 금지.
    identical = "동일한 요청 내용임."
    identical_shared = [Factor(id="S3", text="동일한 요청 내용임", side="both")]
    blocked = evaluate_diff_coverage(identical, identical, identical_shared, [])
    assert blocked.basis == "incomplete_analysis" and blocked.fired_rule == "G5-blocked"
    allowed = evaluate_diff_coverage(
        identical,
        identical,
        identical_shared,
        [],
        precedent_admissible=True,
    )
    assert allowed.basis == "identical_after_metadata" and allowed.fired_rule == "G5"
