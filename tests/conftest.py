"""Repository-wide pytest guards that do not add standalone test cases."""

from app.agents.deciding_factor import Factor, evaluate_diff_coverage


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    """C-4 deciding-factor gate 핵심 계약을 매 test session 시작 시 검증한다."""
    # AG-13 / clean 230041형: 공통 조건과 결정적 조건을 절 단위로 나눈다.
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
        )
    ]
    result = evaluate_diff_coverage(request, precedent, shared, decisive)
    assert result.basis == "decisive_difference"
    assert result.fired_rule == "G4"
    assert result.decisive_confirmed_ids == ("F1",)

    # 표면 공통문장만 내놓고 실제 차이를 분석하지 않으면 applies 금지.
    surface_only = evaluate_diff_coverage(request, precedent, shared, [])
    assert surface_only.basis == "incomplete_analysis"
    assert surface_only.fired_rule == "G2"
    assert surface_only.uncovered_differences

    # 원문에 없는 decisive factor는 접지 실패로 폐기되고 applies로 빠지지 않는다.
    phantom = [
        Factor(
            id="F2",
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
    assert rejected.rejected_factor_ids == ("F2",)
