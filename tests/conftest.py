"""Repository-wide pytest guards that do not add standalone test cases."""

import inspect

from app.agents.deciding_factor import Factor, evaluate_diff_coverage
from app.agents.deciding_factor_prompt import build_prompt, schema
from app.agents.deciding_factor_run import resolve as resolve_s5_plan
from app.infrastructure.schema_rules import check_output_schema


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    """C-4 deciding-factor gate 핵심 계약을 매 test session 시작 시 검증한다."""
    # S5 모델은 결론/basis를 정할 수 없고 두 요청문만 입력받는다.
    s5_schema = schema()
    assert not check_output_schema(s5_schema)
    assert "applicability_basis" not in s5_schema["properties"]
    assert "verdict" not in s5_schema["properties"]
    assert set(inspect.signature(build_prompt).parameters) == {"request", "precedent_request"}
    prompt = build_prompt("요청 A", "선례 B")
    assert "요청 A" in prompt and "선례 B" in prompt

    # 고정 5건의 temporal 검색 상태가 바뀌면 실제 호출 전에 CI에서 멈춘다.
    resolved, drift = resolve_s5_plan()
    assert len(resolved) == 5
    assert not drift, f"C-4 S5 실행 계획 drift: {drift}"

    # AG-13 / clean 230041형: 실재하는 결정적 차이는 literal grounding으로 거절 가능.
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
            text="실시간 데이터 연계가 가능한 스트리밍 방식은 아님",
            side="request",
            axis="스트리밍 방식 여부",
            value_in_request="아님",
            value_in_precedent=None,
            decisive=True,
        ),
        Factor(
            id="F2",
            text="망연계솔루션(스트리밍 방식)을 통해 내부메일시스템으로 전송함",
            side="precedent",
            axis="스트리밍 방식 여부",
            value_in_request=None,
            value_in_precedent="사용",
            decisive=True,
        ),
    ]
    result = evaluate_diff_coverage(request, precedent, shared, decisive)
    assert result.basis == "decisive_difference"
    assert result.fired_rule == "G4"
    assert result.decisive_confirmed_ids == ("F1", "F2")

    # rejection은 shared factor가 부실해도 literal-grounded decisive difference가 있으면
    # fail-safe하게 성립한다. shared completeness는 recovery에만 필수다.
    fake_shared = [Factor(id="S2", text="양쪽에 없는 공통 조건", side="both")]
    reject_with_bad_shared = evaluate_diff_coverage(request, precedent, fake_shared, decisive)
    assert reject_with_bad_shared.basis == "decisive_difference"
    assert reject_with_bad_shared.fired_rule == "G4"
    assert "S2" in reject_with_bad_shared.rejected_factor_ids

    # 표면 공통문장만 내놓고 실제 차이를 분석하지 않으면 recovery 금지.
    surface_only = evaluate_diff_coverage(request, precedent, shared, [])
    assert surface_only.basis == "incomplete_analysis"
    assert surface_only.fired_rule == "G2"
    assert surface_only.uncovered_differences

    # 공통 factor도 양쪽 원문에 실제로 존재해야 G1 recovery gate를 통과한다.
    no_difference_analysis = evaluate_diff_coverage(request, precedent, fake_shared, [])
    assert no_difference_analysis.basis == "incomplete_analysis"
    assert no_difference_analysis.fired_rule == "G1"
    assert "S2" in no_difference_analysis.rejected_factor_ids

    # 원문에 없는 decisive factor는 접지 실패로 폐기되고 applies로 빠지지 않는다.
    phantom = [
        Factor(
            id="F3",
            text="원문에 존재하지 않는 결정적 조건",
            side="request",
            axis="가공 조건",
            value_in_request="있음",
            value_in_precedent=None,
            decisive=True,
        )
    ]
    rejected = evaluate_diff_coverage(request, precedent, shared, phantom)
    assert rejected.basis == "incomplete_analysis"
    assert rejected.fired_rule == "G2"
    assert rejected.rejected_factor_ids == ("F3",)

    # 양쪽에 똑같이 존재하는 문구를 decisive라고 잘못 표시해도 G4로 우회할 수 없다.
    fake_decisive_shared_text = [
        Factor(
            id="F4",
            text="외부 시스템에서 자료를 전달받음",
            side="request",
            axis="공통 조건",
            value_in_request="있음",
            value_in_precedent=None,
            decisive=True,
        )
    ]
    not_unique = evaluate_diff_coverage(request, precedent, shared, fake_decisive_shared_text)
    assert not_unique.basis == "incomplete_analysis"
    assert not_unique.fired_rule == "G2"
    assert not not_unique.decisive_confirmed_ids

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
