"""API 경계의 결정론적 부분.

여기 있는 판정 하나가 두 번 돈을 태웠다. 크레딧 소진은 다음 요청도 100%
실패하는데, 그것을 항목 오류로 취급해 실패가 확정된 호출을 39건씩 두 번
던졌다. 네트워크 없이 검사할 수 있는 부분이라 반드시 검사한다.
"""

from __future__ import annotations

import pytest

from app.infrastructure.anthropic_client import (
    FATAL_MARKERS,
    PRICE_IN,
    PRICE_OUT,
    estimate_cost,
    is_fatal,
)


@pytest.mark.parametrize("detail", [
    "Your credit balance is too low to access the Anthropic API",
    "invalid x-api-key",
    "authentication_error: bad key",
    "permission_error",
    "Your account has been disabled",
])
def test_account_level_errors_are_fatal(detail):
    assert is_fatal(detail), detail


@pytest.mark.parametrize("detail", [
    "Overloaded: please retry",
    "rate_limit_error: too many requests",
    "Internal server error",
    "connection reset by peer",
])
def test_transient_errors_are_not_fatal(detail):
    """일시 오류를 치명으로 잘못 보면 재시도로 살릴 수 있는 것을 버린다."""
    assert not is_fatal(detail), detail


def test_fatal_detection_is_case_insensitive():
    assert is_fatal("YOUR CREDIT BALANCE IS TOO LOW")


def test_body_is_searched_too():
    """메시지가 비어 있고 본문에만 이유가 담기는 경우가 있다."""
    assert is_fatal("", {"error": {"message": "credit balance is too low"}})


def test_every_marker_is_actually_detected():
    """마커를 추가해 놓고 검사에서 빠뜨리는 일을 막는다."""
    for marker in FATAL_MARKERS:
        assert is_fatal(f"prefix {marker} suffix"), marker


def test_cost_estimate_uses_both_directions():
    records = [{"input_tokens": 1_000_000, "output_tokens": 0},
               {"input_tokens": 0, "output_tokens": 1_000_000}]
    assert estimate_cost(records) == pytest.approx(PRICE_IN + PRICE_OUT)


def test_cost_ignores_failed_records():
    """실패한 호출에는 토큰이 없다. 세면 비용이 부풀거나 예외가 난다."""
    records = [{"error": "boom"}, {"input_tokens": 1_000_000, "output_tokens": 0}]
    assert estimate_cost(records) == pytest.approx(PRICE_IN)


def test_cost_of_nothing_is_zero():
    assert estimate_cost([]) == 0.0


# ── 사전 점검 — 계약을 보는가, 그리고 헛짖지 않는가 ────────────────
class _StubClient:
    """call_structured 를 대신할 자리. preflight 는 이 결과만 보고 판단한다."""

    def __init__(self, result):
        self.result = result


def _run_preflight(monkeypatch, result):
    """call_structured 를 고정 결과로 바꿔 preflight 만 검사한다."""
    from app.infrastructure import anthropic_client as ac

    monkeypatch.setattr(ac, "call_structured", lambda *a, **k: result)
    ac.preflight(_StubClient(result), schema={"type": "object"})


def test_contract_rejection_stops_before_any_real_request(monkeypatch, capsys):
    """400 이면 한 건도 보내지 않고 멈추는가 — IN-10 이 일어난 자리."""
    import pytest

    detail = "output_config.format.schema: For 'array' type, property 'maxItems' is not supported"
    with pytest.raises(SystemExit) as exc:
        _run_preflight(monkeypatch, {
            "error": "BadRequestError: 400", "status": 400, "error_detail": detail,
        })
    assert "maxItems" in str(exc.value), "왜 거부됐는지 말해야 고칠 수 있다"


def test_unparseable_probe_output_is_not_a_warning(monkeypatch, capsys):
    """200 을 받았는데 내용이 스키마에 못 미친 것은 계약 통과다.

    점 하나짜리 프롬프트가 스키마에 맞는 내용을 못 내놓는 것은 당연하다.
    이것을 경고로 찍으면 매번 짖고, 매번 짖는 경고는 읽히지 않는다.
    """
    _run_preflight(monkeypatch, {"error": "unparseable_output", "raw": "..."})
    assert capsys.readouterr().out == "", "정상 동작에 경고를 찍었습니다"


def test_probe_that_never_reached_the_api_does_warn(monkeypatch, capsys):
    """연결 자체가 안 된 것은 계약을 확인하지 못한 것이므로 말해야 한다."""
    _run_preflight(monkeypatch, {"error": "connection: timed out"})
    assert "확인하지 못했습니다" in capsys.readouterr().out
