"""Anthropic API 와의 경계. 여기서만 네트워크를 만진다.

이 파일이 생긴 이유는 중복 때문이다. 분류기에 붙여 둔 fail-fast 처리 —
계정 수준 오류를 즉시 알아보고 멈추는 것, 오류 메시지를 통째로 남기는 것,
본 요청 전에 한 번 점검하는 것 — 를 새 과제마다 다시 짜면, 어느 하나는
반드시 빠뜨린다. 실제로 크레딧이 소진됐을 때 실패가 확정된 호출을 39건씩
두 번 던진 적이 있다.

경계를 한 곳에 두면 그 처리가 새 과제에 자동으로 따라온다.
"""

from __future__ import annotations

import json
import sys

MODEL = "claude-opus-5"

# 입·출력 100만 토큰당 달러. 비용 추정에만 쓴다.
PRICE_IN = 5.0
PRICE_OUT = 25.0

# 계정 수준 오류 — 다음 요청도 똑같이 실패한다. 건별로 잡고 계속 돌면
# 남은 전부를 헛되이 던지게 된다. 즉시 중단한다.
FATAL_MARKERS = (
    "credit balance is too low",
    "invalid x-api-key",
    "authentication_error",
    "permission_error",
    "Your account has been disabled",
)


class FatalApiError(RuntimeError):
    """계속 시도해도 소용없는 오류."""


def is_fatal(detail: str, body: object = None) -> bool:
    blob = f"{detail} {body}"
    return any(m.lower() in blob.lower() for m in FATAL_MARKERS)


def connect():
    """클라이언트를 만든다. 못 만들면 이유를 말하고 종료한다."""
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic 패키지가 없습니다: pip3 install -r requirements.txt")
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # 자격증명 없음
        sys.exit(
            f"Anthropic 클라이언트를 만들 수 없습니다: {exc}\n"
            "ANTHROPIC_API_KEY 를 설정하거나 `ant auth login` 을 실행하세요."
        )


# 200 을 받았으나 내용이 스키마에 못 미친 경우. 계약은 통과한 것이다.
CONTENT_ONLY_ERRORS = frozenset({"unparseable_output", "refusal"})

_BILLING_HELP = (
    "사전 점검 실패 — 요청을 하나도 보내지 않았습니다.\n  {detail}\n\n"
    "  Anthropic Console 의 Plans & Billing 에서 크레딧을 확인하세요.\n"
    "  충전 직후에는 반영에 몇 분 걸릴 수 있습니다."
)


def preflight(client, schema: dict | None = None) -> None:
    """본 요청을 던지기 전에 계정과 **요청 계약**이 살아 있는지 확인한다.

    처음에는 계정만 봤다. 스키마 없이 한 글자를 보내 200 이 오면 통과였다.
    그래서 구조화 출력 스키마가 거부되는 것을 잡지 못했고, 83건 × 재시도 =
    332 요청이 전부 400 으로 죽은 뒤에야 드러났다(IN-10).

        output_config.format.schema: For 'array' type,
        property 'maxItems' is not supported

    본 요청과 다른 계약으로 하는 사전 점검은 사전 점검이 아니다. 그래서
    schema 를 주면 **실제로 쓸 스키마 그대로** 한 번 호출한다. 토큰은 최소이고
    비용은 사실상 0 이다.
    """
    if schema is not None:
        _check_contract(client, schema)
    else:
        _check_account(client)


def _check_contract(client, schema: dict) -> None:
    """본 요청과 같은 스키마로 한 번 호출해 계약이 받아들여지는지 본다."""
    try:
        probe = call_structured(client, "간단히 답합니다.", ".", schema, max_tokens=64)
    except FatalApiError as exc:
        sys.exit(_BILLING_HELP.format(detail=str(exc).strip()))

    if probe.get("status") == 400:
        sys.exit(
            "사전 점검 실패 — 요청 계약이 거부됐습니다. "
            "요청을 하나도 보내지 않았습니다.\n"
            f"  {probe.get('error_detail') or probe['error']}\n\n"
            "  구조화 출력 스키마를 고쳐야 합니다."
        )
    # 여기서 묻는 것은 **계약이 받아들여졌는가** 뿐이다. 점 하나짜리 프롬프트가
    # 스키마에 맞는 내용을 못 내놓는 것은 당연하고, 그것을 경고로 찍으면 매번
    # 짖는 가드가 된다 — 매번 짖는 경고는 곧 아무도 읽지 않는 경고가 된다.
    if "data" in probe or probe.get("error") in CONTENT_ONLY_ERRORS:
        return
    if "error" in probe:
        print(f"  ⚠ 계약을 확인하지 못했습니다: {probe['error']}")


def _check_account(client) -> None:
    """스키마가 없을 때 — 계정이 살아 있는지만 본다."""
    import anthropic

    try:
        client.messages.create(
            model=MODEL, max_tokens=1, messages=[{"role": "user", "content": "."}]
        )
    except anthropic.APIStatusError as exc:
        detail = getattr(exc, "message", "") or str(exc)
        if is_fatal(detail, getattr(exc, "body", None)):
            sys.exit(_BILLING_HELP.format(detail=detail.strip()))
        print(f"  ⚠ 사전 점검에서 예상치 못한 오류: {type(exc).__name__} {exc.status_code}")


def _usage(response) -> dict:
    """200 응답이면 파싱 실패/거절이어도 과금 토큰을 잃지 않는다."""
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def call_structured(
    client,
    system: str,
    prompt: str,
    schema: dict,
    max_tokens: int = 2000,
    effort: str = "medium",
) -> dict:
    """스키마를 강제한 한 번의 호출. 성공하면 파싱된 dict, 실패하면 error 키.

    계정 수준 오류만 예외로 올린다 — 그것은 건별 실패가 아니라 전체 중단
    사유이기 때문이다.
    """
    import anthropic

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": effort,
            },
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIStatusError as exc:
        detail = getattr(exc, "message", "") or str(exc)
        body = getattr(exc, "body", None)
        if is_fatal(detail, body):
            raise FatalApiError(detail.strip()) from exc
        return {
            "error": f"{type(exc).__name__}: {exc.status_code}",
            "status": exc.status_code,
            "error_detail": detail[:600],
            "error_body": json.dumps(body, ensure_ascii=False)[:600] if body else None,
            "prompt_chars": len(prompt),
        }
    except anthropic.APIConnectionError as exc:
        return {"error": f"connection: {exc}", "prompt_chars": len(prompt)}

    usage = _usage(response)
    if response.stop_reason == "refusal":
        return {"error": "refusal", **usage}

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "unparseable_output", "raw": text[:400], **usage}

    return {"data": parsed, **usage}


def estimate_cost(records: list[dict]) -> float:
    """토큰 수가 실제 숫자로 남은 호출만 합산한다.

    HTTP/API 오류처럼 usage가 없는 레코드는 0으로 취급한다. 반대로 200 응답 뒤
    refusal/unparseable_output이 난 경우에는 call_structured가 usage를 보존하므로
    비용에서 빠지지 않는다.
    """
    input_tokens = [
        value
        for record in records
        if isinstance((value := record.get("input_tokens")), int)
    ]
    output_tokens = [
        value
        for record in records
        if isinstance((value := record.get("output_tokens")), int)
    ]
    return (
        sum(input_tokens) / 1e6 * PRICE_IN
        + sum(output_tokens) / 1e6 * PRICE_OUT
    )