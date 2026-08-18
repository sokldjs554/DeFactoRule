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


def preflight(client, schema: dict | None = None) -> None:
    """본 요청을 던지기 전에 계정과 **요청 계약**이 살아 있는지 확인한다.

    처음에는 계정만 봤다. 스키마 없이 한 글자를 보내 200 이 오면 통과였다.
    그래서 구조화 출력 스키마가 거부되는 것을 잡지 못했고, 83건 × 4회 = 332
    요청이 전부 400 으로 죽은 뒤에야 드러났다(IN-10).

        output_config.format.schema: For 'array' type,
        property 'maxItems' is not supported

    본 요청과 다른 계약으로 하는 사전 점검은 사전 점검이 아니다. 그래서
    schema 를 주면 **실제로 쓸 스키마 그대로** 한 번 호출한다. 토큰은 최소이고
    비용은 사실상 0 이다.
    """
    import anthropic

    try:
        if schema is None:
            client.messages.create(
                model=MODEL, max_tokens=1, messages=[{"role": "user", "content": "."}]
            )
        else:
            probe = call_structured(client, "간단히 답합니다.", ".", schema, max_tokens=64)
            if probe.get("status") == 400:
                sys.exit(
                    "사전 점검 실패 — 요청 계약이 거부됐습니다. "
                    "요청을 하나도 보내지 않았습니다.\n"
                    f"  {probe.get('error_detail') or probe['error']}\n\n"
                    "  구조화 출력 스키마를 고쳐야 합니다."
                )
            if "error" in probe:
                print(f"  ⚠ 계약 점검에서 예상치 못한 응답: {probe['error']}")
    except anthropic.APIStatusError as exc:
        detail = getattr(exc, "message", "") or str(exc)
        if is_fatal(detail, getattr(exc, "body", None)):
            sys.exit(
                f"사전 점검 실패 — 요청을 하나도 보내지 않았습니다.\n  {detail.strip()}\n\n"
                "  Anthropic Console 의 Plans & Billing 에서 크레딧을 확인하세요.\n"
                "  충전 직후에는 반영에 몇 분 걸릴 수 있습니다."
            )
        print(f"  ⚠ 사전 점검에서 예상치 못한 오류: {type(exc).__name__} {exc.status_code}")


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

    if response.stop_reason == "refusal":
        return {"error": "refusal"}

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "unparseable_output", "raw": text[:400]}

    return {
        "data": parsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def estimate_cost(records: list[dict]) -> float:
    """성공한 호출들의 추정 비용."""
    ok = [r for r in records if "input_tokens" in r]
    return (
        sum(r["input_tokens"] for r in ok) / 1e6 * PRICE_IN
        + sum(r["output_tokens"] for r in ok) / 1e6 * PRICE_OUT
    )
