"""선례의 시간 적격성 — 미래 선례가 검색 순위에 들어오기 전에 막는다.

실제 회신일이 저장소에 없으므로 `serial` 은 시간의 **proxy** 다. 같은 해에서는
일련번호가 작은 건이 먼저였다고 근사한다. 이 정책을 실제 결정일자와 동일시하지
않는다.
"""

from __future__ import annotations

from typing import Literal

TemporalPolicy = Literal["none", "serial"]


def serial_time(serial: object) -> tuple[int, int] | None:
    """YYNNN 형식의 일련번호를 `(연도, 연내 순번)`으로 바꾼다.

    형식을 확신할 수 없으면 None 을 반환한다. 시간 값을 지어내지 않는다.
    """
    text = str(serial or "").strip()
    if len(text) < 3 or not text.isdigit():
        return None
    return 2000 + int(text[:2]), int(text[2:])


def precedent_is_eligible(
    precedent: dict,
    request: dict,
    policy: TemporalPolicy = "serial",
) -> bool:
    """선례가 요청보다 과거인가.

    `serial` proxy 를 쓸 때 시간 정보를 읽을 수 없는 후보는 안전하게 제외한다.
    `none` 은 기존 동작을 그대로 보존한다.
    """
    if policy == "none":
        return True
    if policy != "serial":
        raise ValueError(f"알 수 없는 temporal policy: {policy}")

    precedent_time = serial_time(precedent.get("serial"))
    request_time = serial_time(request.get("serial"))
    if precedent_time is None or request_time is None:
        return False
    return precedent_time < request_time


def eligible_indices(
    precedents: list[dict],
    request: dict,
    policy: TemporalPolicy = "serial",
) -> list[int]:
    """검색기가 점수를 매기기 **전** 사용할 수 있는 후보 인덱스."""
    return [
        index
        for index, precedent in enumerate(precedents)
        if precedent_is_eligible(precedent, request, policy)
    ]
