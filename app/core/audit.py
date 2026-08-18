"""버린 것을 기록한다. 화면이 아니라 데이터로.

## 왜 이 파일이 있는가

같은 실수를 세 번 했다.

    API 오류      상태 코드만 남기고 메시지를 버렸다 -> 39건이 왜 죽었는지 모름
    결측 검사     오류 행만 세고 없는 행은 안 셌다   -> 156/170 이 "결측 0"
    기준 검증     버린 기준을 화면에만 찍었다        -> 0개인 이유를 알 수 없음

세 번 다 "걸러내는 코드가 걸러낸 것을 기록하지 않았다" 는 **하나의 패턴**이다.
사례마다 고치는 것으로는 다음 사례를 막지 못한다. 걸러내기를 할 때 기록이
**따라오게** 만들어야 한다.

## 규칙

파이프라인의 어떤 단계든 항목을 버리면 세 가지를 남긴다.

    무엇을    버려진 항목 자체 (요약이 아니라 원본)
    왜        이유 (여러 개일 수 있다 — 첫 번째만 남기면 나머지를 못 본다)
    얼마나    이유별 건수

`tests/regression/test_discards_are_recorded.py` 가 각 단계를 실제로 돌려
이 셋이 산출물에 들어 있는지 확인한다. 정적 검사가 아니라 동작 검사다 —
"기록하는 코드가 있다" 가 아니라 "기록이 실제로 나온다" 를 본다.
"""

from __future__ import annotations

from collections import Counter


class Discards:
    """버려진 항목과 이유를 모은다.

    `summary()` 는 화면용, `records()` 는 파일용이다. **둘 다 내야 한다.**
    화면만 내면 나중에 되짚을 수 없고, 파일만 내면 그 자리에서 눈치채지 못한다.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._items: list[dict] = []
        self._reasons: Counter = Counter()

    def drop(self, item: object, reasons: list[str]) -> None:
        if not reasons:
            raise ValueError("이유 없이 버릴 수 없다 — 이유가 곧 진단이다")
        self._reasons.update(reasons)
        payload = dict(item) if isinstance(item, dict) else {"value": repr(item)[:400]}
        self._items.append({**payload, "rejected_for": list(reasons)})

    def keep_if(self, item: dict, problems: list[str]) -> bool:
        """문제가 없으면 True. 있으면 기록하고 False."""
        if problems:
            self.drop(item, problems)
            return False
        return True

    def __len__(self) -> int:
        return len(self._items)

    def records(self) -> list[dict]:
        return list(self._items)

    def summary(self) -> dict:
        return {
            "stage": self.stage,
            "dropped": len(self._items),
            "reasons": dict(self._reasons.most_common()),
        }

    def report(self, prefix: str = "  ") -> str:
        if not self._items:
            return f"{prefix}버린 항목 없음"
        lines = [f"{prefix}버림 {len(self._items)}건"]
        for reason, n in self._reasons.most_common():
            lines.append(f"{prefix}  {reason}: {n}")
        return "\n".join(lines)


def has_discard_channel(record: dict) -> bool:
    """산출물 한 줄이 '버린 것' 을 담을 자리를 가지고 있는가.

    비어 있어도 된다 — 자리가 있다는 것이 중요하다. 자리가 없으면 버린 순간
    그 정보는 영영 사라진다.
    """
    return any(
        key in record
        for key in ("rejected", "error", "warnings", "masked_leaks", "discards")
    )
