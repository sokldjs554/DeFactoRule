"""구조화 출력 스키마가 API 계약을 지키는가 — 호출 없이 검사한다.

## 왜 이 파일이 있는가

`extract` 83건 × 재시도 = **332 요청이 전부 400 으로 죽었다.** 원인은 한 줄이다.

    output_config.format.schema: For 'array' type,
    property 'maxItems' is not supported

단위 테스트는 스키마를 통과시켰다. JSON Schema 로서는 완전히 정당한 스키마이기
때문이다. 사전 점검도 통과시켰다 — 스키마 없이 한 글자를 보내고 있었으니까.
**로컬에서 정당한 것과 이 API 가 받아 주는 것은 다르다.**

그래서 두 겹으로 막는다.

    1. 이 파일    — 호출 없이, 실측으로 확인된 금지 키워드를 잡는다
    2. preflight  — 본 요청과 **같은 스키마**로 한 번 호출해 계약을 확인한다

1 은 공짜이고 CI 에서 돌아간다. 2 는 1 이 모르는 새 제약을 첫 요청 전에 잡는다.
어느 쪽도 혼자서는 충분하지 않다.

## 목록의 근거를 구분해 둔다

추측으로 목록을 불리면 멀쩡한 스키마를 막게 된다. 그래서 근거를 함께 적는다.

    실측  400 응답으로 직접 확인했다
    유추  실측된 것과 같은 계열이라 함께 막는다. 확인되면 실측으로 옮긴다
"""

from __future__ import annotations

# 키워드 → (근거, 대신 무엇을 하는가)
UNSUPPORTED: dict[str, tuple[str, str]] = {
    "maxItems": ("실측", "개수 상한은 프롬프트로 요청하고 결정론적 코드가 자른다"),
    "minItems": ("유추", "모자란 항목은 파싱 뒤 기본값으로 메운다"),
    "uniqueItems": ("유추", "중복 제거는 파싱 뒤 코드가 한다"),
}


def check_output_schema(schema: object, path: str = "$") -> list[str]:
    """구조화 출력 스키마에서 API 가 받지 않는 키워드를 찾아 돌려준다.

    비어 있으면 통과. 어디에 무엇이 있는지, 대신 무엇을 해야 하는지 말한다.
    """
    problems: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in UNSUPPORTED:
                basis, instead = UNSUPPORTED[key]
                problems.append(f"{path}.{key} — API 가 거부한다({basis}). {instead}")
            problems += check_output_schema(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            problems += check_output_schema(item, f"{path}[{i}]")
    return problems
