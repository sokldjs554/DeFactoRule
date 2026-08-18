"""구조화 출력 스키마가 API 계약을 지키는가.

332 요청이 전부 400 으로 죽은 뒤에 생긴 파일이다. 그때 잃은 것은 돈이 아니라
(400 은 과금되지 않는다) **한 단계를 통째로 다시 돌려야 한다는 사실**이었다.
"""

from __future__ import annotations

import pytest

from app.agents.criteria import apply_schema, extract_schema
from app.infrastructure.schema_rules import check_output_schema


@pytest.mark.parametrize("name,schema", [
    ("extract", extract_schema()),
    ("apply", apply_schema(5)),
])
def test_live_schemas_are_accepted_shapes(name, schema):
    """실제로 API 에 보내는 스키마에 금지 키워드가 없는가."""
    problems = check_output_schema(schema)
    assert not problems, f"{name} 스키마: " + " / ".join(problems)


def test_checker_finds_the_keyword_that_actually_broke_us():
    """실제로 우리를 깨뜨린 그 스키마를 지금은 잡는가.

    가드는 '지금 통과한다' 가 아니라 '과거의 그것을 잡는다' 로 확인한다.
    """
    broken = {
        "type": "object",
        "properties": {
            "criteria": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        },
    }
    problems = check_output_schema(broken)
    assert len(problems) == 1
    assert "maxItems" in problems[0]
    assert "criteria" in problems[0], "어디에 있는지 말해야 고칠 수 있다"


def test_nested_schemas_are_reached():
    """중첩된 자리에 숨어 있어도 찾는가."""
    deep = {"properties": {"a": {"items": {"properties": {"b": {"minItems": 1}}}}}}
    assert check_output_schema(deep), "중첩된 금지 키워드를 놓쳤습니다"
