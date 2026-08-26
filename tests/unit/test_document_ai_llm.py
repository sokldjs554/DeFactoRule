from __future__ import annotations

import json

from app.document_ai.extraction import extract_fields_llm


class _Usage:
    input_tokens = 50
    output_tokens = 30


class _Block:
    type = "text"

    def __init__(self, payload: dict) -> None:
        self.text = json.dumps(payload, ensure_ascii=False)


class _Response:
    stop_reason = "end_turn"
    usage = _Usage()

    def __init__(self, payload: dict) -> None:
        self.content = [_Block(payload)]


class _Messages:
    def create(self, **kwargs):
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        return _Response(
            {
                "serial": "240001",
                "sector": "전자금융",
                "decision": "비조치",
                "request": "클라우드 이용",
                "quotes": {
                    "serial": "240001",
                    "sector": "전자금융",
                    "decision": "비조치",
                    "request": "클라우드 이용",
                },
            }
        )


class _Client:
    messages = _Messages()


def test_optional_llm_extractor_uses_structured_output() -> None:
    fields = extract_fields_llm(
        "일련번호 240001 전자금융 비조치 클라우드 이용",
        _Client(),
    )
    assert fields.serial == "240001"
    assert fields.quotes["request"] == "클라우드 이용"
