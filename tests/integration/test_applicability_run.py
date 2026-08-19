"""E11b 실행 경로를 **가짜 클라이언트로 끝까지** 돌린다. 호출 없음.

## 왜 이 파일이 있는가

이 프로젝트에서 지금까지 난 실패는 거의 전부 **API 경로**에서 났다.

    IN-10  스키마의 maxItems -> 332 요청이 전부 400
    IN-11  출력 상한이 기준 수를 안 따라가 응답이 잘림
    IN-12  부풀린 비용 추정이 실험을 접게 만들 뻔함

그리고 그 경로는 **한 번도 테스트된 적이 없었다.** 키가 없으니 못 돈다고 여겼기
때문이다. 그건 틀렸다 — 호출 자체를 가짜로 두면 그 위의 배선은 전부 검사할 수
있다. 실제로 이 파일을 쓰다 이어하기 경로에서 버그 셋을 찾았다(IN-16).

가짜로 못 잡는 것은 하나뿐이다: **API 가 이 요청을 받아 주는가.** 그것은
`preflight` 가 본 요청과 같은 스키마로 한 번 호출해 확인한다.
"""

from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class _Recorder:
    """호출을 대신 받아 미리 정한 응답을 돌려준다."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.schemas = []

    def __call__(self, client, system, prompt, schema, max_tokens=2000, effort="medium"):
        self.calls += 1
        self.schemas.append(schema)
        result = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def _answer(verdict, prompt=None, grounded=True, tokens=(1200, 90)):
    if grounded and prompt:
        a = prompt.split("[사안 A — 지금 판단할 요청]\n")[1].split("\n\n[사안 B")[0]
        b = prompt.split("[사안 B — 닮은 선례의 요청]\n")[1]
        quotes = {"quote_a": a.strip()[:20], "quote_b": b.strip()[:20]}
    else:
        quotes = {"quote_a": "원문에 없는 문장", "quote_b": "이것도 없다"}
    return {"data": {"verdict": verdict, **quotes},
            "input_tokens": tokens[0], "output_tokens": tokens[1]}


def _run(monkeypatch, tmp_path, argv, responses):
    """scripts/agent.py 를 가짜 클라이언트로 실행하고 (출력, 호출기) 를 돌려준다."""
    from app.infrastructure import anthropic_client as ac

    recorder = _Recorder(responses)
    monkeypatch.setattr(ac, "call_structured", recorder)
    monkeypatch.setattr(ac, "connect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(ac, "preflight", lambda client, schema=None: None)
    monkeypatch.setattr(sys, "argv", ["agent.py", *argv])
    monkeypatch.chdir(ROOT)

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            runpy.run_path(str(ROOT / "scripts" / "agent.py"), run_name="__main__")
        except SystemExit:
            pass
    return buffer.getvalue(), recorder


@pytest.fixture()
def out_path(tmp_path):
    if not (ROOT / "experiments" / "results" / "trap_risk.json").exists():
        pytest.skip("보정표가 없습니다")
    return tmp_path / "applicability.jsonl"


def _records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ── 정상 경로 ────────────────────────────────────────────────────
def test_dry_run_sends_nothing(monkeypatch, out_path):
    text, recorder = _run(monkeypatch, out_path.parent,
                          ["applicability", "--dry-run", "--limit", "3",
                           "--output", str(out_path)], [])
    assert recorder.calls == 0, "dry-run 이 호출을 보냈습니다"
    assert "요청을 보내지 않습니다" in text
    assert "추정 비용" in text
    assert not out_path.exists()


def test_fatal_error_stops_and_keeps_what_it_had(monkeypatch, out_path):
    """계정 오류면 즉시 멈추고 그때까지의 결과를 지키는가."""
    from app.infrastructure.anthropic_client import FatalApiError

    text, recorder = _run(
        monkeypatch, out_path.parent,
        ["applicability", "--limit", "5", "--output", str(out_path)],
        [_answer("differs"), _answer("differs"),
         FatalApiError("credit balance is too low")])
    assert "중단" in text and "--resume" in text
    assert recorder.calls == 3
    assert len(_records(out_path)) == 2, "중단 전 결과를 잃었습니다"


def test_fabricated_quotes_do_not_recover_an_abstention(monkeypatch, out_path):
    """지어낸 인용으로는 기권을 거두지 못하는가 — 인용이 근거의 조건이다."""
    text, _ = _run(
        monkeypatch, out_path.parent,
        ["applicability", "--limit", "1", "--output", str(out_path)],
        [_answer("applies", grounded=False)])
    assert "인용 미대조 1건" in text
    assert "기권 회수 0건" in text, "인용이 원문에 없는데 기권을 거뒀습니다"
    assert _records(out_path)[0]["ungrounded"] == ["quote_a", "quote_b"]


def test_abort_threshold_is_announced(monkeypatch, out_path):
    """인용 미대조가 20%를 넘으면 사전 등록한 중단 조건을 말하는가."""
    text, _ = _run(
        monkeypatch, out_path.parent,
        ["applicability", "--limit", "2", "--output", str(out_path)],
        [_answer("applies", grounded=False)])
    assert "20% 초과" in text and "중단 조건" in text


# ── 이어하기 — 여기서 버그 셋이 나왔다 (IN-16) ────────────────────
def test_resume_counts_saved_verdicts(monkeypatch, out_path):
    """이어하기 뒤 판정 분포가 **저장분까지** 세는가.

    이번 실행분만 세면 이어하기 한 번에 앞선 결과가 통째로 사라진다.
    """
    from app.infrastructure.anthropic_client import FatalApiError

    _run(monkeypatch, out_path.parent,
         ["applicability", "--limit", "4", "--output", str(out_path)],
         [_answer("differs"), _answer("differs"),
          FatalApiError("credit balance is too low")])
    text, _ = _run(monkeypatch, out_path.parent,
                   ["applicability", "--limit", "4", "--resume",
                    "--output", str(out_path)],
                   [_answer("unclear")])
    assert "'differs': 2" in text, f"저장된 판정이 사라졌습니다:\n{text}"
    assert "'unclear': 2" in text


def test_resume_does_not_duplicate_failed_rows(monkeypatch, out_path):
    """실패 행을 재시도하면 **덮어쓰는가.**

    그대로 두고 새 행을 덧붙이면 성공/실패 개수가 이중으로 세어진다.
    """
    _run(monkeypatch, out_path.parent,
         ["applicability", "--limit", "2", "--output", str(out_path)],
         [{"error": "APIStatusError: 429", "status": 429}])
    first = _records(out_path)
    assert all("error" in r for r in first)

    text, _ = _run(monkeypatch, out_path.parent,
                   ["applicability", "--limit", "2", "--resume",
                    "--output", str(out_path)],
                   [_answer("differs")])
    second = _records(out_path)
    keys = [(r["source"], r["page"], r["serial"], r["pair_index"]) for r in second]
    assert len(keys) == len(set(keys)), f"중복 행이 생겼습니다: {keys}"
    assert len(second) == len(first), "재시도가 행을 덧붙였습니다"
    assert "실패 0" in text


def test_resume_preserves_recoveries(monkeypatch, out_path):
    """이어하기 뒤에도 앞서 회수한 기권이 남아 있는가.

    저장된 판정을 상태에 다시 반영하지 않으면 회수 건수가 이번 실행분만
    세어져 0 으로 떨어진다 — 문서에 그대로 적히면 틀린 결론이 된다.
    """
    from app.infrastructure.anthropic_client import FatalApiError

    def grounded(prompt_holder):
        return None

    # 1회차: 첫 건을 인용까지 맞춰 회수시키고 그다음 중단
    class _First(_Recorder):
        def __call__(self, client, system, prompt, schema, **kw):
            self.calls += 1
            if self.calls == 1:
                return _answer("applies", prompt=prompt)
            raise FatalApiError("credit balance is too low")

    from app.infrastructure import anthropic_client as ac

    recorder = _First([])
    monkeypatch.setattr(ac, "call_structured", recorder)
    monkeypatch.setattr(ac, "connect", lambda: types.SimpleNamespace())
    monkeypatch.setattr(ac, "preflight", lambda client, schema=None: None)
    monkeypatch.setattr(sys, "argv",
                        ["agent.py", "applicability", "--limit", "3",
                         "--output", str(out_path)])
    monkeypatch.chdir(ROOT)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            runpy.run_path(str(ROOT / "scripts" / "agent.py"), run_name="__main__")
        except SystemExit:
            pass
    assert "기권 회수 1건" in buffer.getvalue(), buffer.getvalue()

    text, _ = _run(monkeypatch, out_path.parent,
                   ["applicability", "--limit", "3", "--resume",
                    "--output", str(out_path)],
                   [_answer("unclear")])
    assert "기권 회수 1건" in text, f"이어하기에서 회수가 사라졌습니다:\n{text}"


# ── 계약 ────────────────────────────────────────────────────────
def test_the_schema_actually_sent_is_the_checked_one(monkeypatch, out_path):
    """보낸 스키마가 금지 키워드 검사를 통과한 그 스키마인가 (IN-10)."""
    from app.infrastructure.schema_rules import check_output_schema

    _, recorder = _run(monkeypatch, out_path.parent,
                       ["applicability", "--limit", "1", "--output", str(out_path)],
                       [_answer("differs")])
    assert recorder.schemas, "스키마가 한 번도 안 보내졌습니다"
    for schema in recorder.schemas:
        assert not check_output_schema(schema)
