"""E11b 준비 — **호출 없이** 프롬프트·스키마·사정거리를 검증한다.

명세 §10(비용 관리)이 요구하는 순서다.

    fixture 100% 통과  ->  dry-run  ->  최소 5건 실측  ->  결과 확인  ->  확대 여부

이 파일이 첫 관문이다. 여기서 통과하지 못하면 호출은 없다.
"""

from __future__ import annotations

import pytest

from app.agents.applicability import (
    MAX_TOKENS,
    SYSTEM,
    Applicability,
    build_prompt,
    estimate_cost,
    schema,
    targets,
)
from app.infrastructure.schema_rules import check_output_schema


# ── 계약 ────────────────────────────────────────────────────────
def test_schema_uses_no_keyword_the_api_rejects():
    """332 요청이 전부 400 으로 죽었던 그 자리(IN-10)를 다시 밟지 않는가."""
    assert not check_output_schema(schema())


def test_schema_forces_a_verdict_and_two_quotes():
    props = schema()["properties"]
    assert set(schema()["required"]) == {"verdict", "quote_a", "quote_b"}
    assert set(props["verdict"]["enum"]) == {a.value for a in Applicability}
    assert schema()["additionalProperties"] is False


def test_token_cap_fits_the_answer():
    """답 JSON 이 상한 안에 드는가 — 잘린 JSON 은 파싱 실패다(IN-11)."""
    import json

    body = json.dumps({"verdict": "differs", "quote_a": "가" * 300,
                       "quote_b": "나" * 300}, ensure_ascii=False)
    assert MAX_TOKENS > len(body) / 2.5 * 1.5, "인용이 길어지면 잘립니다"


# ── 순환 차단 ────────────────────────────────────────────────────
def test_prompt_never_carries_the_precedent_conclusion():
    """프롬프트에 선례의 **결론**이 들어가지 않는가.

    들어가면 모델이 결론에서 거꾸로 판단할 수 있고, 그러면 이 검사는 결론을
    한 번 더 베끼는 장치가 된다. 요청문 두 개만 준다.
    """
    poisoned_request = "요청 내용입니다 조치라는정답 이 섞여 있습니다"
    poisoned_precedent = "선례 요청문입니다 비조치 결론이 여기 적혀 있습니다"
    prompt = build_prompt(poisoned_request, poisoned_precedent)

    # 프롬프트 빌더는 **라벨을 인자로 받지 않는다** — 넣을 방법 자체가 없다
    import inspect

    signature = inspect.signature(build_prompt)
    assert set(signature.parameters) == {"request", "precedent_request"}, (
        f"결론을 받을 수 있는 인자가 생겼습니다: {list(signature.parameters)}"
    )
    # 요청문 안에 있는 낱말은 그대로 간다 — 그것까지 지우면 요청문이 망가진다
    assert poisoned_precedent.split()[0] in prompt


def test_system_prompt_forbids_stating_a_conclusion():
    for phrase in ("결론을 말하지 마십시오", "unclear", "글자 단위로 대조"):
        assert phrase in SYSTEM, f"시스템 프롬프트에 '{phrase}' 가 없습니다"


# ── 사정거리 ─────────────────────────────────────────────────────
class _State:
    def __init__(self, abstained: bool, score: float) -> None:
        self.abstained, self.precedent_score = abstained, score


def test_targets_are_only_abstentions_that_have_a_precedent():
    """선례가 없는 기권은 대상이 아니다 — 검사할 것이 없다.

    기권 78건 중 선례가 문턱 위인 것은 22건뿐이다. 그 사실을 코드로 못 박아
    두면 "왜 22건뿐이냐" 를 다시 세지 않아도 된다.
    """
    states = [
        _State(True, 0.40),    # 대상
        _State(True, 0.05),    # 선례 없음 — 대상 아님
        _State(False, 0.80),   # 기권 안 함 — 대상 아님
        _State(True, 0.15),    # 문턱 정확히 — 대상
    ]
    assert targets(states, None, 0.15) == [0, 3]


def test_cost_estimate_scales_with_targets():
    """22건이 87건보다 훨씬 싸다는 것이 수치로 나오는가."""
    small, large = estimate_cost(22), estimate_cost(87)
    assert small < large / 3, f"22건 ${small:.2f} · 87건 ${large:.2f}"
    assert small < 1.0, f"22건에 ${small:.2f} 는 예상보다 비쌉니다"


# ── 실제 데이터에서의 사정거리 ────────────────────────────────────
def test_real_scope_is_small_and_we_know_it():
    """실제 test 에서 이 검사가 닿는 건수를 확인한다.

    설계서는 87건 기준 $2.4 로 잡았다. 실제 사정거리는 그보다 훨씬 좁다 —
    돈을 쓰기 전에 그 사실을 알아야 한다.
    """
    import json

    from app.agents.workflow import VARIANTS, Workflow
    from app.core.io import load_jsonl
    from app.core.paths import EVAL, PROCESSED, RESULTS
    from app.domain.similarity import DOUBT
    from app.retrieval.lexical import LexicalRetriever

    for path in (EVAL / "nonaction_dev.jsonl", RESULTS / "trap_risk.json"):
        if not path.exists():
            pytest.skip(f"{path.name} 이 없습니다")

    dev = [r for r in load_jsonl(EVAL / "nonaction_dev.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
              for c in cases]
    rules = json.loads((RESULTS / "e6_rules.json").read_text(encoding="utf-8"))["rules"]
    risk = json.loads((RESULTS / "trap_risk.json").read_text(encoding="utf-8"))

    flow = Workflow(LexicalRetriever().fit(dev, [t for t in corpus if t]), dev,
                    rules, risk, policy=VARIANTS["router"])
    states = [flow.run(row) for row in test]
    scope = targets(states, test, DOUBT)
    abstained = sum(1 for s in states if s.abstained)

    assert scope, "이 검사가 닿는 건이 하나도 없습니다"
    assert len(scope) < abstained / 2, (
        f"사정거리 {len(scope)}건 / 기권 {abstained}건 — 문서의 설명과 다릅니다"
    )
    assert estimate_cost(len(scope)) < 1.0
