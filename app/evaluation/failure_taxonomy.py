"""실패 taxonomy 와 케이스 레지스트리.

명세 §11 은 "최소 30개 이상의 실패 케이스를 의도적으로 구축하고, 각 실패를
taxonomy 로 분류하고, 개선 전/후를 숫자로 비교한다" 를 요구한다.

**산문으로 적힌 실패 목록은 §11 이 아니다.** 고쳤다고 적어 두기만 하면 다음에
같은 자리가 다시 깨졌을 때 아무도 모른다. 그래서 케이스마다 실행 가능한
`probe` 를 붙이고, 회귀 테스트가 매번 전부 돌린다. probe 가 실패하면 그
수정이 풀린 것이다.

숫자는 두 종류다.

  measured    실제로 재 본 값. 출처(docs/…)를 함께 적는다.
  live        probe 가 이번 실행에서 직접 계산한 값. 옛 구현을 함께 들고 있어
              before 와 after 를 같은 입력으로 잰다.

`measured` 가 아닌 값은 적지 않는다. 재 보지 않은 것은 `metric: null` 이다.
"""

from __future__ import annotations

from app.core.paths import FAILURES

REGISTRY_PATH = FAILURES / "registry.jsonl"

# 계층 → 그 계층에서 관찰된 실패 범주
TAXONOMY: dict[str, dict[str, str]] = {
    "extraction": {
        "format-unhandled": "문서 서식을 코드가 모른다",
        "silent-empty": "못 읽고도 예외 없이 빈 값을 남긴다",
        "encoding-normalization": "같은 글자가 다른 코드포인트로 들어온다",
        "boundary-missplit": "경계를 잘못 잡아 조각이 어긋난다",
        "unreadable-source": "원본 자체가 글자를 내주지 않는다",
    },
    "labeling": {
        "answer-leakage": "입력에 정답이 비친다",
        "label-conflation": "성질이 다른 상태를 한 라벨에 담는다",
        "split-discipline": "dev/test 경계나 라벨링 절차가 무너진다",
    },
    "evaluation": {
        "undiagnosable-discard": "버린 것을 기록하지 않아 원인을 되짚을 수 없다",
        "sample-mismatch": "서로 다른 표본의 숫자를 나란히 놓는다",
        "metric-misuse": "질문에 답하지 못하는 지표를 대표로 쓴다",
        "incomparable-comparison": "비교 조건이 달라 비교가 성립하지 않는다",
        "misdiagnosis": "숫자는 맞는데 해석이 틀렸다",
        "partial-guard": "가드가 대상의 일부만 보고 통과시킨다",
        "phantom-evidence": "증거가 없는데 있는 것처럼 셈한다",
        "uniform-threshold": "일률 문턱이 클래스마다 다른 잣대가 된다",
        "arbitrary-tiebreak": "동점을 판단이 아닌 것으로 가른다",
        "distribution-mismatch": "쓰이지 않을 분포에서 성능을 잰다",
    },
    "retrieval": {
        "degenerate-representation": "표현이 무너져 서로 다른 것이 같아진다",
        "coverage-blind-spot": "필요한 곳에서 아무것도 찾지 못한다",
        "misleading-hit": "찾기는 하는데 그것이 근거가 되지 못한다",
    },
    "agent": {
        "undiagnosable-discard": "버린 것을 기록하지 않아 원인을 되짚을 수 없다",
        "schema-violation": "형식 계약을 벗어난 출력",
        "ungrounded-evidence": "원문에 없는 것을 근거로 든다",
        "miscalibration": "확신의 정도가 정확도와 맞지 않는다",
        "prior-overcorrection": "사전 정보가 소수 클래스를 지운다",
        "unverified-premise": "재보지 않은 가정 위에 설계를 세운다",
    },
    "infrastructure": {
        "undiagnosable-discard": "버린 것을 기록하지 않아 원인을 되짚을 수 없다",
        "error-classification": "회복 가능한 오류와 아닌 오류를 구분하지 못한다",
        "path-resolution": "파일 위치를 코드가 잘못 계산한다",
        "reproducibility": "같은 입력이 같은 결과를 내지 않는다",
        "environment": "실행 환경 차이로 코드가 돌지 않는다",
        "contract-violation": "요청이 외부 API 의 계약을 어겨 거부된다",
        "misleading-estimate": "미리 알려 주는 수가 실제와 달라 판단을 흐린다",
        "continuous-integration": "자동 검사가 실제로는 돌지 않는다",
    },
}

# ══ 재발 패턴 ═══════════════════════════════════════════════════════
#
# 범주(category)는 taxonomy 이지 재발 단위가 아니다. 21개 범주 중 17개가 이미
# 2건 이상인데, 그것을 전부 "반복" 이라고 부르면 아무 뜻도 없다.
#
# 진짜 반복은 **범주를 가로지른다.** "걸러낸 것을 기록하지 않는다" 는 API 오류
# (infrastructure), 결측 검사(evaluation), 기준 검증(agent), 규칙 학습(rules)
# 네 곳에서 같은 모양으로 나왔다. 레지스트리는 각각을 따로 잡았지만 **다음
# 사례를 막지는 못했다.**
#
# 그래서 패턴을 별도 필드로 두고, 2건 이상 붙은 패턴에는 **패턴 단위 가드**를
# 의무화한다. 개별 사례의 probe 가 과거를 지킨다면, 패턴 가드는 현재 코드
# 전체를 훑는다.
PATTERNS: dict[str, str] = {
    "discard-unrecorded": "걸러내는 코드가 걸러낸 것을 기록하지 않는다",
    "enumeration-as-separator": "열거 순번을 질의 구분으로 오인한다",
    "mismatched-sample": "표본이 다른 것을 나란히 놓고 비교한다",
    "guard-narrower-than-claim": "가드가 자기가 지킨다고 말한 것보다 좁게 검사한다",
}

# 패턴 이름 → 그것을 지키는 probe 이름. 2건 이상인 패턴은 여기 있어야 한다.
PATTERN_GUARDS: dict[str, str] = {
    "discard-unrecorded": "every_filter_stage_records_its_discards",
    "enumeration-as-separator": "marks_must_align_before_splitting",
    "mismatched-sample": "comparisons_align_their_samples",
    "guard-narrower-than-claim": "every_guard_is_proven_by_a_counterexample",
}

MIN_CASES_FOR_PATTERN_GUARD = 2


REQUIRED_KEYS = {
    "id",
    "layer",
    "category",
    "title",
    "symptom",
    "detection",
    "fix",
    "metric",
    "probe",
}

MIN_CASES = 30


def validate(case: dict) -> list[str]:
    """레지스트리 한 건의 형식 문제를 모두 모아 돌려준다."""
    problems = []
    missing = REQUIRED_KEYS - set(case)
    if missing:
        problems.append(f"누락 키: {sorted(missing)}")
        return problems

    layer, category = case["layer"], case["category"]
    if layer not in TAXONOMY:
        problems.append(f"알 수 없는 계층: {layer}")
    elif category not in TAXONOMY[layer]:
        problems.append(f"{layer} 에 없는 범주: {category}")

    metric = case["metric"]
    if metric is not None:
        for key in ("name", "before", "after", "kind"):
            if key not in metric:
                problems.append(f"metric 에 {key} 없음")
        if metric.get("kind") == "measured" and not metric.get("source"):
            problems.append("measured 수치는 출처를 적어야 한다")
        if metric.get("kind") not in (None, "measured", "live"):
            problems.append(f"알 수 없는 metric.kind: {metric.get('kind')}")

    pattern = case.get("pattern")
    if pattern is not None and pattern not in PATTERNS:
        problems.append(f"알 수 없는 재발 패턴: {pattern}")
    return problems


def patterns_needing_guards(cases: list[dict]) -> dict[str, list[str]]:
    """2건 이상 붙은 패턴과 그 사례 ID. 이들에는 패턴 가드가 있어야 한다."""
    from collections import defaultdict

    grouped: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        if case.get("pattern"):
            grouped[case["pattern"]].append(case["id"])
    return {
        name: sorted(ids)
        for name, ids in grouped.items()
        if len(ids) >= MIN_CASES_FOR_PATTERN_GUARD
    }


def load_registry(path=None) -> list[dict]:
    from app.core.io import load_jsonl

    return load_jsonl(path or REGISTRY_PATH)
