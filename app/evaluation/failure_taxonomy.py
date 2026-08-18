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
        "sample-mismatch": "서로 다른 표본의 숫자를 나란히 놓는다",
        "metric-misuse": "질문에 답하지 못하는 지표를 대표로 쓴다",
        "incomparable-comparison": "비교 조건이 달라 비교가 성립하지 않는다",
        "misdiagnosis": "숫자는 맞는데 해석이 틀렸다",
        "distribution-mismatch": "쓰이지 않을 분포에서 성능을 잰다",
    },
    "agent": {
        "schema-violation": "형식 계약을 벗어난 출력",
        "ungrounded-evidence": "원문에 없는 것을 근거로 든다",
        "miscalibration": "확신의 정도가 정확도와 맞지 않는다",
        "prior-overcorrection": "사전 정보가 소수 클래스를 지운다",
    },
    "infrastructure": {
        "error-classification": "회복 가능한 오류와 아닌 오류를 구분하지 못한다",
        "path-resolution": "파일 위치를 코드가 잘못 계산한다",
        "reproducibility": "같은 입력이 같은 결과를 내지 않는다",
        "environment": "실행 환경 차이로 코드가 돌지 않는다",
        "continuous-integration": "자동 검사가 실제로는 돌지 않는다",
    },
}

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
    return problems


def load_registry(path=None) -> list[dict]:
    from app.core.io import load_jsonl

    return load_jsonl(path or REGISTRY_PATH)
