# 32. Live LLM Contract Audit

## 목적

DeFactoRule의 live Claude 경로가 실제 API 호출에서도 기존 계약을 지키는지 확인하기 위한 소규모 contract audit이다. 새로운 기능을 추가하거나 final decision 성능을 개선하는 실험이 아니다.

핵심 질문은 두 가지다.

1. Structured Output schema를 통과한 LLM 출력이 실제 원문에도 grounded되어 있는가?
2. grounded되지 않은 출력이 들어왔을 때 deterministic validator가 fail-closed로 동작하는가?

이번 결과는 **production benchmark, 고객 문서 benchmark, decision accuracy benchmark가 아니다.**

## 고정된 평가 조건

- 모델: `claude-opus-5`
- 총 표본: **29건**
  - S5 arm: 19건
  - RAG arm: 10건 / evidence-eligible population 88건
- 난수/seed 없음
- S5 선정: frozen temporal Router에서 abstain이면서 `precedent_score >= DOUBT`인 행 전체
- RAG 선정: T-serial + similarity floor를 통과한 evidence가 1개 이상인 행을 row key 정렬 후 stride 9로 선택
- prompt / threshold / Router / temporal policy / similarity floor / frozen E6 / S5 policy 변경 없음
- 결과를 본 뒤 재튜닝하지 않음
- 더 좋은 출력을 얻기 위한 sample 교체 또는 재호출 없음

실행 전 frozen profile은 다음과 일치했다.

| 항목 | Frozen clean profile |
|---|---:|
| n | 168 |
| answered / abstained | 76 / 92 |
| correct / wrong | 63 / 13 |
| coverage | 45.24% |
| answered accuracy | 82.89% |

## Live Claude 실행 결과

| 항목 | 결과 | 분모 |
|---|---:|---:|
| API success | **27 / 29** | attempted calls 29 |
| Structured schema valid | **27 / 27** | successful calls 27 |
| S5 factor literal grounding | **141 / 152** | extracted S5 factors 152 |
| S5 rejected factors | **11** | 10 S5 samples에서 관측 |
| RAG invalid citation IDs | **0** | live RAG memo validation |
| RAG ungrounded exact quotes | **1** | 1 RAG sample에서 관측 |
| Unsupported output items | **12** | 11 / 29 fixed samples에서 관측 |

가장 중요한 관찰은 **schema-valid와 evidence-grounded가 같은 조건이 아니었다는 점**이다.

성공한 Claude 호출 27건은 모두 structured schema를 통과했지만, content validation에서는 S5 factor 11개가 literal grounding을 통과하지 못했고 RAG memo에서 ungrounded exact quote 1개가 관측됐다.

즉, 구조적으로 올바른 JSON을 받았다는 사실만으로 원문 근거에 충실한 출력이라고 간주할 수 없었다.

## Validator가 실제로 차단한 사례

RAG arm에서 **ungrounded exact quote 1건**이 발생했다.

```text
Claude structured memo
        ↓
Schema valid
        ↓
Exact quote grounding fail
        ↓
Deterministic validator reject
        ↓
abstain (fail-closed)
```

이 사례는 `invalid citation ID` 문제가 아니라, 존재하는 evidence context를 참조하면서도 **인용 문구 자체가 원문과 일치하지 않은 경우**다. citation ID validation만으로는 충분하지 않고 exact quote grounding이 별도로 필요한 이유를 live output에서 확인했다.

S5에서도 10개 sample에 걸쳐 총 11개의 factor가 literal grounding에서 거부됐다. 이를 "Claude가 11번 틀렸다"거나 "37.9% 오류율"로 표현하지 않는다. factor 단위와 sample 단위의 분모가 다르고, 이 audit은 작은 fixed contract audit이기 때문이다.

## Frozen profile 보존

Live audit 전후 final clean profile은 변하지 않았다.

- answered / abstained = 76 / 92
- correct / wrong = 63 / 13
- coverage = 45.24%
- answered accuracy = 82.89%
- `frozen_profile_unchanged = true`

이번 audit은 LLM output contract와 grounding/validation 경로를 검증한 별도 실험이며, final 168건 decision 성능 향상으로 해석하지 않는다.

## 해석

이번 audit에서 얻은 핵심 결론은 다음 한 문장이다.

> **Schema-valid != Evidence-grounded.**

LLM은 schema를 잘 지킬 수 있어도 원문에 없는 factor나 exact quote를 생성할 수 있다. 따라서 DeFactoRule은 LLM 출력을 최종 verdict로 사용하지 않고, factor / citation / quote를 deterministic validator의 입력으로 제한한다.

이는 "LLM이 잘 답하도록 프롬프트를 개선했다"는 주장보다, **LLM 출력도 검증해야 하는 데이터로 취급한다**는 시스템 계약에 가깝다.

## 한계

- 표본은 29건으로 작다.
- fixed contract audit이며 일반적인 Claude 오류율을 추정하지 않는다.
- 실제 고객 문서나 production traffic을 사용하지 않았다.
- domain expert가 결과를 별도로 판정한 benchmark가 아니다.
- 2건의 API 실패가 있었고, 이는 content quality 분모와 분리해서 기록한다.
- S5 factor grounding과 RAG quote grounding은 서로 다른 validator 경로이므로 하나의 단일 "정확도"로 합치지 않는다.
- 이번 결과를 보고 prompt, Router, retrieval threshold를 다시 맞추지 않았다.

## Artifact

Aggregate freeze:

- `experiments/results/clean/live_llm_audit_summary.json`

Audit harness:

- `app/evaluation/live_llm_audit.py`
- `scripts/evaluate_live_llm_audit.py`
- `tests/unit/test_live_llm_audit.py`
