# Final Clean Agent Freeze — C-1 ~ C-5

이 문서는 DeFactoRule의 **최종 제출용 engineering state**를 고정한다. 이전 실험의 수치를 다시 포장하는 문서가 아니라, split leakage를 발견한 뒤 clean protocol로 다시 구성한 Agent Workflow가 어디까지 검증됐고 어디서 멈췄는지를 기록한다.

## 1. 최종 문제 정의

DeFactoRule은 “가장 비슷한 과거 사례의 결론을 복사하는 검색기”가 아니다.

> 과거 예외 승인/비조치 이력에서 실제 판단 기준을 복원하고, 새 업무 요청에 그 선례를 적용할 때 **결정적 차이·충돌·불확실성**을 검증하는 AI Agent Workflow.

최종 workflow의 핵심은 세 가지다.

1. **과거 선례만 검색한다.** 미래 사례는 retrieval 후보가 될 수 없다.
2. **표면 유사도와 적용 가능성을 분리한다.** 비슷한 문장이 있다고 같은 판단을 적용하지 않는다.
3. **확실하지 않으면 답하지 않는다.** abstain/human handoff는 실패가 아니라 서비스 계약이다.

## 2. 왜 clean protocol을 다시 만들었나

기존 dev/test split을 감사하면서 test 27건이 dev의 요청과 similarity >= .99였고, 날짜 표현을 제거하면 27/27이 같은 normalized request group으로 묶이는 것을 확인했다. 기존 split은 행 순서 기반이어서 동일 사안·날짜 변형이 dev/test를 넘을 수 있었다.

최종 split은 요청문에서 serial/date 변형을 정규화한 **group 단위 split**이다.

- dev 87
- test 168
- 전체 255행 보존
- dev/test shared group 0
- 기존 leakage 27/27을 같은 split으로 이동

이후의 clean 결과는 이 split을 기준으로만 해석한다.

## 3. Temporal eligibility — ranking보다 먼저

기존 검색은 요청 시점보다 뒤의 사례도 후보로 볼 수 있었다. 실제 날짜 필드가 일관되게 존재하지 않으므로 두 정책을 비교했다.

- `T-strict`: precedent year < request year
- `T-serial`: `(year, serial-within-year)`가 request보다 작은 precedent만 허용

clean test 168건에서:

| 정책 | threshold-positive anchored | zero eligible |
|---|---:|---:|
| no filter | 60 | 0 |
| T-serial | 47 | 1 |
| T-strict | 29 | 33 |

T-strict는 근거를 지나치게 제거했다. 최종 정책은 **T-serial**이다.

중요한 한계가 있다. 실제 날짜가 있는 within-year pair에서 serial 순서는 날짜 순서와 완전히 일치하지 않았고 약 18.4% inversion이 관측됐다. 따라서 문서와 코드 모두 T-serial을 **chronology proxy / temporal eligibility proxy**라고 부르며 “temporal-safe”라고 부르지 않는다.

구현 계약은 단순하다.

```text
request
  → eligible precedent indices 계산
  → candidate subset을 retriever에 전달
  → subset 안에서만 ranking
```

후보를 먼저 전체 ranking한 뒤 미래 결과를 지우는 방식이 아니다.

## 4. Temporal-matched risk calibration

retrieval 정책이 바뀌었으므로 risk calibration도 같은 temporal contract에서 다시 계산했다.

clean dev 87건 중:

- eligible predecessor 있음: 86
- 없음: 1
- overall precedent-following risk: 28/86 = 0.3256

| band | n | wrong | risk | Wilson 95% CI |
|---|---:|---:|---:|---:|
| trust | 27 | 2 | 0.074 | [0.021, 0.234] |
| middle | 9 | 3 | 0.333 | [0.121, 0.646] |
| doubt | 50 | 23 | 0.460 | [0.330, 0.596] |

trust upper bound < doubt lower bound라 band separation은 유지됐다.

하지만 temporal-matched risk table로 Router를 다시 돌려도 C-2의 temporal Router 결과와 **완전히 동일**했다. 즉 calibration provenance mismatch는 해결됐지만 C-2의 작은 성능 변화 원인은 아니었다. threshold를 다시 튜닝해 숫자를 올리지 않았다.

## 5. C-4 — LLM Deciding-Factor Agent + deterministic gate

### 5.1 AG-13에서 시작한 문제

이전 applicability audit에서 LLM이 원문에 존재하는 공유 문장을 근거로 들었지만 실제 결론을 가르는 차이를 놓친 사례가 있었다. grounding만 맞는다고 근거가 충분한 것이 아니었다.

이를 위해 S5를 다음 구조로 분리했다.

LLM output:

- `shared_factors`
- `only_in_request`
- `only_in_precedent`
- `metadata_candidates`

LLM이 출력할 수 없는 것:

- `verdict`
- `applies / differs / unclear`
- `applicability_basis`

최종 basis는 deterministic `Diff Coverage Gate`가 계산한다.

### 5.2 안전성은 비대칭

`differs`와 `applies`에 같은 completeness를 요구하지 않는다.

- 원문에 grounded되고 반대쪽에는 없는 **결정적 차이 하나**가 확인되면 선례 적용을 거절할 수 있다.
- 반대로 “결정적 차이가 없다”며 선례를 회수하려면 모든 실질 차이가 설명돼야 한다.

이 비대칭은 의도적이다. 프로젝트에서 가장 위험한 실패는 “차이를 놓친 채 비슷하다고 회수하는 것”이기 때문이다.

### 5.3 5-case qualitative audit

최종 audit summary는 `experiments/results/clean/c4_s5_audit_summary.json`에 고정했다.

| case | 성격 | 최종 gate | 해석 |
|---|---|---|---|
| 240006 | 고유사도 충돌 | G4 decisive difference | 표면 유사와 실제 규제 대상 차이 분리 |
| 230067 | 모호/부분 적용 | G4 decisive difference | grounded 차이를 통한 보수적 거절 |
| 240022 | 잘못된 top + opposing | G4 decisive difference | 잘못된 top precedent transfer 차단 |
| 220070 | 잘못된 top + opposing 0 | G4 decisive difference | opposition guard 없이도 surface-match 위험 포착 |
| 250055 | 적용 후보 | **G4 decisive difference** | 사전 기대 recovery가 확인되지 않음 |

결론은 성능 홍보가 아니다.

> **S5는 자동 recovery 모듈로 검증되지 않았다.** 현재 채택 범위는 grounded decisive difference가 있을 때 잘못된 precedent transfer를 막는 **fail-closed safety veto**다.

따라서 S5는 final 168-row aggregate metric을 높이는 데 사용하지 않았다.

## 6. Runtime rule schema mismatch

clean E6 asset에는 11개 rule이 있었고, #8은 `sector='공통' -> 기타`였다.

문제는 production Router의 live input이 request text뿐이라는 점이다. `sector`는 사례집 page/editorial metadata이고 새 요청에 항상 주어진다고 보장할 수 없다. `AgentState`에 임의로 sector를 추가하는 것은 문제를 숨길 뿐이다.

### 6.1 폐기한 진단

먼저 `sector`를 atom vocabulary에서 빼고 clean dev로 E6를 처음부터 재유도해봤다. 9개 text-only rule이 생겼지만 clean test에서:

- answered 101
- correct 75
- wrong 26
- coverage 60.1%
- answered accuracy 74.3%

C-3의 wrong 13 → 26으로 악화됐다.

이 결과를 본 뒤 test 성능에 맞춰 일부 rule을 고르는 행동은 하지 않았다. 이 re-induction은 **진단으로 폐기**했다.

### 6.2 최종 해결 — capability projection

최종 production asset은 이미 frozen된 clean E6를 기준으로 한다.

- runtime-supported atom: `ngram`, `length`
- unsupported: `sector`
- unsupported atom이 들어간 rule 전체를 제거
- replacement rule 재학습 없음
- original rule order 보존

결과:

- source rules 11 → runtime rules 10
- drop: rule #8 하나
- Router matcher와 induced semantics 일치
- C-3 대비 168행 **row-level behavior change 0**

즉 스키마 계약을 정리했지만 모델 행동은 바꾸지 않았다.

## 7. Final clean operating profile

`experiments/results/clean/final_clean_temporal.json`이 최종 source of truth다.

| 항목 | 값 |
|---|---:|
| test | 168 |
| answered | 76 |
| abstained | 92 |
| correct | 63 |
| wrong | 13 |
| coverage | **45.24%** |
| accuracy on answered | **82.89%** |
| Path A / B / C | 15 / 61 / 92 |

Abstention reasons:

- conflicting evidence: 60
- no evidence: 23
- surface only: 7
- ambiguous margin: 2

Routes:

- R1 60
- R2 23
- R3 48
- R5 20
- R8 15
- R9 2

C-3 anchor와 final capability-projected profile의 transition은:

```text
abstain → abstain   92
correct → correct   63
wrong   → wrong     13
changed rows         0
```

## 8. Reproducibility guards

최종 merge에서 다음을 CI로 고정했다.

1. Python 3.9 / 3.11 lint + full tests
2. 테스트 553개 유지
3. clean final profile 76/92/63/13 invariant
4. runtime E6 rule 10개, dropped rule #8 invariant
5. final JSON artifact == fresh deterministic recomputation
6. historical E8~E11a experiment variant set에서 `router-temporal` 제외
7. C-4 fixed 5-case plan drift 검사
8. S5 schema에 verdict/basis가 들어오지 않는지 검사
9. phantom/shared-only factor가 recovery로 우회하지 않는지 검사

최종 C-2~C-5 통합 main commit:

```text
5206b6a92e7da5f42bd8372f1a5fa2a66b9c3075
```

main push CI에서 Python 3.9/3.11 모두 성공했다.

## 9. Claim boundary

이 프로젝트가 주장할 수 있는 것:

- PDF 업무문서를 구조화된 사례/요청 단위로 변환하는 deterministic pipeline을 구현했다.
- group split으로 동일 사안의 dev/test leakage를 제거했다.
- temporal eligibility를 retrieval ranking 전에 적용했다.
- temporal policy와 같은 조건에서 risk calibration을 다시 계산했다.
- LLM이 결정적 조건을 구조화하고 deterministic code가 literal grounding을 검증하는 Agent contract를 구현했다.
- 잘못된 선례 적용을 막는 fail-closed deciding-factor veto를 5-case qualitative audit로 검증했다.
- runtime에서 관측 불가능한 learned metadata rule을 제거하면서 168행 behavior equivalence를 증명했다.

주장하지 않는 것:

- T-serial이 실제 결정 시점을 완벽히 복원한다.
- S5가 abstention을 안전하게 자동 회수한다.
- 5건 audit로 AG-13 failure class를 해결했다.
- 최종 Agent가 모든 legacy 모델보다 성능이 우월하다.
- 실제 금융회사 production traffic에서 운영했다.

## 10. 남은 한계

- coverage 45.24%로 보수적이다.
- action-label precedent evidence가 매우 희소하다.
- T-serial은 proxy이며 날짜 순서 inversion이 존재한다.
- DOUBT/TRUST는 clean dev에서 separation을 지지하지만 유일 최적값으로 튜닝된 값은 아니다.
- S5의 `no_decisive_difference` / recovery 경로는 충분히 검증되지 않았다.
- PDF extraction은 deterministic document structuring이며 OCR/layout model 자체를 연구한 것은 아니다.

제출용 설명에서는 이 한계를 숨기기보다 **왜 fail-closed 계약을 택했는지**와 함께 설명한다.
