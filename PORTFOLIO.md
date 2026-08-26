# DeFactoRule — Enterprise Decision Agent

> **과거 예외 승인 이력에서 조직의 실제 판단 기준을 복원하고, 새 업무 요청에 선례를 적용할 때 결정적 차이·충돌·불확실성을 검증하는 AI Agent Workflow**

## 프로젝트 한 줄 요약

금융규제 사례집 PDF를 구조화된 업무 사례로 변환하고, **명시 규칙 + 과거 선례 검색 + LLM deciding-factor 분석 + deterministic validation + abstention/handoff**를 연결해 잘못된 선례 적용을 줄이는 의사결정 Agent를 설계·구현했다.

- 개인 프로젝트
- Python / FastAPI / PyMuPDF / Anthropic API / pytest / GitHub Actions
- 데이터: 금융당국 사례집 1,095 사례 · 1,122 질의–회답 쌍
- 최종 clean evaluation: test 168건

---

## 1. 해결하려던 문제

기업의 규정에는 원칙이 적혀 있지만 실제 업무 판단은 과거 예외 승인·비조치 이력에 축적된다. 담당자는 새 요청을 받으면 규정뿐 아니라 “과거에는 비슷한 경우 어떻게 처리했는가”를 함께 본다.

문제는 가장 비슷한 선례가 항상 올바른 선례가 아니라는 점이다. 실제 데이터에서 최근접 선례를 그대로 따르는 baseline은 결론이 갈리는 TRAP 구간에서 정확도 0.000이었다.

그래서 목표를 단순 문서 검색이 아니라 다음 workflow로 정의했다.

```text
비정형 업무문서
  → 구조화
  → 규칙/선례 검색
  → temporal eligibility
  → evidence Router
  → LLM deciding-factor analysis
  → deterministic evidence validation
  → decision / abstain / human handoff
```

## 2. 내가 구현한 핵심

### A. 비정형 PDF → 구조화 업무 데이터

사례집 PDF를 단순 텍스트로 저장하는 데서 끝내지 않고 다음 단위까지 결정론적으로 분해했다.

```text
PDF
 → 사례
 → 요청 / 회답
 → 질의–회답 pair
 → 문서 체크박스 기반 label
```

파싱 오류를 failure registry와 regression test로 관리했고, 동일 데이터에서 같은 결과가 재생성되도록 CLI pipeline을 분리했다.

**의미:** Agent가 자연어 문서를 바로 소비하는 것이 아니라, 업무 절차에서 사용할 수 있는 구조화된 case/evidence 단위로 변환한다.

### B. De Facto Rule + precedent retrieval

두 종류의 evidence를 사용했다.

1. **Induced rule** — dev 사례의 결론에서 반복되는 text condition을 역추출
2. **Precedent** — 요청문과 유사한 과거 사례 검색

검색 baseline을 분석하면서 표면 유사도가 높은데 결론이 반대인 사례를 TRAP으로 분리했다. 이 실패를 통해 “retrieval score가 높다 = 적용 가능하다”라는 가정을 버렸다.

### C. Temporal eligibility before ranking

과거 선례 기반 시스템에서 미래 사례를 보는 것은 정보 누수다. 검색 결과를 만든 뒤 미래 문서를 지우는 대신 **후보 pool 자체를 먼저 제한**했다.

```text
eligible_indices(request)
  → candidate subset
  → retrieval ranking
```

실제 결정일자가 모든 사례에 존재하지 않아 `(year, serial)`을 chronology proxy로 사용하는 T-serial 정책을 채택했다. strict-year는 근거 pool을 지나치게 없앴다.

T-serial의 한계도 명시했다. serial 순서는 실제 날짜의 ground truth가 아니므로 “temporal-safe”라고 주장하지 않는다.

### D. Risk-aware Router + Abstention

retrieval similarity만으로 답하지 않고 dev LOO에서 precedent-following error risk를 calibration했다.

Temporal policy와 동일한 조건에서 clean dev를 다시 보정한 결과:

| Band | n | Wrong | Risk |
|---|---:|---:|---:|
| trust | 27 | 2 | 7.4% |
| middle | 9 | 3 | 33.3% |
| doubt | 50 | 23 | 46.0% |

Router는 evidence agreement, conflict, similarity band, rule evidence를 조합해 path를 고르고 근거가 약하면 기권한다.

### E. LLM Deciding-Factor Agent

이 프로젝트에서 LLM은 최종 결론 생성기가 아니다.

LLM이 하는 일:

- 두 요청의 `shared_factors` 구조화
- `only_in_request` 추출
- `only_in_precedent` 추출
- 판단을 가를 수 있는 factor 후보 표시

LLM이 하지 못하게 한 일:

- 최종 verdict
- `applies / differs / unclear`
- 최종 applicability basis

Structured Output schema 자체에서 이 필드를 제거했다.

### F. Deterministic Diff Coverage Gate

LLM이 근거를 말했는지만 확인하면 부족했다. 높은 lexical similarity 사례에서 양쪽에 공통인 문장을 올바르게 인용하고도 **실제 결정 조건은 놓치는 실패**를 발견했기 때문이다.

그래서 factor를 다시 원문에 대조한다.

- factor가 원문에 literal-grounded되는가
- declared side에 실제 존재하는가
- 반대쪽에는 없는가
- shared factor가 실제 양쪽에 존재하는가
- decisive difference가 확인됐는가
- recovery를 주장한다면 실제 diff를 빠짐없이 설명했는가

안전성은 비대칭으로 설계했다.

> **결정적 차이 하나가 확인되면 선례 적용을 막을 수 있지만, “결정적 차이가 없다”고 선례를 회수하려면 모든 차이를 설명해야 한다.**

## 3. 대표 실패와 설계 변경

### 실패 1 — split leakage

기존 행 순서 split에서 test 27건이 dev와 거의 동일한 요청이었다. 날짜 표현을 정규화하면 전부 같은 request group으로 묶였다.

**수정:** normalized group 단위 clean split.

### 실패 2 — 미래 선례

2024년 3월 요청이 2024년 6월 사례를 선례로 검색하는 케이스를 발견했다.

**수정:** temporal eligibility를 ranking 이전에 적용.

### 실패 3 — surface match as evidence

LLM이 실제 원문 문장을 근거로 들었지만 결론을 가르는 clause를 놓쳤다. 단순 grounding 검사는 통과했다.

**수정:** deciding-factor 구조화 + deterministic side/uniqueness/diff validation.

### 실패 4 — runtime schema mismatch

E6가 `sector='공통'`이라는 metadata rule을 학습했지만 live Agent는 request text만 받기 때문에 해당 rule은 runtime에서 절대 실행할 수 없었다.

처음에는 sector 없이 rule을 다시 학습했지만 wrong이 13→26으로 악화됐다. test 결과에 맞춰 rule을 고르는 대신 re-induction을 폐기했다.

**최종 수정:** frozen E6에서 runtime-unobservable rule #8만 whole-rule capability projection으로 제거. 대체 rule을 학습하지 않았고 168행 Router 결과 변화 0건을 확인했다.

## 4. 최종 결과

### Final clean operating profile

최종 조건:

- group-based clean split
- T-serial temporal eligibility
- temporal-matched risk calibration
- runtime-compatible frozen E6
- Router + deterministic validation
- S5는 aggregate 성능을 올리는 recovery가 아니라 fail-closed safety veto

| Metric | Result |
|---|---:|
| test | 168 |
| answered | 76 |
| abstained | 92 |
| correct | 63 |
| wrong | 13 |
| coverage | **45.24%** |
| accuracy on answered | **82.89%** |

답변 정확도만 따로 강조하지 않는다. 82.89%는 **coverage 45.24%**와 한 쌍이다.

### S5 qualitative audit

5건의 preregistered clean 사례를 별도 검증했다.

- 위험/모호 precedent에서 grounded decisive difference를 포착
- opposing precedent가 0인 surface-match 위험 사례도 차단
- 그러나 사전 “적용 가능” 후보까지 decisive difference로 차단

따라서 결론은:

> **잘못된 선례 적용을 막는 safety veto는 채택. 자동 abstention recovery는 미검증.**

## 5. 평가를 운영 관점으로 만든 방법

### 결과보다 먼저 protocol을 고정

- 같은 request group이 dev/test를 넘지 않게 split
- temporal filter를 retrieval 이전에 적용
- calibration source/provenance 고정
- historical experiment와 final clean profile 분리
- test 결과를 보고 threshold를 재튜닝하지 않음

### 결과 artifact를 CI에 연결

최종 CI는 코드 테스트뿐 아니라 다음을 검사한다.

- final 168-row profile invariant
- runtime E6 dropped rule invariant
- committed final JSON == fresh recomputation
- historical experiment variant set 불변
- S5 structured-output schema 불변
- C-4 5-case selection drift

Python 3.9 / 3.11에서 553 tests를 실행하고 main push CI를 통과시켰다.

## 6. 기술 스택

**Language / Service**

- Python
- FastAPI
- Pydantic

**Document / Data**

- PyMuPDF
- JSON / JSONL
- deterministic preprocessing pipeline

**Retrieval / Agent**

- character n-gram IDF retrieval
- dense / hybrid retriever abstraction
- Anthropic structured output
- evidence Router
- deciding-factor Agent
- abstention / handoff contract

**Quality / Evaluation**

- pytest
- GitHub Actions
- Wilson confidence interval
- paired bootstrap / Holm correction in historical model study
- failure registry + executable probes
- frozen result artifacts

## 7. 이 프로젝트에서 보여주고 싶은 역량

### 업무 절차를 Agent Workflow로 바꾸기

“LLM에게 문서를 주고 답을 생성”하는 방식 대신 실제 의사결정 절차를 노드로 나눴다.

```text
입력 구조화
→ evidence retrieval
→ eligibility
→ risk routing
→ deciding-factor analysis
→ deterministic validation
→ decision / abstention
→ trace / evaluation
```

### 비정형 문서를 구조화된 evidence로 바꾸기

PDF 파싱, 사례 분리, 요청·회답 pairing, label extraction을 별도 pipeline과 regression test로 구현했다.

### LLM 출력도 다시 검증하기

LLM이 정확한 문장을 인용했다는 것만으로 신뢰하지 않고, **그 문장이 실제 결론을 가르는 차이인지** deterministic code로 다시 검사한다.

### 실험 실패를 제품 계약으로 연결하기

- 미래 선례 발견 → temporal retrieval contract
- split leakage 발견 → group split
- surface grounding 실패 → deciding-factor gate
- over-abstention 발견 → applicability 단계
- recovery 미검증 → S5를 safety veto로 제한
- runtime metadata mismatch → capability projection

즉 metric을 높이는 패치보다 **실패 원인을 운영 가능한 contract로 바꾸는 것**을 프로젝트의 중심으로 삼았다.

## 8. 한계

- T-serial은 실제 chronology가 아니라 proxy다.
- clean action-label precedent evidence가 희소하다.
- S5 audit은 5건이라 class-wide 성능을 주장할 수 없다.
- safe recovery는 아직 검증되지 않았다.
- Document AI는 PDF 구조화 pipeline이며 OCR/layout model 자체를 학습한 프로젝트는 아니다.
- 실제 고객 production traffic에서 운영한 시스템은 아니다.

이 한계를 포함한 최종 engineering record는 [`docs/28-final-clean-freeze.md`](docs/28-final-clean-freeze.md)와 `experiments/results/clean/`의 frozen artifacts에 남겼다.
