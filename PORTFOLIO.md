# DeFactoRule — Enterprise Decision Agent

> **비정형 금융 문서를 구조화하고, 과거 판단 기준과 선례를 검색한 뒤 결정적 차이·충돌·불확실성을 검증하는 AI Agent Workflow**

## 프로젝트 한 줄 요약

금융규제 사례집을 구조화된 업무 사례로 변환하고, **OCR-aware Document AI + 명시 규칙 + temporal Evidence RAG + LLM deciding-factor 분석 + deterministic validation + abstention/handoff**를 연결해 잘못된 선례 적용을 줄이는 의사결정 Agent를 설계·구현했다.

- 개인 프로젝트
- Python / FastAPI / PyMuPDF / Tesseract OCR / Pydantic / Anthropic Structured Output / pytest / GitHub Actions
- 데이터: 금융당국 사례집 1,095 사례 · 1,122 질의–회답 쌍
- 최종 decision clean evaluation: test 168건
- Document AI evaluation: 60개 금융 요청 × 3 synthetic scan profiles

---

## 1. 해결하려던 문제

기업의 규정에는 원칙이 적혀 있지만 실제 업무 판단은 과거 예외 승인·비조치 이력에 축적된다. 담당자는 새 요청을 받으면 규정뿐 아니라 “과거에는 비슷한 경우 어떻게 처리했는가”를 함께 본다.

문제는 가장 비슷한 선례가 항상 올바른 선례가 아니라는 점이다. 실제 데이터에서 최근접 선례를 그대로 따르는 baseline은 결론이 갈리는 TRAP 구간에서 정확도 0.000이었다.

또 실제 업무 입력은 항상 깨끗한 text PDF가 아니다. scan/image 입력은 OCR 오독을 포함할 수 있고, OCR 결과 안에서 quote가 존재한다는 사실만으로 원문 정확성을 보장할 수 없다.

그래서 목표를 단순 문서 검색이 아니라 다음 workflow로 정의했다.

```text
PDF / Scan / Image
  → native text or OCR-aware Document AI
  → structured extraction / validation
  → temporal Evidence RAG + rules
  → risk Router
  → LLM deciding-factor analysis
  → deterministic evidence validation
  → decision / abstain / human handoff
```

## 2. 내가 구현한 핵심

### A. 비정형 문서 → 구조화 업무 데이터

기존 사례집 PDF를 다음 단위까지 결정론적으로 분해했다.

```text
PDF
 → 사례
 → 요청 / 회답
 → 질의–회답 pair
 → 문서 체크박스 기반 label
```

추가로 live/document input 경로에서는 native PDF와 scan/image를 구분한다.

```text
PDF / image
  ├─ healthy native text → PyMuPDF
  └─ scan / image → Tesseract Korean OCR
        ↓
serial / sector / decision / request + source quote
        ↓
schema / grounding / OCR-confidence validation
        ├─ review_required → stop
        └─ validated → Evidence RAG
```

Tesseract는 교체 가능한 OCR adapter의 baseline이다. OCR 모델을 직접 학습했다고 주장하지 않는다.

### B. De Facto Rule + Temporal Evidence RAG

두 종류의 업무 근거를 사용한다.

1. **Induced rule** — dev 사례에서 반복되는 text condition
2. **Precedent evidence** — 새 요청과 유사한 과거 사례

Evidence RAG는 lexical + dense hybrid retrieval을 사용하고 provenance를 유지한다.

- `evidence_id`
- source / page / serial
- request / outcome
- retrieval score

T-serial temporal eligibility를 **ranking 이전**에 적용하고, similarity floor 0.15 미만은 LLM context에 넣지 않는다.

Clean test 168건 retrieval audit:

| Metric | Result |
|---|---:|
| threshold-passing evidence | 88 / 168 |
| evidence coverage | **52.38%** |
| zero evidence | 80 |
| temporal violation | **0** |
| duplicate evidence-id query | **0** |

`top1 outcome agreement 70.45%`는 human relevance label이 없는 diagnostic이므로 RAG accuracy라고 부르지 않는다.

### C. Risk-aware Router + Abstention

retrieval similarity만으로 답하지 않고 clean dev에서 precedent-following risk를 calibration했다.

| Band | n | Wrong | Risk |
|---|---:|---:|---:|
| trust | 27 | 2 | 7.4% |
| middle | 9 | 3 | 33.3% |
| doubt | 50 | 23 | 46.0% |

Router는 evidence agreement, conflict, similarity band, rule evidence를 조합하고 근거가 약하면 기권한다.

### D. LLM Deciding-Factor Agent

LLM이 하는 일:

- `shared_factors`
- `only_in_request`
- `only_in_precedent`
- 결정적 차이 후보

LLM이 하지 못하게 한 일:

- 최종 verdict
- `applies / differs / unclear`
- 최종 applicability basis

Structured Output schema 자체에서 최종 판단 필드를 제거했다.

### E. Deterministic Diff Coverage Gate

LLM이 정확한 문장을 인용해도 실제 결론을 가르는 조건을 놓칠 수 있었다. 그래서 factor를 다시 원문에 대조한다.

- literal grounding
- declared side 존재
- opposite-side absence
- shared factor 양쪽 존재
- decisive difference
- recovery 시 전체 diff coverage

안전성은 비대칭이다.

> **결정적 차이 하나가 확인되면 선례 적용을 막을 수 있지만, “차이가 없다”고 회수하려면 모든 차이를 설명해야 한다.**

## 3. 실패가 설계를 바꾼 사례

### Split leakage

기존 행 순서 split에서 test 27건이 dev와 사실상 동일 요청이었다.

**수정:** normalized request group 단위 clean split.

### Future precedent

2024년 3월 요청이 2024년 6월 사례를 검색하는 케이스를 발견했다.

**수정:** temporal eligibility를 ranking 이전에 적용.

### Surface-match as evidence

LLM이 실제 원문 문장을 인용했지만 결론을 가르는 clause를 놓쳤다.

**수정:** deciding-factor 구조화 + deterministic side/uniqueness/diff validation.

### Runtime schema mismatch

E6가 live request에서 관측할 수 없는 `sector='공통'` 규칙을 학습했다. sector 없이 re-induction했더니 wrong이 13→26으로 악화됐다.

**최종 수정:** frozen E6에서 runtime-unobservable rule #8만 whole-rule capability projection. 대체 rule을 학습하지 않았고 168행 Router 결과 변화 0건.

### OCR grounding blind spot

60-document realistic benchmark에서 OCR 결과 안의 quote grounding만으로는 **OCR 자체의 오독**을 잡지 못한다는 문제가 드러났다.

첫 평가 후 post-gate 결과를 보기 전에 다음 quality gate를 고정했다.

- mean OCR word confidence < 80 → review
- confidence < 60 token 비율 > 20% → review

재평가에서 degraded scan은 오류 문서의 58.33%를 review로 보냈지만 clean/standard의 high-confidence 오독 detection recall은 1.69% / 10.53%에 그쳤다.

**결론:** confidence는 보조 신호로 채택하지만 ground-truth correctness 보증으로 사용하지 않는다. threshold를 결과에 맞춰 다시 튜닝하지 않았다.

## 4. 최종 결과

### Decision Agent - final clean operating profile

| Metric | Result |
|---|---:|
| test | 168 |
| answered | 76 |
| abstained | 92 |
| correct | 63 |
| wrong | 13 |
| coverage | **45.24%** |
| accuracy on answered | **82.89%** |

82.89%는 반드시 **coverage 45.24%**와 함께 읽는다.

S5 qualitative audit은 안전 veto로는 채택했지만 safe recovery는 검증되지 않아 aggregate 168건 recovery에 사용하지 않았다.

### OCR-aware Document AI - frozen synthetic scan benchmark

60개 실제 금융 request text를 clean-test pool 전반에서 고정 선택하고 세 가지 scan profile로 rasterize했다. 쉬운 scalar field만 평가하지 않고 serial / sector / decision / 긴 request 전체를 exact field metric에 포함했다.

| Profile | Request char acc. | Field F1 exact | Review rate | Error-detection recall |
|---|---:|---:|---:|---:|
| clean 220dpi PNG | **94.38%** | **75.42%** | 1.67% | 1.69% |
| standard 170dpi JPEG | **89.79%** | **75.11%** | 10.00% | 10.53% |
| degraded 120dpi JPEG | **93.53%** | **62.44%** | 58.33% | 58.33% |

이 benchmark는 **synthetic scanned-document benchmark**이며 실제 고객 스캔 성능을 의미하지 않는다. request char accuracy가 scan 난이도에 따라 단조 감소하지 않은 결과도 측정값 그대로 보존했다.

## 5. 평가와 운영 계약

### 결과보다 protocol을 먼저 고정

- same request group dev/test 분리
- temporal filter before ranking
- temporal-matched calibration
- retrieval similarity floor 0.15
- OCR confidence gate preregistration before post-gate benchmark
- test 결과를 보고 threshold 재튜닝 금지

### 결과 artifact를 CI에 연결

Core CI:

- **577 collected = 566 passed + 11 skipped**
- Python 3.9 / 3.11
- Evidence RAG offline audit
- final 168-row profile invariant
- frozen artifact regression guards

Document AI:

- **17 dedicated checks**
- real Tesseract Korean OCR contract test
- native/scan detection
- structured extraction schema
- hallucinated/ungrounded quote rejection
- valid document → temporal RAG bridge
- invalid/review document → RAG 호출 금지
- 60-document × 3-profile benchmark on CI

## 6. 기술 스택

**Language / Service**

- Python
- FastAPI
- Pydantic

**Document AI / Data**

- PyMuPDF
- Tesseract OCR (`kor` baseline)
- native / OCR intake routing
- structured field + provenance quote extraction
- OCR confidence / grounding validation
- JSON / JSONL

**Retrieval / Agent**

- character n-gram IDF retrieval
- dense / hybrid retriever abstraction
- temporal Evidence RAG
- Anthropic Structured Output
- evidence Router
- deciding-factor Agent
- abstention / handoff contract

**Quality / Evaluation**

- pytest
- GitHub Actions
- CER / exact field F1 / document exact-match
- risk calibration / Wilson confidence interval
- paired bootstrap / Holm correction in historical model study
- failure registry + executable probes
- frozen result artifacts

## 7. 이 프로젝트에서 보여주고 싶은 역량

### 업무 절차를 Agent Workflow로 바꾸기

```text
문서 intake
→ structured extraction / validation
→ evidence retrieval
→ temporal eligibility
→ risk routing
→ deciding-factor analysis
→ deterministic validation
→ decision / abstention / handoff
→ trace / evaluation
```

### LLM/RAG/OCR을 각각 검증 가능한 경계로 두기

- OCR은 text source를 만들지만 correctness를 스스로 보증하지 않는다.
- RAG는 evidence를 제공하지만 similarity를 applicability로 간주하지 않는다.
- LLM은 factor를 구조화하지만 최종 verdict를 직접 생성하지 않는다.
- downstream code가 provenance, grounding, temporal, conflict contract를 검사한다.

### 실험 실패를 제품 계약으로 연결하기

- 미래 선례 → temporal retrieval contract
- split leakage → group split
- surface grounding 실패 → deciding-factor gate
- over-abstention → applicability 단계
- recovery 미검증 → S5 safety veto
- runtime metadata mismatch → capability projection
- OCR high-confidence 오독 → confidence gate의 한계를 명시하고 독립 검증 채널 필요성 도출

## 8. 한계

- T-serial은 실제 chronology가 아니라 proxy다.
- clean action-label precedent evidence가 희소하다.
- S5 audit은 5건이며 safe recovery는 검증되지 않았다.
- OCR 평가는 synthetic scans 60건 × 3 profiles이며 실제 고객 문서 성능이 아니다.
- Tesseract pretrained baseline을 사용했고 OCR detector/recognizer를 직접 학습·fine-tuning하지 않았다.
- table structure recognition과 image-VLM benchmark는 수행하지 않았다.
- optional LLM structured extraction은 schema/contract까지 구현했지만 live extraction 품질·비용은 이번 phase에서 측정하지 않았다.
- OCR confidence는 clean/standard high-confidence transcription error를 충분히 검출하지 못했다.
- 실제 고객 production traffic에서 운영한 시스템은 아니다.

상세 frozen record:

- [`docs/28-final-clean-freeze.md`](docs/28-final-clean-freeze.md)
- [`docs/29-evidence-rag.md`](docs/29-evidence-rag.md)
- [`docs/30-document-ai-ocr.md`](docs/30-document-ai-ocr.md)
- [`docs/31-portfolio-capture-guide.md`](docs/31-portfolio-capture-guide.md)
- `experiments/results/clean/`
