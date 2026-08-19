# Phase 3 — 업무 의사결정 Agent Workflow 설계

> 이 문서는 **코드를 쓰기 전에** 쓴다. 여기 적힌 수치는 전부
> `experiments/results/*.json` 에서 읽은 실측값이고, 아직 실행하지 않은 것은
> "예상" 이라고 명시한다.

---

## 1. 목표 및 현재 Phase 2와의 연결

### 무엇이 바뀌는가

| | Phase 2 (지금) | Phase 3 (여기) |
|---|---|---|
| 질문 | 문서에 없는 판단 기준을 **복원할 수 있는가** | 요청이 들어왔을 때 **무엇을 근거로 판단할지 스스로 정할 수 있는가** |
| 산출 | 7개 모델의 라벨 예측 + 점수 | 판단 + **선택한 근거 경로** + 실행 흔적 |
| 실패의 의미 | 라벨이 틀렸다 | **틀릴 자리에서 안 멈췄다** |

### 왜 지금 이것인가 — E5 가 이미 지목했다

이건 새로 만들어 낸 문제의식이 아니다. E5 의 실측이 그대로 요구한다.

```
test 170건 · 선례 풀 = dev 85건 · 닮음 문턱 0.15 · 문자 4-gram IDF 코사인

  순응(AGREE)    72건    최근접 선례의 결론 = 정답
  함정(TRAP)     15건    최근접 선례의 결론 ≠ 정답
  선례 없음      83건

  neighbor 모델   전체 0.724 · 순응 1.000 · 함정 0.000
```

**닮은 선례를 따라가는 전략은 순응 구간에서 100%, 함정 구간에서 0%다.** 평균을
내면 그럴듯해 보이고, 결론이 갈리는 자리에서 전부 틀린다. 그리고 눈이 머는
자리가 하필 제일 비싼 쪽이다.

```
정답 클래스별로, dev 에 닮은 선례가 있는 비율

  조치     1/14  =  7.1%
  기타    16/30  = 53.3%
  비조치  70/126 = 55.6%
```

제재로 이어지는 `조치` 는 드물고, 드문 만큼 선례도 없다. **검색 기반 접근은
가장 비용이 큰 판단에서 구조적으로 무력하다.**

그러므로 이 도메인에서 Agent 의 핵심 능력은 답을 잘 만드는 것이 아니라
**"지금 검색 결과를 믿어도 되는가"를 판정하는 것**이다. Phase 3 은 그 판정을
시스템의 1급 시민으로 올린다.

### Phase 2 산출물이 Phase 3 에서 맡는 역할

| Phase 2 자산 | Phase 3 에서 |
|---|---|
| `confusable.py` — 4-gram IDF 코사인, `nearest`, `partition` | Retriever(L) 와 Router 신호의 계산 근거 |
| TRAP 지표 · `anchoring_by_class` | Router 의 **목적 함수**. 함정 구간 정확도가 개선 여부의 판정선 |
| E6 유도 규칙 11개 (`e6_rules.json`) | PATH B 의 근거 원천 |
| `selective.py` — 위험-커버리지, AURC | 기권 게이트의 평가. 그대로 재사용 |
| `comparison.py` — 짝지은 부트스트랩 + Holm | E8~E11 통계 처리. 그대로 재사용 |
| 실패 레지스트리 67건 + probe | Phase 3 실패도 같은 레지스트리에 들어간다 |
| `validity.py` — 무효 기준 | E8~E11 수치의 게이트 |
| dev/test 결정론적 분할 · 누출 제거 | 그대로. **새 분할을 만들지 않는다** |

---

## 2. 전체 Agent Workflow Diagram

```
                          ┌──────────────────────────────┐
  요청문 (test 170건)  ──▶ │  Retriever                   │
  선례 풀 = dev 85건       │   L  문자 4-gram IDF 코사인   │  (기존 코드)
                          │   D  LSA (SVD, k=128)        │  (신규, numpy만)
                          │   H  RRF(L, D)               │  (신규)
                          └──────────────┬───────────────┘
                                         │ Evidence[] (선례 후보 top-k)
                                         ▼
                          ┌──────────────────────────────┐
                          │  Rule Matcher                │  (기존 E6 규칙)
                          │   유도 규칙 11개 매칭         │
                          └──────────────┬───────────────┘
                                         │ Evidence[] (규칙 후보)
                                         ▼
     ┌───────────────────────────────────────────────────────────────┐
     │  Router  — 전부 결정론. dev 에서만 보정.                        │
     │    신호: evidence_count · top_similarity · margin ·            │
     │          label_agreement · source_diversity · recency_gap ·   │
     │          rule_fired · rule_conflict · **trap_risk**           │
     └───────┬───────────────────────┬───────────────────────┬───────┘
             │ PATH A                │ PATH B                │ PATH C
             ▼                       ▼                       ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ precedent-driven │   │ rule-driven      │   │ abstain          │
   │ 선례 결론을 따름  │   │ 유도 규칙 결론    │   │ 판단 보류        │
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            └──────────┬───────────┘                      │
                       ▼                                  │
     ┌───────────────────────────────────────────┐        │
     │  Validator — 6종                          │        │
     │   1 schema          4 rule consistency    │        │
     │   2 evidence exists 5 unsupported claim   │        │
     │   3 source consist. 6 conflict detection  │        │
     └──────────────────┬────────────────────────┘        │
                        │ 실패 → 강등                      │
                        ▼                                  ▼
     ┌───────────────────────────────────────────────────────────┐
     │  Abstention Gate — 결정론. 모델은 관여하지 않는다.           │
     │   route == C · validator 실패 · 신뢰도 < 문턱 · 근거 소진   │
     └──────────────────────────┬────────────────────────────────┘
                                ▼
                  Result + Execution Trace
                  { decision | ABSTAIN, confidence, route,
                    evidence_used[], abstention_reason,
                    trace[ node, input, output, elapsed ] }
```

**LLM 이 개입하는 자리는 단 한 곳이다** — Validator 3번(source consistency)의
심화 검사, "이 선례가 이 요청에 실제로 적용되는가". 나머지는 전부 결정론이다.
그 이유는 §7 과 §9(E11)에 적는다.

---

## 3. State Schema

Pydantic 모델로 둔다. **한 요청의 처리 전체가 이 객체 하나에 남고, 그것만으로
재현된다.**

```python
class Evidence(BaseModel):
    id: str                      # "prec:2024비조치#017" | "rule:e6#3"
    kind: Literal["precedent", "rule"]
    source: str                  # 사례집 파일명 | "e6_rules"
    serial: Optional[str]
    year: Optional[int]
    label: str                   # 이 근거가 가리키는 결론
    score: float                 # 선례=유사도 · 규칙=dev 정밀도
    rank: int
    retriever: Optional[str]     # "L" | "D" | "H" | None(규칙)
    quote: Optional[str]         # 원문에서 그대로 딴 구절

class RouterSignals(BaseModel):
    evidence_count: int
    top_similarity: float
    margin: float                # top1 - top2 유사도
    label_agreement: float       # top-k 중 top1 라벨과 같은 비율
    source_diversity: int        # top-k 의 서로 다른 사례집 수
    recency_gap: Optional[int]   # 요청 연도 - 선례 연도
    rule_fired: int
    rule_conflict: bool
    trap_risk: float             # dev 에서 보정한 "선례를 따르면 틀릴 확률"

class AgentState(BaseModel):
    request: str
    request_key: tuple           # (source, page, serial, pair_index)
    retrieved_evidence: list[Evidence] = []
    rule_evidence: list[Evidence] = []
    signals: Optional[RouterSignals] = None
    precedent_score: float = 0.0
    route: Optional[Literal["A", "B", "C"]] = None
    route_reason: Optional[str] = None      # 결정 표의 어느 줄이 발화했는가
    decision: Optional[str] = None          # 비조치 | 조치 | 기타
    confidence: Optional[str] = None        # high | medium | low
    validation: list[ValidationResult] = []
    abstained: bool = False
    abstention_reason: Optional[str] = None
    execution_trace: list[TraceStep] = []
```

**`route_reason` 이 결정 표의 줄 번호를 그대로 담는 것이 핵심이다.** "왜 이
경로를 골랐는가"를 사람이 아니라 데이터가 답해야 한다.

---

## 4. Router Decision Table

### 4-1. trap_risk — 이 설계의 심장 · **실측 완료**

> **trap_risk** = 이 요청에 대해 **최근접 선례를 따르면 틀릴 확률**의 dev 추정치

추론 시점에 알 수 있는 것만 조건으로 쓴다. dev 85건을 **leave-one-out** 으로
재본다(자기 자신을 빼지 않으면 유사도 1.0 이 나와 "언제나 믿어라" 가 된다).

```
dev 85건 · LOO · 선례를 그대로 따랐을 때 전체 오류율 0.329

  구간                건수   오류   위험     95% CI
  trust  (>= 0.60)     32     2   0.062   [0.017, 0.201]
  middle (0.15~0.60)    5     2   0.400   [0.118, 0.769]
  doubt  (<  0.15)     48    24   0.500   [0.364, 0.636]
```

**믿음 구간의 상한 0.201 이 못믿음 구간의 하한 0.364 보다 낮다 — 겹치지
않는다.** 유사도는 진짜 신호이고, 설계의 전제는 살았다.

#### 초안을 버렸다 — 2차원 표는 쓸 수 없다

초안은 (유사도 구간 × 선례가 가리키는 라벨) 2차원 표를 쓰려 했다. E5 가
클래스별 앵커링 차이를 보였으니 라벨도 신호일 것이라고 **가정**했다. 재보니
칸 8개 중 **4개가 2건 이하**였다. 그 표로 문턱을 정하면 한두 건이 정책을
결정한다.

라벨 단독으로는 신호가 있다(비조치 0.214 · 기타 0.611 · 조치 0.455). 그러나
유사도와 겹치면 칸이 무너지고, 유사도만으로도 구간이 갈린다. **그래서 유사도
1차원으로 간다.** 자유 변수를 줄이는 쪽이 dev 85건에서 과적합을 덜 한다.
(레지스트리 AG-10)

#### 구간을 셋으로만 자르는 이유

85건 중 **80건이 두 끝에 몰려 있다**(48 + 32). 가운데는 5건뿐이다. 중간
문턱을 미세 조정하는 것은 5건을 이리저리 옮기는 일이다. 그러므로 세 구간으로
족하고, 문턱은 `app/domain/similarity.py` 한 곳에서만 정한다.

재현: `python3 scripts/calibrate.py` → `experiments/results/trap_risk.json`

### 4-2. 결정 표

위에서부터 먼저 맞는 줄이 이긴다. 전부 결정론이고, 모든 문턱은 **dev 에서만**
정한다.

| # | 조건 | PATH | 이유 |
|---|---|---|---|
| R1 | `rule_conflict` = True | **C** | 규칙끼리 반대를 가리키면 판단할 근거가 없다 |
| R2 | `evidence_count` = 0 **AND** `rule_fired` = 0 | **C** | 근거가 아예 없다 |
| R3 | `evidence_count` = 0 (규칙은 있다) | **B** | 선례가 닿지 않는다. 규칙으로 간다 |
| R5 | `trap_risk` > `RISK_CEILING` | **B** if `rule_fired` else **C** | **표면만 닮았을 위험이 크다 — 선례를 버린다** |
| R6 | `label_agreement` < `MIN_AGREEMENT` **AND** `source_diversity` ≥ 2 | **C** | 서로 다른 출처가 서로 다른 결론을 가리킨다 |
| R7 | `recency_gap` > `MAX_YEAR_GAP` **AND** `rule_fired` ≥ 1 | **B** | 오래된 선례보다 현행 규칙 |
| R9 | `margin` < `MIN_MARGIN` **AND** `label_agreement` < 1.0 | **C** | 1·2등이 붙어 있는데 결론이 갈린다 |
| R8 | `top_similarity` ≥ `TRUST` | **A** | 매우 닮았고 함정 위험이 낮다 |
| R10 | 그 밖에 `evidence_count` ≥ 1 | **A** | 남은 경우 — 선례를 따르되 Validator 가 본다 |

**초안의 R4 는 지웠다.** "선례도 규칙도 없다" 는 R2 가 이미 잡으므로 **한 번도
발화할 수 없었다.** 죽은 줄을 찾는 검사를 쓰다 드러났다 — 그리고 그 검사의 첫
판은 `return` 뒤만 보는 정규식이어서 삼항식 뒤쪽의 R4 를 **놓쳤다.** AST 로
바꿔 잡히는 것을 확인했다.

문턱 값과 근거:

| 기호 | 값 | 근거 |
|---|---|---|
| `TRUST` | 0.60 | dev LOO 오류율 0.062 [0.017, 0.201] |
| `DOUBT` | 0.15 | dev LOO 오류율 0.500 [0.364, 0.636] · 위 구간과 비중첩 |
| `RISK_CEILING` | 0.20 | 믿음 구간 오류율의 95% 상한(0.201)에 맞췄다 |
| `MIN_AGREEMENT` | 0.60 | 상위 근거의 과반이 같은 결론을 가리켜야 한다 |
| `MIN_MARGIN` | 0.02 | 1·2등 유사도가 이보다 붙으면 순서가 의미 없다 |
| `MAX_YEAR_GAP` | 5 | **데이터에서 나오지 않았다.** 코퍼스가 2021~2025 이므로 5년 초과는 현재 세대 이전이라는 뜻으로 잡았다 — 근거가 약하고, 그 사실을 적어 둔다 |

문턱은 전부 `app/domain/similarity.py` 와 `app/agents/router.py` 에서만 정한다.
세 파일에 흩어져 값이 어긋나던 문제는 IN-13 으로 정리했다.

---

## 5. Evidence Schema

§3 의 `Evidence` 가 스키마이고, 여기서는 **어떻게 채워지는가**를 정한다.

| 필드 | 선례 근거 | 규칙 근거 |
|---|---|---|
| `id` | `prec:{source}#{serial}` | `rule:e6#{order}` |
| `label` | 그 선례의 실제 결론(dev 라벨 — 문서 체크박스) | 규칙이 가리키는 라벨 |
| `score` | 코사인 유사도 | `dev_precision` |
| `quote` | 선례 요청문에서 겹치는 최장 구절 | 규칙의 n-gram 원형 |
| `retriever` | `L` / `D` / `H` | `None` |

**선례의 라벨은 dev 에서 온다.** dev 는 정답이 있는 집합이고, test 는 없다고
가정한다. test-test 이웃은 절대 만들지 않는다 — 정답끼리 정보가 오간다.

`quote` 는 Validator 3번이 대조할 대상이다. 없으면 그 근거는 검증을 통과할 수
없다.

---

## 6. Abstention Rules

기권은 **결정론이다. 모델은 신뢰도 등급까지만 말한다.**

| 코드 | 조건 | 언제 |
|---|---|---|
| `NO_EVIDENCE` | 선례도 규칙도 없다 | R2 · R4 |
| `CONFLICTING_EVIDENCE` | 규칙 충돌 또는 다출처 라벨 불일치 | R1 · R6 |
| `SURFACE_ONLY` | `trap_risk` 높고 대체 근거 없음 | R5 (규칙 미발화) |
| `AMBIGUOUS_MARGIN` | 1·2등 근거가 붙어 있다 | R9 |
| `VALIDATION_FAILED` | Validator 가 막았다 | §7 |
| `LOW_CONFIDENCE` | 최종 신뢰도 < dev 문턱 | 게이트 |

**기권률 자체를 목표로 삼지 않는다.** 전부 기권하면 위험이 0 이 되므로,
평가는 반드시 위험-커버리지 곡선 전체로 한다(§9).

---

## 7. Validator Rules

여섯 개. **다섯 개는 결정론이고 공짜다.** 여섯 번째만 LLM 을 쓰며, 그것도
선택 항목이다.

| # | 검사 | 방법 | 실패 시 |
|---|---|---|---|
| V1 | schema | Pydantic | 즉시 기권 |
| V2 | evidence existence | 인용한 `id` 가 검색 결과 안에 실재하는가 | 즉시 기권 |
| V3 | source consistency | `quote` 가 원문에 글자 그대로 있는가 (`quote_is_grounded` 재사용) | 그 근거 폐기 |
| V4 | rule consistency | 결정이 발화한 고정밀 규칙(dev precision ≥ 0.95)과 반대인가 | 강등 → C |
| V5 | unsupported claim | 결정 라벨을 가리키는 근거가 하나도 없는가 | 즉시 기권 |
| V6 | conflict detection | 문턱 이상의 근거들이 서로 다른 라벨을 가리키는가 | 신뢰도 강등 |
| **V3+** | *(선택)* applicability | **LLM: "이 선례가 이 요청에 적용되는가 — 두 사안의 차이가 결론을 바꿀 만한가"** | 근거 폐기 → 재라우팅 |

**V3+ 가 이 프로젝트에서 LLM 이 값어치를 하는 유일한 자리다.** 표면 유사도가
할 수 없는 판단이고, 정확히 TRAP 구간을 겨냥한다. 그래서 E11 의 실험 대상이며,
**결정론 검증만으로 먼저 결과를 내고 그 다음에 붙인다**(§10).

폐기된 근거는 조용히 사라지지 않는다 — `app/core/audit.Discards` 에 이유와 함께
남는다(IN-02·EV-16 에서 배운 것).

---

## 8. Failure Scenarios

전부 **합성 fixture 로 먼저 만든다. API 호출 없이.** 각 시나리오는 단위
테스트 하나 + 실패 레지스트리 항목 하나가 된다.

| # | 시나리오 | fixture 만드는 법 | 기대 동작 |
|---|---|---|---|
| F1 | 근거 0건 | 선례 풀을 비우고 규칙도 안 맞는 요청 | `C` / `NO_EVIDENCE` |
| F2 | 근거 1건 (약함) | 유사도 0.16 짜리 하나만 | `trap_risk` 에 따라 A 또는 B, **절대 무조건 A 가 아님** |
| F3 | 근거 다수 일치 | 같은 라벨 선례 5건 | `A`, 신뢰도 high |
| F4 | 근거 충돌 (다른 출처) | 유사도 비슷한 선례가 비조치/조치로 갈림 | `C` / R6 / `CONFLICTING_EVIDENCE` |
| F4b | 근거 충돌 (같은 출처) | R6 이 못 잡는 자리 — 1·2등이 붙어 있고 결론이 갈림 | `C` / R9 / `AMBIGUOUS_MARGIN` |
| F5 | **잘못된 선례 (TRAP)** | dev 의 실제 함정 15건을 그대로 씀 | `trap_risk` 가 높게 나와 **B 또는 C 로 빠져야 함** |
| F6 | 부분 근거 | 선례는 있으나 `quote` 가 원문에 없음 | V3 가 근거 폐기 → 재라우팅 또는 기권 |
| F7 | 버전 불일치 | 2021년 선례 vs 2025년 요청 | R7 로 B, `recency_gap` 이 trace 에 남음 |
| F8 | 모호한 요청 | 요청문 30자 미만 / 질의가 여러 개 | `margin` 작음 → `C` / `AMBIGUOUS_MARGIN` |
| F9 | **조치 앵커링** | 정답이 `조치` 인 test 14건 | 선례가 1건뿐이므로 대부분 B/C 로 가야 함 |

**F5 와 F9 가 이 Phase 의 존재 이유다.** 나머지가 다 통과해도 이 둘이 안 되면
Phase 3 은 실패다.

---

## 9. E8~E11 Evaluation Protocol

### 9-0. 공통 규약 — 먼저 못 박는다

- **표본**: test 170건 고정. 새 분할 없음. `조치` 14건.
- **선례 풀**: dev 85건 고정. test-test 이웃 금지.
- **보정**: 모든 문턱은 dev LOO 에서만. test 는 한 번만 본다.
- **통계**: 짝지은 부트스트랩 5,000회 + Holm 보정. `comparison.py` 재사용.
- **주 종점(primary endpoint) 하나를 미리 정한다** — **함정 구간(TRAP) 정확도**.
  나머지는 부차 종점이며 다중비교 보정 후 유의성을 주장하지 않는다.
  (지표를 여럿 재고 이긴 것을 고르면 그건 실험이 아니다.)
- **무효 기준**(`validity.Claim`): 부분 표본 · 모집단 대비 치우침 · 어느 쪽
  판정도 불가능한 분모 — 하나라도 걸리면 **판정을 내지 않는다.**
- `조치` 14건에 대한 주장은 전부 Wilson 구간으로 말한다. 점추정 단독 금지.

### 9-1. 실험

| | 비교 | 무엇을 묻는가 | API 비용 |
|---|---|---|---|
| **E8** | Naive RAG ↔ Router | 검색 결과를 무조건 믿는 것 대비 나아지는가 | **$0** |
| **E9** | Always-precedent ↔ Router | 경로 선택 자체가 값어치가 있는가 | **$0** |
| **E10** | Router (기권 없음) ↔ Router + 기권 | 기권이 위험을 실제로 낮추는가 | **$0** |
| **E11a** | Router ↔ Router + 결정론 Validator | 검증 5종이 값어치가 있는가 | **$0** |
| **E11b** | *(선택)* + LLM applicability (V3+) | 의미 판단이 TRAP 을 깨는가 | **약 $2.4 (추정)** |

- **E8 의 Naive RAG** = top-1 선례의 라벨을 그대로 답한다. 이것이 곧 기존
  `neighbor` 모델이므로 **이미 계산돼 있다**(`pred_nonaction_neighbor.jsonl`).
- **E9 의 Always-precedent** = Router 를 R10 하나만 남긴 것. 항상 A.
- 검색기 세 종(L·D·H)은 E8 안에서 함께 비교한다. 별도 실험을 만들지 않는다.

### 9-2. 지표 — 네 실험 모두 같은 표로

| 지표 | 정의 | 이미 있는 코드 |
|---|---|---|
| Decision Accuracy | 답한 것 중 맞은 비율 (커버리지 명시) | `metrics.macro_f1` |
| **TRAP Accuracy** ★주 종점 | 함정 15건 위에서의 정확도 | `confusable.accuracy` |
| Abstention Accuracy | 기권한 건 중, 답했다면 틀렸을 비율 | 신규 (작음) |
| Unsupported Claim Rate | V5 를 어긴 결정의 비율 | 신규 (작음) |
| Coverage | 답한 비율 | `selective.operating_points` |
| Risk-Coverage / AURC | 곡선 전체 | `selective.aurc` |
| Failure Type 분포 | 오답을 taxonomy 로 분류 | `failure_taxonomy` |

**Abstention Accuracy 를 단독으로 자랑하지 않는다.** 전부 기권하면 100%가
된다. 반드시 Coverage 와 짝으로만 적는다.

### 9-3. 사전 등록 — 성공/실패를 지금 적는다

**주 종점**: Router 의 TRAP 정확도가 Naive RAG(=`neighbor`, 실측 **0.000**)보다
높은가.

| 결과 | 해석 |
|---|---|
| TRAP 정확도 구간 하한 > 0 | Router 가 표면 선례 함정을 실제로 깬다 ✅ |
| 구간이 0 을 포함 | 판정 보류. 15건으로는 못 가른다 |
| 전체 정확도는 오르고 TRAP 은 그대로 | **개선이 아니라 다수 클래스 이득** — 그대로 적는다 |

**미리 인정하는 것**: 함정 15건이다. Wilson 구간으로 `0/15` 는 [0, 0.20],
`4/15` 는 [0.11, 0.52]. **4/15 이상이어야 하한이 0 을 넘는다.** 그 아래면
"못 깼다" 가 아니라 "못 가른다" 이고, 그렇게 적는다.

---

## 10. Cost / Data Reuse Plan

### 10-1. 지금 있는 것으로 어디까지 가는가

| 필요한 것 | 있는가 |
|---|---|
| test 170건 요청문 + 정답 | ✅ `data/eval/nonaction_test.jsonl` |
| dev 85건 (선례 풀 + 보정) | ✅ `data/eval/nonaction_dev.jsonl` |
| Naive RAG 예측 | ✅ `pred_nonaction_neighbor.jsonl` |
| 유도 규칙 11개 | ✅ `experiments/results/e6_rules.json` |
| 함정/순응 구간 정의 | ✅ `experiments/results/trap.json` |
| 유사도·IDF·최근접 | ✅ `app/evaluation/confusable.py` |
| 위험-커버리지·AURC | ✅ `app/evaluation/selective.py` |
| 짝지은 부트스트랩 + Holm | ✅ `app/evaluation/comparison.py` |

**E8·E9·E10·E11a 는 전부 $0 이다.** 새 코퍼스도, 새 라벨링도, 새 API 호출도
필요 없다.

### 10-2. 새 데이터 — 최소한

| 무엇 | 크기 | 왜 |
|---|---|---|
| 합성 fixture (F1~F8) | 8종 × 각 3~5건 | Workflow 극단 검증. 손으로 만든다 |
| F9 fixture | 없음 — test 의 실제 `조치` 14건 사용 | 합성으로 대체하면 의미가 없다 |

**새 문서 코퍼스는 만들지 않는다.**

### 10-3. API 호출 — 순서를 고정한다

E11b 만 API 를 쓴다. 그리고 §10 지시대로 **fixture 를 먼저 전부 통과한 뒤에만**
호출한다.

```
1. F1~F9 fixture 전부 통과      비용 0     ← 여기가 게이트
2. dev LOO 로 문턱 보정          비용 0
3. E8·E9·E10·E11a 실행           비용 0     ← 여기서 결과가 나온다
4. (선택) V3+ dry-run            비용 0     프롬프트·비용만 출력
5. (선택) V3+ dev 5건            약 $0.08   실제 단가 측정
6. (선택) V3+ 본 실행            추정 $2.4  단가 × 실제 대상 건수로 재계산
```

**5번 없이 6번으로 가지 않는다.** 추정은 실측 앞에서 물러난다.

### 10-4. 쓰지 않기로 한 기술과 그 이유

| 기술 | 판단 | 근거 |
|---|---|---|
| **Elasticsearch** | 쓰지 않음 | 선례 풀이 **85건**이다. 전체 코퍼스라 해도 1,095건. 메모리 안에서 전수 비교가 밀리초 단위다. ES 는 여기서 문제를 하나도 풀지 않고 운영 부담만 만든다. *바뀔 조건*: 코퍼스가 10만 건을 넘거나, 다중 사용자 동시 질의가 요구될 때 |
| **LangGraph** | 쓰지 않음 (Phase 3) | 이 워크플로는 분기 3개짜리 DAG 이고 순환이 없다. 필요한 건 **결정론적 재현**인데, LangGraph 는 그것을 더 쉽게 만들지 않고 상태 직렬화 계층을 하나 더 얹는다. `AgentState` + `execution_trace` 로 직접 구현하는 편이 재현·테스트·디버깅 모두 낫다. *바뀔 조건*: 사람 개입(human-in-the-loop) 중단·재개나 노드 병렬 실행이 필요해질 때. Phase 9~10 에서 재검토 |
| **사전학습 임베딩** (sentence-transformers) | **보류 — 검증 후 결정** | torch + 모델 400MB 다운로드가 필요하고, 이 환경에서 아직 확인 못 했다. 대신 dense 는 **코퍼스 자체에서 만든 LSA(SVD, numpy)** 로 구현한다. 다운로드 없이 재현되고, "의미 유사도가 표면 유사도보다 함정을 잘 피하는가" 라는 **질문 자체는 그대로 물을 수 있다.** *한계는 §설계의 문제점 4번에 적는다* |
| **PostgreSQL** | 쓰지 않음 | 저장할 상태가 JSONL 로 충분하다 |
| **Docker** | Phase 10 | Phase 3 의 문제가 아니다 |

---

## 검색기 비교 — **실측 완료.** 검색을 바꿔서 풀 문제가 아니다

설계서 §문제점 4에 이렇게 적어 뒀다.

> L 도 D 도 결국 이 코퍼스의 표면 통계다. **둘 다 TRAP 을 못 깰 가능성이 높다.**
> 그리고 그게 사실이면 그것이야말로 Router 가 필요한 이유의 증명이 된다.

재봤다. 사실이었다.

```
선례 풀 dev 85건 · 평가 test 170건 · 코퍼스 256건 · 문턱 0.15

  검색기        순응   함정   선례 없음   함정 비율
  L             72     15      83       0.172
  D             71     32      67       0.311
  H(L+D)        59     13      98       0.181
```

선례를 그대로 따르는 전략의 함정 구간 정확도는 **정의상 0** 이다. 그러므로
비교할 수는 함정 구간의 크기, 곧 **찾았을 때 틀릴 비율**이다.

### D 는 더 많이 찾고 더 많이 틀린다

| 검색기 | 라벨 | 건수 | 선례 있음 | 함정 | 함정 비율 |
|---|---|---|---|---|---|
| L | `비조치` | 126 | 70 | 6 | 0.086 |
| L | **`조치`** | 14 | **1** | 0 | 0.000 |
| L | `기타` | 30 | 16 | 9 | **0.562** |
| D | `비조치` | 126 | 79 | 19 | 0.241 |
| D | **`조치`** | 14 | **6** | 3 | **0.500** |
| D | `기타` | 30 | 18 | 10 | 0.556 |
| H | `조치` | 14 | **0** | 0 | — |

잠재 검색(D)은 `조치` 선례를 여섯 배 많이 찾는다(1건 → 6건). **그런데 그중
절반이 함정이다.** 없던 근거가 생긴 것이 아니라 **"근거 없음" 이 "틀린 근거"
로 바뀌었을 뿐이고, 그쪽이 더 나쁘다** — 없으면 기권하지만 틀린 것이 있으면
따라간다.

`비조치` 에서도 D 의 함정 비율은 L 의 세 배다(0.241 vs 0.086).

### 새로 드러난 것 — `기타` 가 진짜 함정 지대다

세 검색기 모두 `기타` 의 함정 비율이 0.54~0.56 이다. **표면은 `비조치` 사례와
닮았는데 결론이 갈린다.** E5 는 `조치` 의 앵커링 부재를 보였지만 `기타` 의
이 성질은 드러내지 못했다.

보정표의 선례 라벨별 위험이 같은 것을 가리킨다 — 최근접 선례가 `기타` 를
가리킬 때 dev 오류율 0.611 [0.386, 0.797], `비조치` 는 0.214 [0.127, 0.338].

**그런데 Router 에 이 신호를 넣지 않았다.** 유사도와 겹친 2차원 표가 너무
성기고(AG-10), 라벨만으로 보정하면 유사도가 높은 경우까지 눌러 버린다.
`비조치` 의 주변 위험 0.214 는 `RISK_CEILING` 0.20 을 넘으므로, 그대로 쓰면
**F3(강한 선례 다수)조차 A 로 못 간다.** 쓸 수 있는 신호를 안 쓰는 것이 아니라
**지금 표본으로는 올바로 쓸 방법이 없다** — 그 사실을 적어 둔다.

### 그래서 Router 다

검색기를 바꿔서 함정을 줄일 수 없다는 것이 이 표의 결론이다. 셋 중 가장 나은
것은 이미 쓰고 있던 L 이다. 남은 길은 **찾은 것을 언제 버릴지 판정하는 것**이고,
그것이 Router 다.

재현: `python3 scripts/compare_retrievers.py` → `experiments/results/e8_retrievers.json`

---

## A. 기존 코드 중 재사용할 것

| 파일 | 무엇을 |
|---|---|
| `app/evaluation/confusable.py` | `idf_table` `weighted_vector` `cosine` `nearest` `partition` `accuracy` — Retriever(L) 와 TRAP 채점 |
| `app/retrieval/neighbor.py` | `band` `predict` — Naive RAG 기준선(E8) |
| `app/rules/induction.py` + `e6_rules.json` | 규칙 매칭 → PATH B 근거 |
| `app/evaluation/selective.py` | `operating_points` `aurc` — 기권 평가 |
| `app/evaluation/comparison.py` | `aligned_pairs` + 짝지은 부트스트랩 + Holm |
| `app/evaluation/metrics.py` | `macro_f1` `wilson_interval` `verdict_against` |
| `app/evaluation/validity.py` | `Claim` — 모든 수치의 무효 기준 게이트 |
| `app/core/audit.py` | `Discards` — 폐기된 근거 기록 |
| `app/agents/criteria.py` | `quote_is_grounded` — Validator V3 |
| `app/api/*` | 엔드포인트 추가만. 기존 계약 유지 |
| `app/evaluation/failure_*` | 새 실패도 같은 레지스트리로 |

**evaluation 계층 3,337줄은 손대지 않는다.** 추가만 한다.

## B. 새로 구현할 최소 컴포넌트

| 파일 | 줄 수(예상) | 무엇 |
|---|---|---|
| `app/retrieval/lexical.py` | ~60 | L — 기존 코사인을 Retriever 인터페이스로 감쌈 |
| `app/retrieval/dense.py` | ~120 | D — 4-gram 행렬의 절단 SVD(numpy), 질의 투영 |
| `app/retrieval/hybrid.py` | ~50 | H — RRF 융합 |
| `app/agents/state.py` | ~90 | `Evidence` `RouterSignals` `AgentState` `TraceStep` |
| `app/agents/router.py` | ~150 | 신호 계산 + 결정 표 R1~R10 + `trap_risk` 보정 |
| `app/agents/workflow.py` | ~140 | 노드 연결 · trace 기록 · 기권 게이트 |
| `app/agents/validator.py` | ~130 | V1~V6 (+V3+ 는 선택 경로) |
| `scripts/agent.py` | ~80 | CLI — `calibrate` `run` `experiment` |
| `tests/unit/test_router.py` 외 | ~400 | F1~F9 fixture + 극단 검사 |

합계 약 1,200줄. **evaluation 계층보다 작다** — 그게 맞는 비율이다.

## C. 새로 필요한 데이터

- 합성 fixture 8종 (F1~F8) — 손으로, 저장소 안에
- **그 외 없음.** 새 코퍼스·새 라벨링 없음

## D. 새로 필요한 외부 API 호출

- **E8·E9·E10·E11a: 0회.**
- E11b(선택): dry-run → 5건 실측 → 본 실행. 추정 $2.4, 실측 후 재계산.
- 새 의존성: `numpy` 만 (설치 확인함, 2.4.6)

## E. Phase 3 완료 판단 기준

§12 의 7개를 **검사 가능한 형태로** 옮긴다.

| | 요구 | 어떻게 증명하는가 |
|---|---|---|
| 1 | 검색 결과를 무조건 믿지 않는다 | 함정 15건 중 **B/C 로 라우팅된 비율 > 0**, F5 fixture 통과 |
| 2 | 선례/규칙 경로를 구분한다 | test 170건의 route 분포에 A·B 둘 다 존재, `route_reason` 이 R번호로 남음 |
| 3 | 근거 부족 시 기권한다 | F1·F4·F8 fixture 통과, `abstention_reason` 이 코드로 남음 |
| 4 | 잘못된 선례를 탐지한다 | **주 종점** — TRAP 정확도 Wilson 하한 > 0 (4/15 이상) |
| 5 | evidence 추적 가능 | 모든 결정에 `evidence_used[]` 비어 있지 않음, V2 100% 통과 |
| 6 | 기존 평가 체계와 연결 | E8~E11 이 `comparison.py` 로 채점되고 `validity.Claim` 을 통과 |
| 7 | 실제 업무 관점의 개선 | 같은 커버리지에서 위험이 Naive RAG 보다 낮다 (AURC, Holm 보정 후) |

**4번이 안 되면 Phase 3 은 미완이다.** 나머지가 다 돼도 그렇다.

---

## 설계의 문제점 — 지금 보이는 것

숨기지 않고 먼저 적는다. 이 중 몇은 실제로 터질 것이다.

### 1. `trap_risk` 의 전제가 틀릴 수 있다 — **가장 큰 위험**

E5 는 `P(선례 있음 | 정답=조치) = 7.1%` 를 보였다. Router 가 쓰는 건 그걸
뒤집은 `P(틀림 | 유사도, 선례의 라벨)` 이다. **두 조건부는 같은 것이 아니다.**
뒤집은 쪽이 실제로 함정을 가리키는지는 dev 에서 확인해야 하고, 확인 못 하면
Router 의 R5·R8 이 통째로 근거를 잃는다.

- **먼저 할 일**: 코드 쓰기 전에 dev LOO 로 이 표를 만들어 본다. 비용 0.
- **표가 평평하면**(유사도·라벨과 무관하게 오류율이 비슷하면) 설계를 바꾼다.
- 지금 이걸 "확인됨" 으로 적지 않는다. **미확인이다.**

### 2. dev 85건으로 문턱 5개를 보정한다 — 과적합

`τ_high` `τ_low` `α` `δ` `Y` 다섯 개를 85건에서 정한다. 칸마다 몇 건씩이다.
격자 탐색을 돌리면 dev 에서는 반드시 좋아 보이고 test 에서 무너질 수 있다.

- **완화**: 문턱을 자유 변수로 두지 않고 **거친 격자**(3~4개 값)로만 탐색하고,
  선택 근거를 dev LOO 성능이 아니라 **분포의 모양**(구간별 오류율 표)에서 읽는다.
- 그래도 남는 위험이므로, E8~E11 결과에 "문턱은 dev 에서 정했다" 를 명시한다.

### 3. PATH B 가 PATH A 만큼 눈이 멀었을 수 있다

E6 실측: 규칙 전이가 클래스 빈도와 **반대로** 간다.

```
비조치  dev 100% -> test 92.0%
기타    dev 100% -> test 53.3%
조치    dev 100% -> test 20.0%
```

`조치` 규칙이 test 에서 20% 로 무너진다. 그러면 R5(선례 버리고 B 로)가
**눈먼 곳에서 눈먼 곳으로 옮기는 것**이 된다.

- **예상 결과**: `조치` 는 대부분 C(기권)로 갈 것이다.
- 그건 실패가 아니라 **정직한 답**이다 — "이 도메인에서 조치는 근거가 없으면
  판단하면 안 된다". 다만 커버리지가 크게 떨어지므로 §9 의 성공 기준 7번을
  **위험-커버리지 곡선 전체**로 재게 설계했다. 한 점만 보면 억울해진다.

### 4. LSA 를 "dense retrieval" 이라 부르는 것에 대한 반론

면접관이 물을 수 있다 — "dense 라면서 왜 사전학습 인코더가 아닌가?"

- **답변 가능**: LSA 는 dense 표현이고, 이 코퍼스에서 다운로드 없이 재현된다.
- **약점 인정**: 사전학습 인코더가 더 강한 비교였을 것이다. 이 환경에서 검증을
  못 해 보류했다. 검증되면 네 번째 검색기로 추가하고, 안 되면 **왜 안 했는지**
  를 그대로 적는다.
- **더 정직한 지점**: L 도 D 도 결국 이 코퍼스의 표면 통계다. **둘 다 TRAP 을
  못 깰 가능성이 높다.** 그리고 그게 사실이면 그것이야말로 Router 가 필요한
  이유의 증명이 된다 — 검색을 바꿔서 풀 문제가 아니라는 것.

### 5. 문턱 기본값이 발표한 수치를 재현하지 않는다 — **실측으로 확인**

설계서를 쓰다 확인한 것이고, 이미 저장소에 있는 결함이다.

```
confusable.SIMILARITY_FLOOR = 0.25    ← CLI 의 --floor 기본값
trap.json  similarity_floor  = 0.15    ← 발표된 수치를 만든 값
neighbor.MEDIUM              = 0.15
```

`python3 scripts/confusable.py` 를 **인자 없이** 돌리면 발표된 표가 나오지
않는다. 직접 재봤다.

| floor | 순응 | 함정 | 선례 없음 |
|---|---|---|---|
| 0.15 (문서에 실린 값) | 72 | **15** | 83 |
| 0.25 (코드 기본값) | 57 | **10** | 103 |

함정 구간이 15건에서 10건으로 바뀐다. **Phase 3 의 주 종점이 이 구간 위에
있으므로**, 이걸 정리하지 않고 Router 를 얹으면 무엇을 개선했는지 말할 수 없다.

- IDF 출처도 인자로 갈린다 — 발표 수치는 `cases_nonaction.jsonl` 259건에서
  뽑은 IDF 다. dev+test 요청문으로 뽑으면 순응 70 · 함정 15 로 어긋난다.
  (기록된 출처로 재현하면 72/15/83 이 **정확히** 나온다 — 수치 자체는 무사하다.)
- **먼저 할 일**: 문턱과 IDF 출처를 한 곳에 못 박고, 발표 수치가 기본값으로
  재현되게 만든다. 실패 레지스트리에 올린다.

### 6. 다중비교 — 실험 4개 × 지표 7개 = 28개 비교

Holm 보정하면 거의 다 죽는다. 그래서 §9-0 에서 **주 종점 하나(TRAP 정확도)** 를
미리 못 박았다. 나머지는 기술 통계로만 보고하고 유의성을 주장하지 않는다.

### 7. 기권이 공짜로 이기는 문제

기권률을 올리면 Decision Accuracy 와 Abstention Accuracy 가 동시에 올라간다.
**단일 운영점 비교는 금지한다.** 전 구간 위험-커버리지 곡선과 AURC 로만 비교하고,
표에 커버리지를 항상 함께 적는다.

### 8. Phase 5 잔여물

`data/interim/answers_dev.jsonl` 26건과 `criteria.jsonl` 88개가 남아 있다.
Phase 3 은 이것을 **쓰지 않는다.** 지우지도 않는다 — 왜 중단했는지가
`docs/16` 에 있고, 그 기록이 이 프로젝트의 일부다.

---

## 예상 failure case — Phase 3 을 돌렸을 때

| | 무엇이 일어날 수 있나 | 그러면 |
|---|---|---|
| P1 | `trap_risk` 표가 평평하다 (§문제점 1) | Router R5·R8 재설계. 신호를 유사도 대신 **근거 다양성·라벨 일치도**로 옮긴다 |
| P2 | 함정 15건 중 Router 가 3건만 살린다 | Wilson [0.05, 0.45] — **하한이 0 을 넘지만 아슬아슬**. 그대로 적고 과장하지 않는다 |
| P3 | `조치` 14건이 전부 기권 | 커버리지 0.92, `조치` 재현율 0. **정직한 실패**로 기록하고, 그 자체를 결론으로 쓴다 |
| P4 | Router 가 Naive RAG 와 거의 같은 결정을 낸다 | 경로 분포를 본다. A 가 95% 면 결정 표가 사실상 안 발화한 것이다 |
| P5 | dense(D)가 lexical(L)보다 나쁘다 | 그대로 적는다. §문제점 4의 "둘 다 표면 통계" 가설이 지지된다 |
| P6 | Validator 가 아무것도 안 막는다 | V1~V6 각각의 발화 횟수를 세서 죽은 검사를 찾는다. 안 막는 검사는 검사가 아니다 |
| P7 | fixture 는 통과하는데 실제 test 에서 이상하다 | fixture 가 실제 분포를 대표하지 못한 것. **fixture 를 실제 사례에서 뽑는 쪽으로 바꾼다** |

---

## 다음에 할 일 — 순서 고정

1. ~~`trap_risk` 표를 먼저 만든다~~ **완료.** 전제는 살았고, 2차원 초안은
   버렸다(AG-10). `app/agents/calibration.py` · `scripts/calibrate.py`
2. ~~문턱 통일 (§문제점 5)~~ **완료.** `app/domain/similarity.py` 로 모았고,
   발표 수치가 기본값으로 재현된다(IN-13)
3. `AgentState` · `Evidence` 스키마
4. F1~F9 fixture (합성, 비용 0)
5. Retriever L/D/H
6. Router + 결정 표
7. Validator V1~V6
8. E8~E11a 실행 (비용 0)
9. (선택) E11b

**1번을 먼저 하는 이유**: 설계 전체가 그 표 위에 서 있는데 아직 안 봤다. 코드를
1,200줄 쓰고 나서 전제가 틀린 걸 알면, 이번에도 같은 실수를 반복하는 것이다.
