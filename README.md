# DeFactoRule

**규제 예외 승인 이력에서, 문서 어디에도 적혀 있지 않은 판단 기준을 복원한다.**

금융당국 사례집 1,095 사례로 "이 요청은 조치 대상인가" 를 판정하고,
**틀릴 것 같으면 기권하는** 시스템. 결론은 정확도가 아니라 *어디서 틀리는지를
아는가* 에 있다.

`1. Problem` · `2. Why` · `3. Different` · `4. Overview` · `5. Demo` ·
`6. Architecture` · `7. Evaluation` · `8. Failures` · `9. Experiments` · `10. Limitations`

---

## 1. Problem

어떤 조직이든 규정에는 **원칙**만 적혀 있고, 실제 판단은 **예외 승인 이력에
축적된 암묵 기준**을 따른다. "이런 조건이면 예외를 허용한다" 는 규칙이 문서
어디에도 없이 관행으로 존재한다. 담당자가 바뀌면 이 기준이 사라지고, 판단이
흔들린다.

금융규제에서 이 현상은 공개 데이터로 관측된다. 금융회사는 "이 행위를 해도
제재하지 않겠다" 는 확인을 당국에 요청하고, 당국은 **비조치의견서**로 답한다.
회신문에는 결론과 근거 법령이 적힌다. 그러나 **왜 이 사안은 되고 저 사안은
안 되는지를 가르는 기준은 어디에도 공표되지 않는다.**

증거는 사례집 안에 있다. 요청문이 거의 같은 사안들이 반복해서 나오는데,
그중 일부는 결론이 갈린다 — 표면에 적혀 있지 않은 무언가 때문에
([7. Evaluation](#7-evaluation) 의 함정 구간).

## 2. Why This Problem

- **사람이 처리하는 방식**: 경력자의 머릿속. 인수인계 문서에는 "케이스 바이
  케이스" 라고 적힌다.
- **검색으로 안 되는 이유**: 정답이 문서에 없다. 검색은 적혀 있는 것만 찾는다.
  이 저장소가 실제로 재봤다 — 가장 닮은 선례를 그대로 따라가는 전략(`neighbor`)은
  선례와 결론이 같은 구간에서 정확도 **1.000**, 결론이 갈리는 구간에서 **0.000**
  이다. 평균을 내면 그럴듯해 보이고, 갈리는 자리에서 전부 틀린다.
- **더 나쁜 것은 눈이 먼 자리가 하필 중요한 쪽이라는 점이다.** 닮은 선례가
  존재하는 비율을 정답 클래스별로 갈라 보면:

<!-- README_BLIND:시작 -->
| 정답 | test 건수 | dev 에 닮은 선례가 있는 건수 | 비율 |
|---|---|---|---|
| `조치` | 14 | 1 | **7.1%** |
| `기타` | 30 | 16 | **53.3%** |
| `비조치` | 126 | 70 | **55.6%** |
<!-- README_BLIND:끝 -->

  제재로 이어지는 `조치` 는 드물고, 드문 만큼 선례도 없다. 검색 기반 접근은
  **가장 비용이 큰 판단에서 구조적으로 무력하다.**
- 그러므로 필요한 것은 생성이 아니라 **귀납**, 그리고 귀납이 실패하는 자리를
  아는 것이다.

## 3. What Makes It Different

| 흔한 프로젝트 | DeFactoRule |
|---|---|
| 문서에 적힌 답을 찾는다 | 문서에 없는 판단 기준을 복원한다 |
| LLM 이 답을 생성한다 | LLM 은 의미 해석과 후보 생성만 한다. 수치·논리·판정은 결정론적 코드가 한다 |
| "LLM 이 그럴듯한 이유를 말했다" 로 끝 | held-out 예측과 **부트스트랩 + Holm 보정**으로 진위를 가른다 |
| 항상 답을 낸다 | **기권을 서비스 계약에 넣는다.** 모델은 신뢰도까지만 말하고 기권은 코드가 판정한다 |
| 잘 된 결과만 싣는다 | 기각된 가설과 뒤집힌 진단을 그대로 남긴다 |

**이 프로젝트가 실제로 내놓은 결론은 겸손한 쪽이다.** LLM 이 규칙 기준선보다
매크로 F1 이 높다는 주장은 7개 모델 1,122쌍 비교에서 다중비교 보정 후 대부분
살아남지 못한다(F1 7/21 · **AURC 10/21 유의**). 살아남는 것은 **위험–커버리지**
쪽이다 — 같은 모델이 자기 신뢰도로 기권할 때 AURC 0.125 대 규칙 기준선 0.282.
즉 이 도메인에서 LLM 의 값어치는 *더 맞히는 것* 이 아니라 **자기가 틀릴 때를
아는 것** 이다. 서비스가 기권을 못 돌려주면 그 값어치는 경계에서 사라진다.

## 4. System Overview

```
사례집 PDF ─▶ 파서 ─▶ 사례 1,095건 ─▶ 질의–회답 1,122쌍 ─▶ 라벨(문서 체크박스)
                │                                              │
                │ 전 과정 결정론적 · 서식 회귀 테스트가 지킨다      │ dev/test 결정론적 분할
                ▼                                              ▼
        ┌───────────────────────── 판정기 7종 ─────────────────────────┐
        │ majority · keyword · induced   결정론  (LLM 없이 어디까지)      │
        │ neighbor                       검색    (닮은 선례를 따른다)     │
        │ llm · prior · sector           LLM     (의미 해석 + 신뢰도)     │
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
             채점 하네스 — 매크로 F1 · 위험–커버리지(AURC) · TRAP
                                       ▼
             기권 판정(결정론) ─▶ FastAPI ─▶ 시각화 화면 4종
```

**§9 의 분리를 코드로 강제한다.** 무엇을 LLM 에게 맡기고 무엇을 맡기지 않는지가
계층 경계와 같다.

| LLM 이 하는 일 | 결정론적 코드가 하는 일 |
|---|---|
| 요청문의 의미 해석 | 라벨 판정 · 클래스 집계 |
| 회답에서 판단 기준 **후보** 생성 | 후보의 순환성·인용 대조 검증 |
| 신뢰도 등급 표명 | **기권 여부 결정** · 문턱 운용 |
| — | 통계 검정 · 다중비교 보정 · 분할 규율 |

`agents` 는 `evaluation` 을 임포트하지 않는다. LLM 계층이 채점 계층에 기대면
이 분리가 코드에서 무너지기 때문이다.

## 5. Demo

**Python 3.9 이상.** macOS 에서 `python` 은 시스템에 남은 2.7 을 가리키는 경우가
많으니 반드시 `python3` 를 쓴다.

```bash
git clone https://github.com/sokldjs554/DeFactoRule
cd DeFactoRule                      # 모든 명령은 저장소 루트에서
pip3 install -r requirements.txt
python3 scripts/check_env.py        # 무엇이 빠졌는지 알려준다
python3 scripts/serve.py            # http://127.0.0.1:8000/  ·  API 문서 /docs
```

`/` 는 **대화창이 아니다.** 이 프로젝트의 핵심 질문 — *모델이 자기가 틀릴 때를
아는가* — 를 네 화면으로 보여준다. 빌드 단계도 외부 자산도 없다.

| 화면 | 무엇을 보여주는가 |
|---|---|
| 위험–커버리지 곡선 | 기권을 허용하면 위험이 어떻게 떨어지는가. 모델별 AURC |
| 판정 + 문턱 슬라이더 | 같은 요청에 문턱을 올려 보며 **기권이 생기는 지점**을 직접 본다 |
| 업권별 기저율 | 어디가 어려운 구간인가. dev 에서 뽑은 값만 노출한다 |
| 실패 레지스트리 | 58건의 **지금** 상태 — 재현 검사를 그 자리에서 돌린 결과 |

| 엔드포인트 | 무엇을 주는가 |
|---|---|
| `POST /classify` | 결론 예측. `min_confidence` 를 올리면 **기권**한다 |
| `GET /base-rates` | dev 기저율(업권별). test 에서 뽑은 값은 노출하지 않는다 |
| `GET /evaluation/models` | 커버리지 100% 에서의 매크로 F1. 결측이 있으면 표시한다 |
| `GET /evaluation/risk-coverage` | 위험–커버리지 곡선과 AURC. 결측이 있는 예측은 제외 |
| `GET /failures` | 실패 레지스트리 + 재현 검사 결과 |

전체 파이프라인을 원본 PDF 부터 다시 돌리는 명령은 [부록](#부록--재현)에 있다.

## 6. Architecture

```
app/
  core/            공용 입출력과 레코드 키. 도메인을 모른다.
  domain/          라벨 체계 · 판정 지침 · 기저율
  extraction/      PDF → 사례 → 질의·회답 쌍 (전 과정 결정론적)
  rules/           결정론적 기준선 — LLM 없이 어디까지 되는가
  retrieval/       근거 검색 — 아직 비어 있다
  agents/          LLM 호출. 의미 해석과 후보 생성만 맡는다
  evaluation/      채점 · 통계 · 오류 분석
  infrastructure/  외부 시스템과의 경계
  api/             서비스 진입점

scripts/           얇은 CLI 진입점 (이름은 그대로, 구현은 app/ 에 있다)
tests/             unit · integration · evaluation · regression
```

의존 방향은 위에서 아래로만 흐른다. 자세한 근거와 재편 과정은
[docs/11-architecture.md](docs/11-architecture.md).

테스트 <!--TESTS-->397<!--/TESTS-->개가 네 갈래로 나뉘어 있다. `regression` 은
고친 실패가 다시 열리는 것을, `evaluation` 은 문서 수치가 산출물과 어긋나는 것을
막는다.

```bash
pip3 install -r requirements-dev.txt
python3 -m pytest tests -q
```

## 7. Evaluation

**분할 규율.** dev/test 는 난수가 아니라 결정론적 규칙으로 나눈다(3건마다 1건).
라벨은 사람이 붙인 것이 아니라 **문서에 인쇄된 체크박스**다. 정답 표지가 요청문에
새어 들어가는 경로를 세 겹으로 지웠고, 클래스별 용어 감사(이항 검정 + Holm)로
남은 누출을 다시 확인했다.

**지표.**

| 지표 | 왜 이것을 보는가 |
|---|---|
| 매크로 F1 | 클래스가 심하게 치우쳐 정확도는 다수 클래스만 맞혀도 올라간다 |
| AURC (위험–커버리지) | 기권을 허용했을 때의 성능. 이 프로젝트의 실제 주장이 놓인 자리 |
| **TRAP** ★ | 이 프로젝트를 위해 만든 지표 — 아래 |
| 짝지은 부트스트랩 + Holm | 7개 모델 1,122쌍을 한 번에 비교하므로 보정 없이는 우연이 섞인다 |

<!-- README_F1:시작 -->
| 모델 | 매크로 F1 (커버리지 100%) |
|---|---|
| `sector` | 0.636 |
| `llm` | 0.587 |
| `neighbor` | 0.538 |
| `prior` | 0.504 |
| `keyword` | 0.494 |
| `induced` | 0.434 |
| `majority` | 0.284 |
<!-- README_F1:끝 -->

**TRAP — 표면선례 함정 정확도.** 매크로 F1 도 AURC 도 기성품이고, 둘 다
"문서에 적혀 있지 않은 기준을 복원했는가" 에는 답하지 못한다. 그래서 test 사례마다
dev 에서 가장 닮은 선례를 찾아 두 무리로 가른다 — 선례의 결론이 정답과 같으면
**순응**, 다르면 **함정**. 함정 구간의 정확도가 TRAP 이다. 표면 유사도를 그대로
베끼는 전략은 여기서 **구조적으로 0%** 이므로, TRAP 은 *표면 너머를 읽었는가* 의
직접 측정이 된다. 최근접 선례는 dev 에서만 찾는다(test 끼리 찾으면 정답이 오간다).

<!-- README_TRAP:시작 -->
순응 72건 · 함정 15건 · 선례 없음 83건 (닮음 문턱 0.15, 문자 4-gram IDF 코사인)

| 모델 | 전체 | 순응 72건 | 함정 15건 (TRAP) | 격차 |
|---|---|---|---|---|
| `sector` | 0.800 | 0.958 | **0.467** | 0.492 |
| `majority` | 0.741 | 0.889 | **0.400** | 0.489 |
| `llm` | 0.769 | 0.915 | **0.357** | 0.558 |
| `prior` | 0.763 | 0.915 | **0.357** | 0.558 |
| `keyword` | 0.747 | 0.903 | **0.133** | 0.769 |
| `neighbor` | 0.724 | 1.000 | **0.000** | 1.000 |
<!-- README_TRAP:끝 -->

`neighbor` 가 순응 1.000 · 함정 0.000 인 것은 지표가 설계대로 작동한다는 뜻이다.
전체 정확도만 보면 `neighbor` 는 `majority` 와 비슷해 보이지만, **결론이 갈리는
자리에서는 하나도 맞히지 못한다.**

이 표의 모든 수치는 `experiments/results/*.json` 에서 생성한다. 손으로 적지
않는다 — 한 번 손으로 적었다가 판정이 뒤집힌 문장을 문서에 남긴 적이 있다(EV-14).

```bash
python3 scripts/sync_docs.py --check   # 어긋난 곳만 본다
python3 scripts/sync_docs.py           # 산출물에서 다시 써 넣는다
```

## 8. Failure Cases

실패 케이스 67건을 계층·범주로 분류하고, **각 건마다 재현 검사(probe)를 코드로**
남겼다. 고쳤다고 기록된 케이스가 실패하면 그 수정이 풀린 것이고, 열려 있다고
기록된 케이스가 통과하면 레지스트리가 낡은 것이다. 둘 다 테스트가 잡는다.

```bash
python3 scripts/failure_report.py              # 전체 재현 검사
python3 scripts/failure_report.py --layer extraction
```

기록에 남길 값어치가 있었던 것 몇 가지:

| ID | 무엇이 잘못됐나 | 어떻게 드러났나 | 전 → 후 |
|---|---|---|---|
| EX-05 | 항목명의 뒷조각이 값 맨 앞에 남았다 | 레지스트리 probe 가 코퍼스를 훑어 | 406건 → 0건 |
| EX-16 | 순번 교집합으로 짝을 지어 서로 다른 질의·회답을 붙였다 | 분할 함수에 단위 테스트를 붙이다 | 2건 → 0건 |
| EV-14 | 학습기를 고친 뒤 문서만 갱신을 놓쳐 **판정이 뒤집힌 문장**이 남았다 | 7개 모델 전수 재계산 | 1건 → 0건 |
| EV-16 | 규칙 학습기가 후보를 버리며 아무 데도 기록하지 않았다 | 공용 폐기 장치를 만들며 | 0개 → 165개 기록 |

**개별 사례가 아니라 패턴을 지킨다.** 같은 종류의 실수가 2건 이상 쌓이면
그 *패턴* 자체에 가드를 붙인다 — 예를 들어 "걸러내는 코드가 걸러낸 것을 기록하지
않는다" 는 이제 모든 필터 단계를 훑어 검사한다. 전/후 수치와 taxonomy 는
[docs/12-failure-registry.md](docs/12-failure-registry.md).

## 9. Experiments

| 실험 | 물음 | 결과 |
|---|---|---|
| [E1](docs/07-e1-llm-vs-baseline.md) | LLM 이 규칙 기준선보다 나은가 | 매크로 F1 0.587 vs 0.494 — **판정 보류** |
| [E2](docs/08-e2-risk-coverage.md) | 기권을 허용하면 달라지는가 | AURC 0.125 vs 0.282 — **유의** |
| [E3](docs/09-e3-sector-analysis.md) | 어느 구간이 어려운가 | E1 진단 정정 · 전자금융이 어려운 구간 |
| [E4](docs/10-e4-prompt-variants.md) | 기저율을 알려주면 나아지는가 | **가설 기각** — AURC 불변 |
| [E5](docs/13-retrieval-baseline.md) | 검색만으로 어디까지 되는가 | F1 0.538 · **`조치` 앵커링 7.1%** |
| [E6](docs/14-rule-induction.md) | 규칙을 역추출하면 전이되는가 | **`조치` 규칙 전이 100%→20%** |
| [E7](docs/15-full-comparison.md) | 7개 모델 전수 비교, 보정 후 | F1 7/21 · **AURC 10/21 유의** |

기각된 가설(E4)과 뒤집힌 진단(E3)을 지우지 않고 남겼다. 실행하지 않은 수치는
이 저장소 어디에도 쓰지 않는다.

전체 문서: [주제 선정](docs/01-topic-research.md) · [W1 게이트](docs/02-w1-gate.md) ·
[분할과 회귀](docs/03-splitting-and-regression.md) · [띄어쓰기 복원](docs/04-spacing-restoration.md) ·
[Phase 2 baseline](docs/05-phase2-baseline.md) · [비조치 baseline](docs/06-nonaction-baseline.md) ·
[아키텍처](docs/11-architecture.md) · [실패 레지스트리](docs/12-failure-registry.md) ·
[회답 근거 구조화](docs/16-criteria-extraction.md)

## 10. Limitations

- **소수 클래스 표본이 작다.** test 의 `조치` 는 14건이다. 이 프로젝트의 소수
  클래스 수치는 모두 그 위에 있고, 신뢰구간이 넓다. 확대 해석하지 않는다.
- **회답 근거 구조화가 아직 실행되지 않았다.** 파이프라인·순환 차단·인용 대조는
  구현과 테스트를 마쳤으나 실제 추출은 미실행이다(Phase 5b). 설계와 사전 등록한
  성공 기준은 [docs/16](docs/16-criteria-extraction.md).
- **`retrieval/` 은 비어 있다.** E5 의 검색 기준선은 `evaluation` 안에서 돌고,
  독립된 색인 계층(Elasticsearch)은 아직 없다.
- **Track B(혁신금융서비스 부가조건)는 미착수.** 게이트에서 반려·미지정 사례가
  공개되지 않는다는 제약을 확인했고, 타겟을 "어떤 부가조건이 어느 수준으로
  붙는가" 로 재정의해 둔 상태다. → [docs/01](docs/01-topic-research.md)
- **법령해석 트랙(836건)에는 아직 정답이 없다.** 비조치 트랙만 문서 체크박스라는
  순환 없는 라벨을 갖는다.
- **미적용 스택**: LangGraph 워크플로 · Docker 패키징 · PostgreSQL. 필요가
  생기지 않은 것을 쓰기 위해 설계를 비틀지 않았다.

---

## 부록 — 진행 상태

| Phase | 내용 | 상태 |
|---|---|---|
| **−1** | 주제 발굴 35개 → 채점 → 최종 1개 선정 | ✅ 완료 |
| **−1b** | 데이터 생존 게이트 | ✅ 조건부 통과 (설계 변경) |
| **1a** | 사례집 파서 · 코퍼스 실측 | ✅ **1,095건 — W1 게이트 통과** |
| **1b** | 질의–회답 분할 · 서식 회귀 테스트 | ✅ 1,122쌍 |
| **1c** | 띄어쓰기 복원 | ✅ 45건 복원 · 비조치 F1 0.816 |
| **2a** | 라벨 체계 · 규칙 baseline · 평가 하네스 | ✅ 법령해석 커버리지 41.1% |
| **2b** | LLM 분류기 | ✅ 비조치 170/170 |
| **2c** | 비조치 트랙 baseline (순환 없는 평가) | ✅ 매크로 F1 0.494 |
| **2d** | E1 — LLM vs 규칙 baseline | ✅ 매크로 F1 0.587 (판정 보류) |
| **2e** | E2 — 위험-커버리지 곡선 | ✅ AURC 0.125 vs 0.282 (**유의**) |
| **2f** | E3 — 업권별 분석 | ✅ E1 진단 정정 |
| **2g** | E4 — 기저율 프롬프트 변형 | ✅ 가설 기각 (AURC 불변) |
| **2h** | E5 — 검색 기준선 · 표면선례 함정 지표 | ✅ 검색 F1 0.538 · **조치 앵커링 7.1%** |
| **3a** | 아키텍처 재편 · 테스트 4분할 | ✅ app/ 9계층 |
| **3b** | 실패 케이스 레지스트리 | ✅ 실행 가능한 재현 검사 |
| **3c** | API 계층 (FastAPI · Pydantic) | ✅ 기권을 계약에 포함 |
| **3d** | 시각화 UI (채팅창 아님) | ✅ 화면 4종 |
| **4a** | E6 — 규칙 역추출 (결정론적 학습기) | ✅ **조치 규칙 전이 100%→20%** |
| **4b** | E7 — 7개 모델 전수 비교 (Holm 보정) | ✅ F1 7/21 · **AURC 10/21 유의** |
| **5a** | 회답 근거 구조화 — 파이프라인·안전장치 | ✅ 순환 차단 · 인용 대조 · dry-run |
| **5b** | 회답 근거 구조화 — 실행 | ⏸ API 필요 (~$5.5, 단계별 중단 가능) |
| 6~11 | Agent 워크플로 · 배포 | ⬜ 대기 |

**현재 규모** — 실패 케이스 67건 · 테스트 331개 · 문서 16편.
이 숫자들은 `tests/regression/test_documented_numbers.py` 가 매번 대조한다.

## 부록 — 재현

원본 PDF 는 저장소에 커밋하지 않는다. [data/SOURCES.md](data/SOURCES.md) 를 보고
`data/raw/casebooks/` 에 넣은 뒤:

```bash
# 1. 사례집 PDF → 사례
python3 scripts/parse_casebook.py --input data/raw/casebooks --output data/processed

# 2. 띄어쓰기 복원
python3 scripts/restore_spacing.py train --input data/processed --model models/spacing.json
python3 scripts/restore_spacing.py apply --input data/processed --model models/spacing.json --threshold -0.25

# 3. 질의–회답 쌍으로 분할
python3 scripts/split_queries.py --input data/processed --output data/processed

# 4. 평가셋 (정답은 문서 체크박스)
python3 scripts/make_nonaction_gold.py --input data/processed/cases_nonaction.jsonl --output data/eval

# 5. 결정론적·검색 기준선
python3 scripts/baseline_nonaction.py --gold data/eval/nonaction_test.jsonl \
    --output data/processed/pred_nonaction_majority.jsonl --strategy majority
python3 scripts/baseline_neighbor.py --dev data/eval/nonaction_dev.jsonl \
    --gold data/eval/nonaction_test.jsonl \
    --output data/processed/pred_nonaction_neighbor.jsonl

# 6. LLM 판정기 (API 키 필요 · --limit 로 먼저 비용을 확인한다)
export ANTHROPIC_API_KEY=...
python3 scripts/classify_llm.py --task nonaction \
    --input data/eval/nonaction_test.jsonl \
    --output data/processed/pred_nonaction_llm.jsonl --limit 30

# 7. 채점
python3 scripts/evaluate.py --gold data/eval/nonaction_test.jsonl \
    --pred data/processed/pred_nonaction_majority.jsonl --labels nonaction --name majority
```

## 부록 — 개발 환경 주의사항

수집 대상 도메인은 개발 컨테이너에서 접근이 차단되어 있다.

```
better.fsc.go.kr    EGRESS_BLOCKED   (금융규제·법령해석포털)
www.data.go.kr      EGRESS_BLOCKED   (공공데이터포털)
```

**수집 스크립트는 로컬에서 실행한다.** 수집 결과를 커밋한 뒤 전처리·모델링·평가를
진행한다.
