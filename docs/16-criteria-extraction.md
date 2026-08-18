# Phase 5 — 회답 근거 구조화 (설계와 안전장치)

> 이 문서는 **파이프라인과 그 안전장치**를 적는다. 실험 결과는 아직 없다.
> API 호출이 필요한 단계는 실행 전이며, 실행하지 않은 수치는 적지 않는다.

## 왜 이제 회답을 보는가

세 갈래가 같은 곳을 가리켰다. `조치` 여부를 가르는 신호가 **요청문 표면에 없다.**

| 실험 | 증거 |
|---|---|
| E5 검색 | `조치` 14건 중 닮은 선례가 있는 것 **1건(7.1%)**, 최대 유사도 0.151 |
| E6 규칙 | `조치` 규칙 dev 100% → test **20%** |
| E1/E3 LLM | `조치` 재현율 **0.286** |

요청문에 없는 것을 요청문에서 찾은 것이 여기까지의 한계였다. 그런데 **회답에는
당국이 왜 그렇게 판단했는지가 적혀 있다.** 판단이유는 255건 중 254건이 채워져
있고 평균 569자다. 특히 `조치` 사례의 판단이유가 841자로 가장 길다(비조치 479자).

## 순환을 코드가 막는다

회답에는 결론도 함께 적혀 있다. 회답을 넣고 결론을 맞히면 100%가 나오고
아무것도 배우지 못한다. 요청문에서 이미 누출을 세 번 겪었고 **세 번 다 눈으로
찾았다.** 이번에는 규율을 코드로 강제한다.

| 규율 | 어떻게 |
|---|---|
| 기준은 dev 회답에서만 | 추출 단계가 dev 키로 걸러진 사례만 읽는다 |
| 적용은 test 요청문에만 | 적용 프롬프트에 회답이 들어가지 않는다 |
| 가중치는 dev 에서만 | `fit()` 이 dev 답과 dev 라벨만 본다 |
| 결론을 되묻는 질문은 폐기 | `question_is_circular()` 가 정규식으로 막는다 |
| 근거 없는 기준은 폐기 | 인용을 판단이유 원문과 글자 단위로 대조한다 |

마지막 두 개가 핵심이다. "이 사안은 조치 대상인가?" 는 기준이 아니라 결론을
되묻는 것이고, 사람이 눈으로 거를 일이 아니다.

## 명세 §9 의 분리를 끝까지

```
LLM          회답을 읽고 판단 기준을 뽑는다
             요청문이 그 기준에 해당하는지 예/아니오/모름으로 답한다

결정론 코드   순환 검사 · 인용 대조 · 기준 통합 · 가중치 산출
             최종 라벨 결정 · 신뢰도 부여 · 채점
```

**최종 라벨은 모델이 정하지 않는다.** 모델에게 라벨을 물으면 기준은 장식이
된다 — 답이 왜 그렇게 나왔는지 되짚을 수 없고, 기준 하나를 빼면 무엇이
달라지는지도 알 수 없다.

가중치는 기저율 대비 로그 승산이다.

```
w(c, label) = log( P(label | c=yes) / P(label) )
```

기저율로 나누므로 다수 클래스가 저절로 이기지 않는다. `unknown` 은 0 이다 —
모르는 것은 증거가 아니다.

## 실행하기 전에 잡은 것

`--dry-run` 으로 프롬프트를 눈으로 보다가 **판단이유 98.8%에 조판 잔재가 있다는
것**을 발견했다.

| 글자 | 이름 | 횟수 |
|---|---|---|
| U+2244 | NOT ASYMPTOTICALLY EQUAL TO | 360 |
| U+25A1 | WHITE SQUARE | 362 |
| U+00B7 | MIDDLE DOT | 303 |
| U+200C | ZERO WIDTH NON-JOINER | 266 |

U+2244 는 글머리 기호가 깨져 들어온 것이다. 이 상태로 돌렸다면 모델이 잔재를
빼고 옮겨 적은 **정상적인 인용이 전부 대조에 실패**해서, 기준이 하나도 채택되지
않은 채 $2.5 를 쓰고 "모델이 인용을 지어낸다" 는 잘못된 결론에 이르렀을 것이다.

읽기용과 대조용 위생을 나눴다(`app/core/text.py`). 읽기용은 보이지 않는 잡티만
걷어내고 글머리 기호는 남긴다 — 목록 구조가 사라지면 사람도 모델도 문단을
잘못 읽는다. 대조용은 글머리 기호까지 걷어내되 **글자는 건드리지 않는다.**
뜻이 바뀐 인용은 여전히 걸러져야 하기 때문이다.

## 단계와 비용

```bash
# 0. 지금 어디까지 왔는지 (파일만 읽는다, 비용 0)
python3 scripts/criteria.py status

# 1. dev 회답에서 기준을 뽑는다                      약 $2.5 · 83건
python3 scripts/criteria.py extract \
    --dev data/eval/nonaction_dev.jsonl \
    --cases data/processed/cases_nonaction.jsonl \
    --output data/interim/criteria_raw.jsonl --resume

# 2. 하나의 목록으로 합친다                          비용 0
python3 scripts/criteria.py consolidate \
    --input data/interim/criteria_raw.jsonl \
    --output data/eval/criteria.jsonl

# 3. dev 요청문에 적용해 가중치의 근거를 만든다        약 $1.0 · 85건
python3 scripts/criteria.py apply \
    --gold data/eval/nonaction_dev.jsonl \
    --criteria data/eval/criteria.jsonl \
    --output data/interim/answers_dev.jsonl --resume

# --- 여기서 멈추고 dev 성능을 먼저 본다 ---

# 4. test 요청문에 적용한다                          약 $2.0 · 170건
python3 scripts/criteria.py apply \
    --gold data/eval/nonaction_test.jsonl \
    --criteria data/eval/criteria.jsonl \
    --output data/interim/answers_test.jsonl --resume

# 5. 답을 라벨로 바꾼다 (결정론)                      비용 0
python3 scripts/criteria.py predict \
    --criteria data/eval/criteria.jsonl \
    --dev data/eval/nonaction_dev.jsonl \
    --dev-answers data/interim/answers_dev.jsonl \
    --test-answers data/interim/answers_test.jsonl \
    --output data/processed/pred_nonaction_criteria.jsonl

# 6. 다른 일곱 모델과 같은 하네스로 채점              비용 0
python3 scripts/evaluate.py --gold data/eval/nonaction_test.jsonl \
    --pred data/processed/pred_nonaction_criteria.jsonl --labels nonaction --name criteria
```

단계를 건너뛰면 무엇을 먼저 돌려야 하는지 알려준다. 순서가 헷갈리면
`status` 가 체크리스트로 보여준다.

모든 단계에 `--dry-run` 과 `--resume` 이 있다. `--dry-run` 은 요청을 하나도
보내지 않고 프롬프트와 추정 비용만 보여준다. 계정 수준 오류는 첫 호출 전
사전 점검에서 걸리고, 도중에 나면 즉시 중단하며 그때까지의 결과를 지킨다.

## 첫 실행에서 332 요청이 전부 죽었다

기록해 둔다. `extract` 83건이 재시도까지 332 요청이었고 **전부 400** 이었다.

    output_config.format.schema: For 'array' type,
    property 'maxItems' is not supported

사례당 기준 수를 4개로 묶으려고 스키마에 `maxItems` 를 적었다. JSON Schema
로서는 정당하고, 단위 테스트도 통과했다. 그러나 이 API 의 구조화 출력은 배열
길이 제약을 받지 않는다. **로컬에서 정당한 것과 API 가 받아 주는 것은 다르다.**

더 뼈아픈 것은 사전 점검이 이것을 잡지 못했다는 점이다. 사전 점검은 스키마 없이
한 글자를 보내 계정이 살아 있는지만 봤다. 계정은 살아 있었고 요청 계약은 죽어
있었다. **본 요청과 다른 계약으로 하는 사전 점검은 사전 점검이 아니다.**

고친 방식은 두 겹이다.

| | 무엇을 하는가 | 비용 |
|---|---|---|
| `app/infrastructure/schema_rules.py` | 호출 없이 금지 키워드를 찾는다. CI 에서 돈다 | 0 |
| `preflight(client, schema)` | **본 요청과 같은 스키마로** 한 번 호출한다 | 토큰 최소 |

금지 키워드 목록은 근거를 실측/유추로 구분해 적었다. 추측으로 목록을 불리면
멀쩡한 스키마를 막게 된다. 길이 상한은 명세 §9 대로 결정론적 코드로 옮겼고,
상한을 넘은 기준은 조용히 잘리지 않고 `사례당 상한 초과` 이유와 함께 남는다.

**과금은 없었다.** 400 은 청구되지 않는다. 실제 비용 출력도 `$0.000` 이었다.
잃은 것은 한 단계를 다시 돌려야 한다는 사실뿐이다. 레지스트리 IN-10.

그리고 이것이 세 번째로 같은 모양이었다 — 가드가 자기가 지킨다고 말한 것보다
좁게 검사한 경우. 그래서 개별 사례가 아니라 패턴(`guard-narrower-than-claim`)에
가드를 붙였다: 이 패턴에 속한 사례는 **반례 테스트를 지명해야 하고**, 그 테스트가
실제로 수집되고 통과하는지 확인한다.

## 무엇이 성공인가

미리 적어 둔다. 나중에 결과를 보고 기준을 옮기지 않기 위해서다.

1. **`조치` 재현율이 0.286 을 넘는가.** 이것이 이 실험의 존재 이유다.
2. **AURC 가 0.124(sector)보다 낮은가.** 그래야 회답 근거가 요청문만 읽는
   것보다 나은 신호를 준다고 말할 수 있다.
3. **채택된 기준이 사람이 읽어서 판단 기준으로 보이는가.** `'하는것이전자금융
   감독규정제1'` 같은 조각이 나오면 그것은 기준이 아니다.

셋 다 실패해도 그대로 적는다. E4 의 기저율 가설이 기각된 것을 그대로 남긴 것과
같다.
