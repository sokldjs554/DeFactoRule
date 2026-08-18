# E7 — 7개 모델 전수 비교, 다중비교 보정 후

앞선 실험은 두세 모델씩 짝지어 비교했다. 이제 예측이 전부 170/170 로 채워졌으므로
한 표에 놓는다. 그러자 곧바로 방법론 문제가 생긴다.

## 21쌍을 보정 없이 검정하면 안 된다

모델 7개는 비교 쌍 21개다. 유의수준 5%에서 21번 검정하면 **아무 차이가 없어도**
66% 확률로 하나쯤 유의하게 나온다(1 − 0.95²¹). 보정 없이 "유의" 라고 적으면
그중 무엇이 진짜인지 말할 수 없다.

그래서 매크로 F1 과 AURC 양쪽에 **Holm-Bonferroni** 보정을 붙였다. Holm 은
Bonferroni 보다 검정력을 덜 잃으면서 family-wise 오류율을 같게 지킨다. 보정 전
p값도 나란히 남긴다 — 숨기면 나중에 왜 이 결과가 사라졌는지 되짚을 수 없다.

## 모델 7개

| 모델 | 무엇인가 | 비용 |
|---|---|---|
| `majority` | 무조건 다수 클래스 | 0 |
| `keyword` | 사람이 dev 를 읽고 만든 어휘 규칙 | 0 |
| `induced` | 기계가 dev 에서 배운 규칙 (E6) | 0 |
| `neighbor` | 최근접 선례 검색 (E5) | 0 |
| `llm` | LLM, 문맥 없음 | $1.7 |
| `prior` | LLM + 전체 기저율 (E4) | $1.5 |
| `sector` | LLM + 업권별 기저율 (E4) | $1.7 |

## 매크로 F1 — 거의 아무것도 말할 수 없다

| 모델 | 매크로 F1 |
|---|---|
| sector | 0.636 |
| llm | 0.587 |
| neighbor | 0.538 |
| prior | 0.504 |
| keyword | 0.494 |
| induced | 0.436 |
| majority | 0.284 |

대응표본 부트스트랩 5,000회. **21쌍 중 보정 후 유의는 7쌍뿐이고, 그중 6쌍이
`vs majority` 다.**

| 비교 | 차이 | p | p(Holm) | 판정 |
|---|---|---|---|---|
| sector − majority | +0.352 | 0.000 | 0.000 | 유의 |
| llm − majority | +0.303 | 0.000 | 0.000 | 유의 |
| neighbor − majority | +0.255 | 0.000 | 0.000 | 유의 |
| prior − majority | +0.220 | 0.000 | 0.000 | 유의 |
| keyword − majority | +0.210 | 0.000 | 0.000 | 유의 |
| induced − majority | +0.152 | 0.000 | 0.000 | 유의 |
| sector − induced | +0.200 | 0.001 | 0.018 | 유의 |
| sector − prior | +0.132 | 0.030 | 0.420 | **보정 후 탈락** |
| llm − induced | +0.151 | 0.046 | 0.593 | **보정 후 탈락** |
| sector − neighbor | +0.098 | 0.134 | 1.000 | 판정 보류 |
| sector − llm | +0.049 | 0.388 | 1.000 | 판정 보류 |
| … 나머지 10쌍 | | | 1.000 | 판정 보류 |

보정을 안 했다면 9쌍을 "유의" 라고 적었을 것이다. 두 쌍이 탈락했다.

**매크로 F1 만 보면 이 프로젝트의 결론은 "다수 기준선보다는 낫다" 하나뿐이다.**
$5 를 쓴 LLM 이 공짜 검색보다 낫다고 말할 수 없다(p=0.134).

## AURC — 여기서 갈린다

기권을 허용하면 이야기가 달라진다. 대응표본 부트스트랩 2,000회.

| 모델 | AURC | 최저 커버리지에서의 위험 | 운영점 |
|---|---|---|---|
| prior | **0.123** | 24.7% 에서 4.8% | 3 |
| sector | 0.124 | 33.5% 에서 7.0% | 3 |
| llm | 0.125 | 14.1% 에서 4.2% | 3 |
| induced | 0.202 | 62.9% 에서 16.8% | 2 |
| neighbor | 0.207 | 30.0% 에서 11.8% | 3 |
| majority | 0.259 | — (한 점) | 1 |
| keyword | 0.282 | 11.8% 에서 50.0% | 2 |

**21쌍 중 보정 후 유의 12쌍.** 매크로 F1 의 7쌍과 대비된다.

| 비교 | 차이 | p(Holm) | 판정 |
|---|---|---|---|
| sector − keyword | −0.158 | 0.000 | 유의 |
| llm − keyword | −0.157 | 0.000 | 유의 |
| prior − keyword | −0.159 | 0.000 | 유의 |
| sector − majority | −0.135 | 0.000 | 유의 |
| llm − majority | −0.134 | 0.000 | 유의 |
| prior − majority | −0.136 | 0.000 | 유의 |
| **sector − neighbor** | **−0.083** | **0.028** | **유의** |
| **llm − neighbor** | **−0.082** | **0.050** | **유의** |
| **neighbor − prior** | **+0.084** | **0.044** | **유의** |
| sector − induced | −0.078 | 0.015 | 유의 |
| llm − induced | −0.077 | 0.028 | 유의 |
| prior − induced | −0.079 | 0.028 | 유의 |
| keyword − induced | +0.080 | 0.225 | 보정 후 탈락 |
| neighbor − keyword | −0.075 | 0.264 | 보정 후 탈락 |
| **sector − llm** | −0.001 | 1.000 | 판정 보류 |
| **sector − prior** | +0.001 | 1.000 | 판정 보류 |
| **llm − prior** | +0.002 | 1.000 | 판정 보류 |

## 세 문장으로 요약하면

**1. 매크로 F1 으로는 LLM 이 공짜 기준선보다 낫다고 말할 수 없다.**
sector − neighbor = +0.098, p=0.134. 보정하면 p=1.000.

**2. 기권을 허용하면 LLM 계열 셋이 검색·규칙·다수 기준선을 전부 유의하게 이긴다.**
sector−neighbor, llm−neighbor, sector−induced, llm−induced 모두 보정 후에도 살아남는다.
LLM 이 사는 곳은 정확도가 아니라 **자기가 틀릴 때를 아는 능력**이다.

**3. 기저율을 프롬프트에 넣는 것은 AURC 를 전혀 바꾸지 못한다.**
sector−llm = −0.001 (p=0.919), llm−prior = +0.002 (p=0.888), sector−prior = +0.001 (p=0.978).
세 변형의 AURC 는 0.123 / 0.124 / 0.125 로 사실상 같은 값이다.

3번은 E4 에서 세운 가설이 **전수 데이터에서 최종 기각**됐다는 뜻이다. E4 당시
`prior` 는 매크로 F1 을 0.587 → 0.504 로 떨어뜨리면서(조치 재현율 0.286 → 0.071)
AURC 는 건드리지 않았는데, 그 관찰이 170건 전수에서 그대로 확인됐다. 기저율은
**어느 답을 고를지는 바꾸지만 어느 답을 믿을지는 바꾸지 않는다.**

## 이 표가 남기는 숙제

`induced` 는 AURC 0.202 로 `neighbor` 0.207 과 구분되지 않으면서(p=0.853)
매크로 F1 은 0.436 대 0.538 로 뒤진다. 규칙 학습기의 신뢰도 신호(dev 정밀도
≥0.9면 high)가 우연히 쓸 만했을 뿐, 규칙 자체가 좋아서가 아니다 — E6 에서
소수 클래스 규칙이 test 에서 20% 로 무너지는 것을 이미 봤다.

그리고 무엇을 해도 `조치` 는 잡히지 않는다. 요청문 표면에는 그 신호가 없다.
다음 실험은 요청문이 아니라 **회답 본문의 판단 근거**를 봐야 한다.

## 재현

```bash
python3 scripts/compare_models.py --gold data/eval/nonaction_test.jsonl --labels nonaction \
    --pred sector=... --pred llm=... --pred neighbor=... --pred prior=... \
    --pred keyword=... --pred induced=... --pred majority=... \
    --report experiments/results/e7_all_models.json

python3 scripts/risk_coverage.py --gold data/eval/nonaction_test.jsonl --labels nonaction \
    --pred ... --report experiments/results/e7_risk_coverage.json
```
