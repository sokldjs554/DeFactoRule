# E2 — 위험-커버리지 곡선

## E1 이 남긴 논리적 구멍

E1 에서 "LLM 은 커버리지 72.9%에서 매크로 F1 0.819" 와 "규칙은 커버리지 100%에서
0.494" 를 나란히 적었다. **그 비교는 성립하지 않는다.** 답하기 쉬운 것만 골라
답하면 점수는 저절로 오른다.

바로잡으려면 규칙 baseline 에도 기권 장치가 있어야 한다. 규칙의 신뢰도는 자연스럽게
정의된다 — 규칙이 실제로 걸렸으면 `high`, 아무것도 안 걸려 다수 클래스로
떨어뜨렸으면 `low`. majority 는 신호가 아예 없으므로 전부 `low` 다.

## 곡선

**AURC** (Area Under the Risk-Coverage curve) = 커버리지에 대한 위험(오류율)의 평균.
**낮을수록 좋다.**

| 모델 | AURC |
|---|---|
| **LLM** | **0.125** |
| majority | 0.259 |
| keyword | 0.282 |

```
■ majority   AURC 0.259   ← 기권 신호 없음 (곡선이 한 점)
   커버리지   건수    위험   정확도  매크로 F1
    100.0%   170   25.9%   74.1%     0.284

■ keyword    AURC 0.282
     11.8%    20   50.0%   50.0%     0.453   ← 규칙이 걸린 건이 오히려 더 틀린다
    100.0%   170   25.3%   74.7%     0.494

■ llm        AURC 0.125
     13.5%    23    4.3%   95.7%     0.489
     72.9%   124   10.5%   89.5%     0.819
    100.0%   170   21.2%   78.8%     0.587
```

### 규칙의 기권 신호는 없느니만 못하다

`keyword` 는 규칙이 걸린 20건(커버리지 11.8%)에서 위험이 **50.0%** 다. 전체 위험
25.3%의 **두 배**다. 규칙이 자신 있다고 고른 곳에서 오히려 더 틀린다.

그래서 AURC 가 majority(0.259)보다 나쁜 0.282 다. 신호가 없는 것보다 **잘못된 신호가
더 해롭다.** 규칙 baseline 은 자기가 언제 틀리는지 모를 뿐 아니라, 반대로 알고 있다.

거의 같은 커버리지에서 직접 붙여 보면 차이가 선명하다.

```
keyword  11.8% 위험 50.0%   │   llm  13.5% 위험 4.3%
```

**12배 차이다.**

## 결론이 뒤집혔다

AURC 차이도 대응표본 부트스트랩으로 검정했다 (2,000회).

| 비교 | 차이 | 95% CI | 판정 |
|---|---|---|---|
| majority − keyword | −0.023 | −0.099 – +0.052 | 판정 보류 |
| majority − llm | +0.133 | +0.079 – +0.189 | **유의** |
| **keyword − llm** | **+0.157** | **+0.081 – +0.233** | **유의** |

E1 에서는 매크로 F1 으로 LLM 과 규칙을 **가를 수 없었다** (차이 +0.093,
CI −0.061 – +0.247, p≈0.241). 그런데 기권을 허용하고 AURC 로 재니 **명확히 갈린다**
(CI 가 0 을 포함하지 않는다).

> **LLM 의 우위는 더 잘 맞히는 데 있지 않다. 언제 모르는지를 아는 데 있다.**

같은 데이터, 같은 예측, 같은 검정 방법이다. 달라진 것은 **무엇을 성능으로 볼
것인가**뿐이다. 답만 놓고 보면 규칙과 구별되지 않던 모델이, 기권까지 포함해서 보면
확실히 낫다.

이것이 이 프로젝트의 문제의식과 정확히 맞닿는다. 찾으려는 것은 "문서에 없는 판단
기준"이고, 그 작업에서 가장 위험한 실패는 틀린 규칙을 자신 있게 내놓는 것이다.
언제 판단을 멈춰야 하는지 아는 모델이라야 그 작업을 맡길 수 있다.

## 남은 한계

- **신호가 거칠다.** confidence 가 세 등급뿐이라 곡선에 점이 세 개다. 연속적인
  점수(예: 라벨 확률)를 받으면 곡선이 매끄러워지고 원하는 커버리지를 정확히
  맞출 수 있다.
- **majority 와 keyword 는 판정 보류.** 규칙의 기권 신호가 majority 보다 나쁘다고
  단정할 수는 없다 (CI 가 0 을 포함). 다만 좋다는 증거도 없다.
- **AURC 는 위험만 본다.** 클래스 불균형은 반영되지 않으므로 매크로 F1 과 함께
  읽어야 한다.

## 재현

```bash
for s in majority keyword; do
  python3 scripts/baseline_nonaction.py --gold data/eval/nonaction_test.jsonl \
      --output data/processed/pred_nonaction_$s.jsonl --strategy $s
done

python3 scripts/risk_coverage.py --gold data/eval/nonaction_test.jsonl \
    --labels nonaction \
    --pred majority=data/processed/pred_nonaction_majority.jsonl \
    --pred keyword=data/processed/pred_nonaction_keyword.jsonl \
    --pred llm=data/processed/pred_nonaction_llm.jsonl \
    --report experiments/results/e2_risk_coverage.json
```
