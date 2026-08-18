# E4 — 기저율을 알려주면 과잉 예측이 줄어드는가 (실행 대기)

## 가설

E3 에서 확인한 것 — 모델이 소수 클래스를 과잉 예측한다. `조치` 라벨은 전자금융과
공통에만 존재하고 나머지 여섯 업권에는 한 건도 없는데, 모델이 그것을 모른 채 모든
업권에서 세 라벨을 고려한다. 정답이 전부 비조치인 업권에서 굳이 다른 답을 골라
틀린 사례가 4건 있었다.

**기저율을 프롬프트에 넣으면 그 과잉 예측이 줄어드는가?**

## 두 변형을 따로 두는 이유

`nonaction_sector` 하나만 돌리면 결과를 해석할 수 없다. 그 변형은 **업권을 알려주는
것**과 **그 업권의 기저율을 알려주는 것**을 동시에 바꾸기 때문이다. 성능이 오르면
둘 중 무엇 덕인지 알 수 없다.

| 변형 | 프롬프트에 추가되는 것 | 무엇을 재는가 |
|---|---|---|
| `nonaction` | (없음) | 대조군 — E1 에서 이미 실행 |
| `nonaction_prior` | 전체 기저율만, 업권은 밝히지 않음 | 기저율 효과만 |
| `nonaction_sector` | 업권 이름 + 그 업권 기저율 | 둘의 합 |

`sector` − `prior` 가 업권 정보 자체의 몫이 된다.

## 기저율은 dev 에서만 뽑는다

test 에서 뽑아 프롬프트에 넣으면 정답을 흘리는 것이고, 그 실험은 무의미해진다.
`scripts/base_rates.py` 는 `data/eval/nonaction_dev.jsonl` 만 읽으며,
`classify_llm.py` 는 파일의 `source` 가 `dev` 가 아니면 실행을 거부한다.

dev(85건)에서 산출한 값:

| | 비조치 | 조치 | 기타 |
|---|---|---|---|
| 전체 | 68.2% | 9.4% | 22.4% |

| 업권 | 건수 | 사용 | 분포 |
|---|---|---|---|
| 공통 | 31 | 업권 | 비조치 55% · 조치 13% · 기타 32% |
| 전자금융 | 18 | 업권 | 비조치 50% · 조치 17% · 기타 33% |
| 상호저축은행업 | 8 | 업권 | 비조치 88% · 기타 12% |
| 보험 | 7 | 업권 | 비조치 86% · 기타 14% |
| 은행 | 6 | 업권 | 비조치 83% · 조치 17% |
| 자본시장 | 6 | 업권 | 비조치 83% · 기타 17% |
| 여신전문금융업 | 6 | 업권 | 비조치 100% |
| 가상자산 | 3 | **전체** | 표본 부족 — 전체 값으로 대체 |

표본 5건 미만인 업권은 업권별 값을 쓰지 않는다. 3건짜리 분포를 100% 라고 적어
주면 잡음을 신호로 위장하는 셈이다.

### dev 와 test 의 분포는 다르다 — 그것도 실험의 일부다

| | dev | test | 차이 |
|---|---|---|---|
| 비조치 | 68.2% | 74.1% | +5.9%p |
| 조치 | 9.4% | 8.2% | −1.2%p |
| 기타 | 22.4% | 17.6% | −4.7%p |

프롬프트에 넣는 값이 test 의 실제 분포와 6%p 정도 어긋난다. 이것은 결함이 아니라
현실이다. 운영 환경에서도 과거 분포로 미래를 안내하게 되며, **어긋난 기저율이
성능을 해치는지**까지가 이 실험의 질문이다.

## 예상되는 반작용

기저율을 주면 모델이 다수 클래스로 더 쏠릴 수 있다. 그러면 정확도는 오르는데
**매크로 F1 은 떨어진다** — 소수 클래스를 아예 예측하지 않게 되기 때문이다.
정확도만 보고 "개선됐다"고 쓰면 안 된다.

특히 여신전문금융업·가상자산처럼 "과거에 조치 결론이 나온 적 없습니다" 를 읽은
모델이 그 업권에서 조치를 영영 예측하지 않게 될 수 있다. 그것이 옳은 학습인지
과잉 순응인지는 test 에서 그 업권에 실제로 조치 사례가 나오는지에 달렸다
(현재 test 에는 없다 — 그래서 이 실험만으로는 판정할 수 없고, 한계로 적어 둔다).

## 실행

```bash
python3 scripts/base_rates.py --dev data/eval/nonaction_dev.jsonl \
    --output data/eval/dev_base_rates.json

python3 scripts/classify_llm.py --task nonaction_prior \
    --input data/eval/nonaction_test.jsonl \
    --output data/processed/pred_nonaction_prior.jsonl

python3 scripts/classify_llm.py --task nonaction_sector \
    --input data/eval/nonaction_test.jsonl \
    --output data/processed/pred_nonaction_sector.jsonl
```

각 약 $2, 합계 약 $4.

## 채점

```bash
for s in prior sector; do
  python3 scripts/evaluate.py --gold data/eval/nonaction_test.jsonl \
      --pred data/processed/pred_nonaction_$s.jsonl --labels nonaction \
      --name $s --report experiments/results/e4_$s.json
done

python3 scripts/compare_models.py --gold data/eval/nonaction_test.jsonl \
    --labels nonaction \
    --pred base=data/processed/pred_nonaction_llm.jsonl \
    --pred prior=data/processed/pred_nonaction_prior.jsonl \
    --pred sector=data/processed/pred_nonaction_sector.jsonl \
    --report experiments/results/e4_comparison.json

python3 scripts/risk_coverage.py --gold data/eval/nonaction_test.jsonl \
    --labels nonaction \
    --pred base=data/processed/pred_nonaction_llm.jsonl \
    --pred prior=data/processed/pred_nonaction_prior.jsonl \
    --pred sector=data/processed/pred_nonaction_sector.jsonl \
    --report experiments/results/e4_risk_coverage.json

python3 scripts/sector_analysis.py --gold data/eval/nonaction_test.jsonl \
    --pred data/processed/pred_nonaction_sector.jsonl --labels nonaction
```

판정은 **대응표본 부트스트랩**으로 한다. 점추정 비교나 주변 신뢰구간 겹침 여부로
결론 내지 않는다. AURC 도 함께 보아 기권 품질이 훼손되지 않았는지 확인한다 —
기저율을 준 탓에 모델이 근거 없이 자신 있어지면 그쪽이 더 큰 손해다.

## 결과

*(실행 후 기록)*
