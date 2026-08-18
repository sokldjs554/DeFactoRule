# DeFactoRule

**규제 예외 승인 이력에서, 문서 어디에도 적혀 있지 않은 판단 기준을 복원한다.**

## 현재 상태

| Phase | 내용 | 상태 |
|---|---|---|
| **−1** | 주제 발굴 35개 → 채점 → 최종 1개 선정 | ✅ 완료 |
| **−1b** | 데이터 생존 게이트 | ✅ 조건부 통과 (설계 변경) |
| **1a** | 사례집 파서 · 코퍼스 실측 | ✅ **1,095건 — W1 게이트 통과** |
| **1b** | 질의–회답 분할 · 서식 회귀 테스트 | ✅ 1,124쌍 · 테스트 25개 |
| **1c** | 띄어쓰기 복원 | ✅ 45건 복원 · 비조치 F1 0.816 |
| **2a** | 라벨 체계 · 규칙 baseline · 평가 하네스 | ✅ 커버리지 41.1% |
| **2b** | LLM 분류기 | ⏸ API 키 필요 |
| **2c** | 비조치 트랙 baseline (순환 없는 평가) | ✅ 매크로 F1 0.494 |
| **2d** | E1 — LLM vs baseline | ✅ 매크로 F1 0.587 (규칙과 판정 보류) |
| **2e** | E2 — 위험-커버리지 곡선 | ✅ AURC 0.125 vs 0.282 (**유의**) |
| **2f** | E3 — 업권별 분석 | ✅ E1 진단 정정 · 전자금융은 어려운 구간 |
| **2g** | E4 — 기저율 프롬프트 변형 | ⏸ 실행 대기 (~$4) |
| 3~11 | 구현 | ⬜ 대기 |

- [Phase −1 주제 선정 보고서](docs/01-topic-research.md)
- [W1 게이트 실측 결과](docs/02-w1-gate.md)
- [질의–회답 분할과 서식 회귀 테스트](docs/03-splitting-and-regression.md)
- [띄어쓰기 복원](docs/04-spacing-restoration.md)
- [Phase 2 — 결론 판정 baseline](docs/05-phase2-baseline.md)
- [비조치 트랙 baseline — 순환 없는 평가](docs/06-nonaction-baseline.md)
- [E1 — LLM vs 규칙 baseline](docs/07-e1-llm-vs-baseline.md)
- [E2 — 위험-커버리지 곡선](docs/08-e2-risk-coverage.md)
- [E3 — 업권별 분석](docs/09-e3-sector-analysis.md)
- [E4 — 기저율 프롬프트 변형](docs/10-e4-prompt-variants.md)

---

## 실행하기

**Python 3.9 이상이 필요합니다.** macOS 에서 `python` 은 시스템에 남아 있는 2.7 을
가리키는 경우가 많습니다. 반드시 `python3` 를 쓰세요.

```bash
git clone https://github.com/sokldjs554/DeFactoRule
cd DeFactoRule                      # 모든 명령은 저장소 루트에서

pip3 install -r requirements.txt
python3 scripts/check_env.py        # 환경 점검 — 무엇이 빠졌는지 알려줍니다
```

`check_env.py` 는 Python 2 로 실행해도 문법 오류 없이 돌아가며, 인터프리터가
잘못됐다는 사실을 알려줍니다.

### 파이프라인

```bash
# 1. 사례집 PDF 를 data/raw/casebooks/ 에 넣는다 (data/SOURCES.md 참고)
python3 scripts/parse_casebook.py --input data/raw/casebooks --output data/processed

# 2. 띄어쓰기 복원
python3 scripts/restore_spacing.py train --input data/processed --model models/spacing.json
python3 scripts/restore_spacing.py apply --input data/processed --model models/spacing.json --threshold -0.25

# 3. 질의-회답 쌍으로 분할
python3 scripts/split_queries.py --input data/processed --output data/processed

# 4. 평가셋 (비조치 트랙 — 정답이 문서 체크박스)
python3 scripts/make_nonaction_gold.py --input data/processed/cases_nonaction.jsonl --output data/eval

# 5. baseline
python3 scripts/baseline_nonaction.py --gold data/eval/nonaction_test.jsonl \
    --output data/processed/pred_nonaction_majority.jsonl --strategy majority
python3 scripts/evaluate.py --gold data/eval/nonaction_test.jsonl \
    --pred data/processed/pred_nonaction_majority.jsonl --labels nonaction --name majority
```

### LLM 분류기

```bash
export ANTHROPIC_API_KEY=...        # 또는 `ant auth login`
python3 scripts/classify_llm.py --task nonaction \
    --input data/eval/nonaction_test.jsonl \
    --output data/processed/pred_nonaction_llm.jsonl --limit 30
```

`--limit` 로 먼저 소량 실행하면 추정 비용이 출력됩니다.

### 테스트

```bash
pip3 install -r requirements-dev.txt
python3 -m pytest tests -q
```

---

## 1. Problem

어떤 조직이든 규정에는 **원칙**만 적혀 있고, 실제 판단은 **예외 승인 이력에 축적된 암묵 기준**을 따른다.
"이런 조건이면 예외를 허용한다"는 규칙이 문서 어디에도 없이 관행으로 존재한다.
담당자가 바뀌면 이 기준이 사라지고, 판단이 흔들린다.

금융규제 영역에서 이 현상은 공개 데이터로 관측된다.
혁신금융서비스는 지정될 때마다 **부가조건**이 붙는다 —
"계모임 최대 3개, 구좌당 월 20만원, 1인당 50만원, 지정기간 2년".
이 숫자를 결정한 기준은 **어디에도 공표된 적이 없다.**

## 2. Why This Problem

- 사람이 처리하는 방식: 경력자의 머릿속. 인수인계 문서에는 "케이스 바이 케이스"라고 적힌다.
- RAG로 안 되는 이유: **정답이 문서에 없다.** 검색은 적혀 있는 것만 찾는다.
- 필요한 것은 생성이 아니라 **귀납** — 사례 집합에서 규칙을 도출하고, 그 규칙이 실제 판단을 예측하는지로 검증해야 한다.

## 3. What Makes It Different

| 흔한 프로젝트 | DeFactoRule |
|---|---|
| 문서에 적힌 답을 찾는다 | 문서에 없는 규칙을 발견한다 |
| LLM이 답을 생성한다 | LLM은 사례를 피처로 구조화하고, 규칙은 ML이 학습한다 |
| "LLM이 그럴듯한 이유를 말했다"로 끝 | **held-out 예측 정확도로 규칙의 진위를 검증**한다 |
| 못 찾은 부분은 실패 | 못 찾은 부분을 **재량의 크기**로 측정한다 (`Discretion Residual`) |

## 4. 데이터 (Phase −1b 실측)

| 트랙 | 코퍼스 | 규모 | 라벨 구조 |
|---|---|---|---|
| **Track A** | 법령해석 회신문 + 비조치의견서 (2021–2025 실측) | 1,095 사례 → 1,124 질의–회답 쌍 | 이진 (해당/미해당, 비조치/조치/기타) |
| **Track B** | 혁신금융서비스 지정 + 부가조건 | 미착수 | 다중 라벨 + 수치 |

**게이트에서 확인된 제약**: 반려·미지정 사례는 공개되지 않는다.
"승인 vs 거부" 이진 라벨을 만들 수 없어, Track B의 타겟을
**"어떤 부가조건이 어느 수준으로 붙는가"**로 재정의했다. → [상세](docs/01-topic-research.md)

## 5. Evaluation

| 지표 | 정의 |
|---|---|
| `Rule Predictive Accuracy` | 도출 규칙의 held-out 예측 정확도 — 규칙의 진위 검증 |
| `Discretion Residual` ★ | 부가조건 중 신청 특성으로 설명되지 않는 잔차 = **재량의 크기** |
| `Undocumented Rule Rate` | 도출 규칙 중 원 규정에 명시되지 않은 비율 — 암묵성의 정량화 |
| `Condition Level MAE` | 수치형 부가조건(한도·기간) 예측 오차 |
| `Contradiction Rate` | 명시 규정과 충돌하는 규칙 비율 |

## 6. 개발 환경 주의사항

데이터 수집 대상 도메인은 개발 컨테이너에서 접근이 차단되어 있다.

```
better.fsc.go.kr    EGRESS_BLOCKED   (금융규제·법령해석포털)
www.data.go.kr      EGRESS_BLOCKED   (공공데이터포털)
```

**수집 스크립트는 로컬에서 실행한다.** 수집 결과를 커밋한 뒤 전처리·모델링·평가를 진행한다.

---

*Phase −1 보고서 기준 2026-08-18. 실행하지 않은 수치는 이 저장소 어디에도 기록하지 않는다.*
