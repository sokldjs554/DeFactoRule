# 실패 케이스 레지스트리 — 71건

명세 §11 은 "최소 30개 이상의 실패 케이스를 의도적으로 구축하고, 각 실패를
taxonomy 로 분류하고, 개선 전/후를 숫자로 비교한다" 를 요구한다.

## 산문으로 적힌 실패 목록은 §11 이 아니다

"이런 걸 고쳤다" 를 문서에 적어 두는 것만으로는 아무것도 지켜지지 않는다.
다음에 같은 자리가 깨졌을 때 알려줄 사람이 없기 때문이다. 실제로 이 프로젝트는
2024년판 사례집을 넣었을 때 결론 검출률이 97%에서 38%로 떨어졌는데도 파싱은
"성공" 했고 건수도 정상이었다.

그래서 케이스마다 **실행 가능한 probe** 를 붙였다.

```bash
python3 scripts/failure_report.py
python3 scripts/failure_report.py --layer extraction
```

`tests/regression/test_failure_registry.py` 가 매번 전부 돌린다. `status` 와
실행 결과가 어긋나면 실패한다 — **어느 방향이든** 어긋나는 것이 문제다.
고쳤다는 케이스가 깨졌으면 회귀이고, 열려 있다는 케이스가 통과하면 레지스트리가
낡은 것이다.

레지스트리는 `data/failures/registry.jsonl` 이고, 71건 중 64건에 probe 가 있다.

## Taxonomy

<!-- TAXONOMY:시작 -->
| 계층 | 건수 | 범주 |
|---|---|---|
| extraction | 16 | boundary-missplit 5 · format-unhandled 4 · silent-empty 3 · encoding-normalization 3 · unreadable-source 1 |
| labeling | 8 | answer-leakage 3 · split-discipline 3 · label-conflation 2 |
| retrieval | 1 | degenerate-representation 1 |
| evaluation | 22 | metric-misuse 6 · misdiagnosis 5 · sample-mismatch 4 · partial-guard 2 · incomparable-comparison 1 · undiagnosable-discard 1 · phantom-evidence 1 · uniform-threshold 1 · arbitrary-tiebreak 1 |
| agent | 10 | miscalibration 3 · ungrounded-evidence 2 · schema-violation 2 · prior-overcorrection 1 · undiagnosable-discard 1 · unverified-premise 1 |
| infrastructure | 14 | reproducibility 4 · continuous-integration 2 · environment 2 · contract-violation 2 · error-classification 1 · undiagnosable-discard 1 · path-resolution 1 · misleading-estimate 1 |
<!-- TAXONOMY:끝 -->

계층이 하나라도 비면 테스트가 실패한다. 한 곳에만 실패가 몰려 있다면 나머지를
들여다보지 않은 것이다.

## 재발 패턴 — 개별 사례가 아니라 패턴을 지킨다

레지스트리의 probe 는 **과거의 그 자리**를 지킨다. 같은 실수가 **다른 자리에서**
다시 나오는 것은 막지 못한다. 실제로 이런 일이 있었다.

| 자리 | 무엇을 버렸나 | 결과 |
|---|---|---|
| API 오류 (IN-02) | 메시지와 본문 | 39건이 왜 죽었는지 모름 |
| 결측 검사 (EV-09) | 파일에 없는 행 | 156/170 이 "결측 0" 으로 통과 |
| 기준 검증 (AG-09) | 걸러낸 기준 | 0개인 이유를 되짚을 수 없음 |
| 규칙 학습 (EV-16) | 문턱 미달 후보 165개 | 후보가 없었나 문턱이 높았나 모름 |

**넷 다 따로 고쳤고 넷 다 probe 를 붙였는데, 다섯 번째를 막을 장치가 없었다.**

범주(category)는 재발 단위가 아니다. 21개 범주 중 17개가 이미 2건 이상인데
그것을 전부 "반복" 이라고 부르면 아무 뜻도 없다. 진짜 반복은 **범주를
가로지른다** — 위 넷은 infrastructure · evaluation · agent 세 계층에 흩어져 있다.

그래서 `pattern` 필드를 따로 두고, **2건 이상 붙은 패턴에는 패턴 단위 가드를
의무화**한다.

| 패턴 | 사례 | 가드 |
|---|---|---|
| `discard-unrecorded` | AG-09 · EV-16 · IN-02 | `every_filter_stage_records_its_discards` |
| `mismatched-sample` | EV-01 · EV-08 · EV-09 | `comparisons_align_their_samples` |
| `enumeration-as-separator` | EX-12 · EX-16 | `marks_must_align_before_splitting` |

`discard-unrecorded` 가드는 파이프라인의 걸러내기 단계 여섯 개를 등록해 두고,
각 단계에 **거부당할 입력을 실제로 넣어** 결과물에 이유가 들어 있는지 본다.
"기록하는 코드가 있다" 가 아니라 "기록이 실제로 나온다" 를 보는 것이다.

한계도 적어 둔다. **새 단계를 만들면서 등록을 잊으면 이 가드는 그것을 모른다.**
없앨 수 없는 구멍이고, 대신 등록 개수를 테스트가 감시해 줄어들면 걸린다.

두 가지를 실제로 확인했다.

- 규칙 학습기의 폐기 기록을 지우자 `AG-09 수정이 풀렸습니다 — 기록하지 않는
  단계 ['rule-induction']` 으로 걸렸다.
- 가드 없는 새 패턴을 두 사례에 붙이자 `패턴 가드가 없는 반복 패턴이 있습니다`
  로 걸렸다.

## 개선 전 → 후

41건에 수치가 있다. 수치는 두 종류로만 적는다. `measured` 는 실제로 재 본 값이고 출처를 함께
남긴다. `live` 는 probe 가 실행 시점에 직접 계산한 값이며, 옛 구현을 함께 들고
있어 before 와 after 를 **같은 입력에서** 잰다. 재 보지 않은 것은 적지 않는다.

<!-- METRICS:시작 -->
| ID | 지표 | 전 → 후 | 종류 |
|---|---|---|---|
| AG-03 | 조치 재현율 | 0.286 → 0.071 | measured |
| AG-07 | 잔재가 든 판단이유 | 252 → 0건 | live |
| AG-10 | 2건 이하인 보정표 칸 | 4 → 0개 | live |
| EV-01 | 커버리지 | 0.176 → 1.000 | measured |
| EV-02 | 다수 클래스만 예측 (정확도 → 매크로 F1) | 0.741 → 0.284 | live |
| EV-03 | AURC (keyword − llm) | 0.282 → 0.125 | measured |
| EV-08 | 결측이 있는 예측 파일 | 3 → 0개 | live |
| EV-09 | 156/170 파일에서 검출한 결측 | 0 → 14건 | live |
| EV-10 | 잘못 등록된 probe | 1 → 0개 | live |
| EV-11 | 학습된 규칙 | 3 → 9개 | measured |
| EV-13 | 매크로 F1 비교에서 유의 판정 | 9 → 7쌍 | measured |
| EV-14 | 판정이 뒤집힌 채 남은 비교 | 1 → 0건 | measured |
| EV-15 | 전체 규칙 중 조각에서 재발견된 것 | 2 → 5개 | measured |
| EV-16 | 기록된 폐기 후보 | 0 → 165개 | live |
| EV-17 | 밀린 채 통과한 코퍼스 수치 | 1 → 0건 | measured |
| EV-18 | 증거 0건인 조치에 붙은 가중치 (로그승산) | 0.201 → -0.981 | live |
| EV-19 | 치우친 표본(다수 20·소수 3)에서 살아남은 소수 클래스 기준 | 0 → 3개 | live |
| EV-20 | 근거 없는 입력의 예측 (라벨) | 기타 → 비조치 | live |
| EV-21 | 잡음 표본에서 나온 조치 재현율 | 0.964 → 0.250 | live |
| EV-22 | 조치 재현율에 붙는 불확실성 폭 (구간 폭) | 0.000 → 0.653 | live |
| EX-01 | 결론 미검출 | 49 → 2건 | measured |
| EX-04 | missing_field:판단이유 | 54 → 2건 | measured |
| EX-05 | 항목명 잔재 | 406 → 0건 | live |
| EX-06 | 업권 미분류 | 159 → 0건 | measured |
| EX-09 | 읽을 수 있는 글자 비율 | 0.374 (통과 문턱 0.800) | measured |
| EX-12 | 오분할 쌍 | 81 → 0쌍 | measured |
| EX-13 | 키 기준 중복 | 153 → 0건 | live |
| EX-14 | 비조치 F1 | 0.786 → 0.816 | measured |
| EX-16 | 오분할된 사례 | 2 → 0건 | measured |
| IN-01 | 실패가 확정된 호출 | 78 → 0회 | measured |
| IN-02 | 진단 가능한 실패 | 0 → 39건 | measured |
| IN-08 | 낡은 채로 짝지어질 뻔한 예측 | 14 → 0건 | measured |
| IN-09 | 테스트 | 25 → 59개 | measured |
| IN-10 | 호출 없이 잡히는 계약 위반 | 0 → 1건 | live |
| IN-11 | 기준 88개에서 답 JSON 대비 상한 여유 (배) | 1.030 → 6.400 | live |
| IN-12 | dev 85건 추정 비용 (달러) | 14.290 → 5.110 | live |
| IN-13 | 기본값으로 재현한 함정 구간 | 10 → 15건 | live |
| IN-14 | 오염된 채 커밋된 산출물 | 1 → 0건 | measured |
| LB-01 | 누출 표현이 있는 요청문 | 61 → 0건 | live |
| LB-03 | 마스크 토큰이 있는 사례 | 60 → 0건 | live |
| RT-01 | 자기 자신을 찾지 못한 선례 | 2 → 0건 | live |
<!-- METRICS:끝 -->

**AG-03 은 개선이 아니다.** 전체 기저율을 프롬프트에 넣으면 소수 클래스가
지워진다는 것이 실험 결과였고, 그것을 그대로 남긴다. 정확도는 78.8%에서
78.2%로 거의 움직이지 않았다 — 정확도만 봤다면 "변화 없음" 으로 넘어갔을
실패다.

## 레지스트리를 만들다 찾은 것

probe 를 쓰는 동안 세 건이 새로 드러났다.

**EX-05 — 항목명의 뒷조각이 값에 남는다.** PDF 에서 "판단이유" 가
"판단\n이유" 로 뽑힐 때, 파서는 이름을 올바로 알아보고도 값의 시작을 이름 줄
**하나** 뒤로만 잡았다. 그래서 "이유" 가 값 맨 앞에 붙었다. 1,095건 중 406건.
값이 비지 않으므로 건수로는 알 수 없다.

더 나쁜 것은, gold set 을 만들 때 `^행위\n` 만 지우는 정규식을 넣어 둔 것이
사실 이 버그의 **증상을 하류에서 덮고 있었다**는 점이다. 그 우회로가 234건만
가리고 나머지 172건은 그대로 통과시켰다. 증상을 덮으면 원인을 찾을 이유가
사라진다.

**IN-03 — 파일을 옮기자 경로가 조용히 틀어졌다.** 아키텍처 재편에서 분류기를
`app/agents/` 로 옮기자 `parents[1]` 이 저장소 루트에서 `app/` 이 됐고, 기저율
경로가 `app/data/eval/...` 를 가리켰다. API 키가 필요한 경로라 어떤 테스트도
건드리지 않는 자리였다.

**IN-08 — 예측이 낡았다는 사실이 드러나지 않는다.** EX-05 를 고치자 gold 본문이
바뀌었는데, 키(출처·쪽·번호·쌍)는 그대로라 낡은 예측이 새 gold 와 조용히
짝지어졌다. test 170건 중 14건. 이제 예측 레코드에 프롬프트 지문을 남기고
`--resume` 이 대조한다.

## probe 자체도 틀릴 수 있다

첫 실행에서 세 건이 실패했는데 그중 둘은 probe 의 결함이었다.

- `bootstrap_is_seeded` — 라벨 조합이 단조로워 매크로 F1 이 몇 개의 값만 갖는
  입력을 썼다. 서로 다른 시드가 우연히 같은 백분위를 냈다.
- `code_parses_on_python_39` — `StrEnum` 이라는 **낱말**이 나오는지만 봤다.
  `labels.py` 는 왜 그것을 쓰지 않는지를 docstring 에 적어 두었고, 검사가 거기
  걸렸다. 실제 `import` 와 상속을 보도록 고쳤다.

세 번째만 진짜였다 — `predictions_are_complete_before_reporting` 이 sector 예측의
결측 39건을 잡았다. **이 케이스는 지금도 열려 있다.**

## 열린 케이스

**EV-08 — 편향된 부분집합으로 3-way 비교를 하려 했다.** sector 예측이
131/170 만 성공했을 때, 빠진 39건은 무작위가 아니었다. 2025년 35건 + 2024년
4건이고 라벨도 비조치 37 / 기타 1 / 조치 1 로 쏠려 있다. 살아남은 131건은
소수 클래스 비율이 32%로 전체 26%보다 높다. 그 위에서 계산한
`sector 0.630 > base 0.578 > prior 0.497` 은 **결과가 아니다.**

170건이 다 채워질 때까지 이 비교는 보고하지 않는다.
