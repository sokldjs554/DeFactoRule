# 기저율 표의 출처를 다시 세운다 (Phase B-2a.6 · B-2b prerequisite wiring)

`docs/25 §D.1` 이 B-2b 의 유일한 선결 조건으로 지목한 것을 처리하고(§1~§4),
이어서 분류기 배선까지 이었다(§5~§7). **모델은 아직 부르지 않았다.**

    만든 것   data/eval/dev_base_rates_clean.json      (clean dev 87건)
    건드리지 않은 것  data/eval/dev_base_rates.json     (legacy dev 85건)
    구현     app/evaluation/base_rate_asset.py · scripts/base_rate_asset.py
    검사     tests/regression/test_base_rate_provenance.py (15개)

**API 0회 · 프롬프트 문구 변경 0 · 문턱 변경 0 · Router 변경 0.**
production 코드는 `classifier.py` 배선만 바뀌었고, **기본값 실행은 예전과 같다.**

---

## 1. 왜 이것만 먼저인가

기저율 표는 **LLM 프롬프트로 들어간다**(`app/agents/classifier.py`). 그런데
지금 실린 표는 legacy dev 85건에서 나왔고, **clean test 168건 중 54건이 그
legacy dev 안에 있었다**(`docs/25 §A.8`). 그 54건의 정답이 집계된 형태로
사전확률에 실려, 바로 그 54건을 분류할 때 프롬프트에 들어간다.

결정론 계층에서는 54건 겹침이 이득으로 나타나지 않았다(`docs/25 §C.4`).
그러나 그것은 **결정론 계층에 대한 이야기**다. 프롬프트에 들어가는 표는
아직 검증된 적이 없었고, B-2b 는 정확히 그 경로를 쓴다.

## 2. legacy 와 clean

### 전체 분포

| | legacy dev 85 | **clean dev 87** | 차이 |
|---|---|---|---|
| 비조치 | 0.682 | **0.724** | +4.2%p |
| 조치 | 0.094 | **0.092** | −0.2%p |
| 기타 | 0.224 | **0.184** | −4.0%p |

### 업권별 — 쓰는 업권이 7개에서 6개로 줄어든다

`MIN_SECTOR_N = 5` 미만이면 업권값 대신 전체값을 쓴다. **`보험` 이 legacy 6건에서
clean 4건으로 내려가 프롬프트에서 빠진다.**

| 업권 | legacy n · 비조치 | clean n · 비조치 | 프롬프트 |
|---|---|---|---|
| 공통 | 31 · 0.548 | 36 · **0.667** | 양쪽 사용 |
| 전자금융 | 18 · 0.500 | 16 · 0.500 | 양쪽 사용 |
| 상호저축은행업 | 8 · 0.875 | 8 · 0.875 | 양쪽 사용 |
| 은행 | 6 · 0.833 | 7 · **0.714** | 양쪽 사용 |
| 여신전문금융업 | — · 1.000 | 7 · 1.000 | 양쪽 사용 |
| 자본시장 | 6 · 0.833 | 5 · 0.800 | 양쪽 사용 |
| **보험** | 6 · 0.857 | **4** · — | **clean 에서 빠진다** |
| 가상자산 | 4 · — | 4 · — | 양쪽 다 빠진다 |

가장 크게 움직인 것은 `공통` 이다(비조치 0.548 → 0.667). clean test 168건 중
57건이 `공통` 이므로, 이 칸의 변화가 프롬프트에 실제로 닿는 범위는 작지 않다.

**어느 쪽이 더 정확한 사전확률인지는 여기서 말하지 않는다.** clean 평가에
쓸 표는 clean dev 에서 나온 것이어야 한다는 것만이 이 단계의 근거다.

## 3. 출처를 이름표가 아니라 지문으로 남긴다

`split: "clean"` 이라고 적어 두는 것만으로는 부족하다 — 손으로 고칠 수 있는
글자다. 그래서 **행 키의 지문**을 함께 적는다.

```json
"source": "clean_dev",
"split": "clean",
"n": 87,
"provenance": {
  "schema_version": 1,
  "method": "app.domain.base_rates.compute",
  "method_version": "4857329245fbf764",
  "input": "data/eval/nonaction_dev_clean.jsonl",
  "input_sha256": "a10324aa…",
  "row_key_digest": "c228ccc2b3cea21b…",
  "n_rows_read": 87,
  "test_files_read": []
}
```

    row_key_digest   정렬한 (source, page, serial, pair_index) 목록의 SHA-256
    method_version   `base_rates.compute` 원문의 SHA-256 앞 16자

검증은 **재계산**이다. clean dev 파일에서 지문을 다시 만들어 같은지 본다.
이름표는 거짓말할 수 있지만 지문은 못 한다. `method_version` 도 같은 이유다 —
셈법이 바뀌면 지문이 바뀌므로, 옛 표가 새 코드의 산출물인 척할 수 없다.

**분포는 다시 계산하지 않았다.** `app.domain.base_rates.compute` 가 그대로
계산하고, 이 모듈은 결과에 출처를 붙이고 붙인 것이 사실인지 다시 잴 뿐이다.
절차를 복제하면 두 값이 언젠가 갈린다.

### 코드로 막아 둔 것 셋

    입력 파일명에 "test" 가 있으면        거부
    출력이 legacy 기저율 파일이면          거부
    출력 파일이 이미 있으면                거부 (조용한 덮어쓰기 금지)

## 4. split identity 검사 — 15개

`probes.base_rates_come_from_dev_only` 는 `source == "dev"` 만 본다. 이제 dev 가
둘이므로 그것은 안전을 뜻하지 않는다. 새 검사는 **어느 dev 인지**를 본다.

    clean 표가 clean dev 로 완전 검증을 통과한다                        ✅
    행 지문이 clean dev 와 같고 legacy dev 와 다르다                    ✅
    split·source·n 이 선언한 대로다 (clean / clean_dev / 87)            ✅
    표를 만든 셈법이 지금 도는 셈법과 같다                               ✅
    clean dev ∩ clean test = ∅ 이고 표가 clean dev 로 재현된다 —
      그러므로 test 행이 기여할 자리가 없다                              ✅
    legacy 표는 여전히 legacy dev 로만 재현되고 clean dev 로는 안 된다   ✅
    이름표만 바꾼 표는 거부된다 (split · source · n · 분포 · 지문)       ✅

## 5. `classifier.py` 배선 (Phase B-2b prerequisite wiring)

`docs/25 §D.1` 이 미뤄 둔 (가)+(나)를 함께 적용했다. **모델은 부르지 않았다.**

### 5.1 무엇이 바뀌었나

    --base-rates <path>   기본값은 legacy 표. 주지 않으면 기존 명령이 그대로 돈다
    --dry-run             기저율 표를 검증하고 프롬프트에 실릴 것까지만 만든 뒤 멈춘다
                          `anthropic` 을 임포트하기 전에 끝난다 — 네트워크·자격증명 0

    build_parser()        인자 정의를 main 밖으로 뺐다. 기본값을 **부르지 않고**
                          확인할 수 있어야 테스트가 배선을 검사할 수 있다
    select_targets()      실행과 dry-run 이 같은 행을 보게 한다 (로직 이동, 변경 없음)
    write_manifest()      예측 파일 옆에 `*.manifest.json` 을 남긴다

옛 가드 두 줄은 `load_validated()` 한 줄로 바뀌었다. **읽기와 검사를 한 함수에
묶은 것이 요점이다** — 예전 가드가 `source == "dev"` 한 줄로 남아 있던 이유는
검사가 호출부에 흩어져 있었기 때문이다.

### 5.2 검증은 이름표가 아니라 지문을 본다

`validate()` 가 보는 것.

| 검사 | 어떻게 |
|---|---|
| `source`·`split` 짝 | `{("dev","legacy"), ("clean_dev","clean")}` 에 있는가 |
| **split ↔ 입력 파일 결속** | `legacy` 는 `nonaction_dev.jsonl`, `clean` 은 `nonaction_dev_clean.jsonl` 에서만 나올 수 있다 |
| `method` · `method_version` | 지금 도는 `base_rates.compute` 원문의 해시와 같은가 |
| `input` | 파일이 실재하는가 · 이름에 `test` 가 있으면 거부 |
| `input_sha256` | 그 파일을 다시 해시해 대조 |
| `row_key_digest` | 그 파일의 행 키를 다시 정렬해 해시해 대조 |
| `n` · `n_rows_read` | 그 파일의 행 수와 같은가 |
| `overall` · `sectors` · `min_sector_n` | 그 파일로 **다시 계산**해 대조 |

**split ↔ 입력 파일 결속은 회귀 테스트가 찾아낸 구멍이다.** 그것이 없으면
"legacy dev 로 만들었는데 `split: clean` 이라고 적은 표" 가 통과한다 —
내부적으로는 지문도 분포도 전부 앞뒤가 맞기 때문이다. 이름표를 **행 집합에
묶어야** 비로소 막힌다.

### 5.3 legacy 표에 출처를 다시 찍었다

`split == legacy` 를 검증하려면 legacy 표에도 이름표와 지문이 있어야 한다.
그래서 **값은 한 글자도 바꾸지 않고 출처만 붙였다**(`--restamp`).

    추가된 키   provenance · split          사라진 키  없음
    기존 키     source · n · min_sector_n · overall · sectors — **값 전부 동일**

`--restamp` 는 옛 표와 분포가 다르면 **쓰지 않고 죽는다.** 재각인이 조용한
재계산으로 바뀌는 것을 막는 장치다.

### 5.4 실행 기록

예측 레코드의 형은 건드리지 않았다 — 채점 하네스가 읽는 필드가 바뀌면 옛
예측 파일과 새 것을 나란히 놓을 수 없다. 대신 옆에 한 장을 남긴다.

    <output>.manifest.json
      task · model · input · output · limit · context · n_targets · dry_run
      base_rates_asset { path · source · split · n · row_key_digest ·
                         method_version · input · input_sha256 }

이 파일 하나면 "이 수치는 어느 기저율 표에서 나왔나" 를 되짚을 수 있다.

## 6. dry-run 검증 결과 — **API 0회**

### legacy 경로 (인자를 주지 않는다)

```
$ python3 scripts/classify_llm.py --task nonaction_sector     --input data/eval/nonaction_test.jsonl --output …/legacy_pred.jsonl     --limit 5 --dry-run

기저율 legacy/dev n=85 · 행 지문 21fe581e111e3e72… · data/eval/dev_base_rates.json
  [전체]   … 과거 유사 사례 85건에서 결론 분포는 비조치 68%, 조치 9%, 기타 22% …
  [전자금융] … 같은 분야 과거 사례 18건에서 … 비조치 50%, 조치 17%, 기타 33% …
```

### clean 경로

```
$ … --base-rates data/eval/dev_base_rates_clean.json --dry-run

기저율 clean/clean_dev n=87 · 행 지문 c228ccc2b3cea21b… · data/eval/dev_base_rates_clean.json
  [전체]   … 과거 유사 사례 87건에서 결론 분포는 비조치 72%, 조치 9%, 기타 18% …
  [전자금융] … 같은 분야 과거 사례 16건에서 … 비조치 50%, 조치 19%, 기타 31% …
```

**clean 표의 값이 프롬프트 문장으로 그대로 옮겨졌다.** 87건 · 비조치 72% ·
전자금융 16건 50/19/31 — `dev_base_rates_clean.json` 에 적힌 것과 같다.

### 위조 표는 실행 경로에서 거부된다 (dry-run 이 아니라 실제 실행)

```
$ … --base-rates forged.json          # clean 표의 source 를 "dev" 로 고친 것
…/forged.json 의 출처를 확인하지 못했습니다:
  - source/split 짝이 허용 목록에 없다: ('dev', 'clean') …
exit=1        출력 파일 생성 안 됨 · anthropic 임포트 전에 종료
```

## 7. 거부되어야 하는 것 — 전부 거부된다

`tests/unit/test_base_rate_validation.py` 16개 · `tests/regression/…` 20개.

    legacy 표에 split=clean 조작                    거부 (split ↔ 파일 결속)
    clean 표에 source=dev 조작                      거부 (짝 목록)
    둘 다 legacy 로 조작                            거부 (split ↔ 파일 결속)
    row_key_digest 불일치                           거부
    input_sha256 불일치                             거부
    method_version 불일치                           거부
    provenance 블록 없음                            거부
    n 불일치                                        거부
    input 을 다른 dev 파일로 바꿔치기                거부 (지문 둘이 동시에 어긋난다)
    legacy dev 로 만들고 clean 이라 주장             거부
    clean test 로 만든 표 (빌더 가드 우회)           거부 ("test 파일에서 만든 표다")
    load_validated 는 나쁜 표를 돌려주지 않는다      ProvenanceError

## 8. 이번 단계에서 하지 않은 것

    LLM/API 호출 0 · E7~E11b 0 · 프롬프트 문구 변경 0 · 문턱 변경 0
    Router 변경 0 · sector matcher 0 · E6 규칙 0 · Temporal 0 · S5 0 · UI 0
    legacy 기저율의 **분포 값** 변경 0 (출처 메타데이터만 추가)
