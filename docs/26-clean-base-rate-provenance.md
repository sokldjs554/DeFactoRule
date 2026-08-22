# 기저율 표의 출처를 다시 세운다 (Phase B-2a.6)

`docs/25 §D.1` 이 B-2b 의 유일한 선결 조건으로 지목한 것을 처리했다.

    만든 것   data/eval/dev_base_rates_clean.json      (clean dev 87건)
    건드리지 않은 것  data/eval/dev_base_rates.json     (legacy dev 85건)
    구현     app/evaluation/base_rate_asset.py · scripts/base_rate_asset.py
    검사     tests/regression/test_base_rate_provenance.py (15개)

**API 0회 · production 코드 0줄 변경 · 프롬프트 문구 변경 0 · 문턱 변경 0.**

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

## 5. `classifier.py` 배선 — **확인만 했고 고치지 않았다**

    app/agents/classifier.py:47   BASE_RATES_PATH = DEV_BASE_RATES
    app/agents/classifier.py:276  task["base_rates"] = json.loads(BASE_RATES_PATH…)
    app/agents/classifier.py:277  if task["base_rates"].get("source") != "dev": sys.exit(…)

세 줄이 말하는 것은 이렇다.

1. **경로가 모듈 상수다.** CLI 인자가 없으므로 legacy 실행과 clean 실행을
   구분할 방법이 지금은 없다.
2. **가드가 `source == "dev"` 를 요구한다.** clean 표의 `source` 는
   `clean_dev` 이므로 **지금 배선으로는 통과하지 못한다.**

즉 파일을 만들어 둔 것만으로는 B-2b 가 clean 표를 쓸 수 없다. 이 사실을
회귀 테스트로 못 박아 두었다(`TestClassifierWiringIsNotReadyYet`) — 고친
것이 아니라 **확인한 것**이다.

필요한 최소 배선은 이렇다. **이번 단계에서 하지 않았고, 결정을 요청한다.**

    (가) `--base-rates` 인자를 두어 실행마다 파일을 고르게 한다
    (나) 가드를 `source in {"dev", "clean_dev"}` 로 넓히되,
         `split` 과 `row_key_digest` 를 함께 확인하도록 바꾼다
    (다) 실행 결과 파일에 어느 기저율 표를 썼는지 기록한다

(나)를 (가) 없이 하면 가드만 헐거워지고 얻는 것이 없다. **둘은 함께 가야 한다.**

## 6. 이번 단계에서 하지 않은 것

    LLM 호출 0 · E7~E11b 0 · Router 변경 0 · 문턱 변경 0
    sector matcher 수정 0 · 프롬프트 문구 변경 0 · Temporal 0 · S5 0
    legacy 기저율 파일 변경 0 · clean test 열람 0
