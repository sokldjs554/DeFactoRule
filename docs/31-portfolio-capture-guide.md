# Portfolio / README screenshot capture guide

이 문서는 실제 실행 화면을 직접 캡처하기 위한 가이드다. 저장소에는 임의의 목업 이미지나 생성된 실행 화면을 커밋하지 않는다.

## 원칙

- 실제 실행 결과만 캡처한다.
- API key, 사용자 경로, 개인 정보가 화면에 나오지 않게 자른다.
- 숫자를 편집하거나 합성하지 않는다.
- synthetic scan은 반드시 `synthetic scanned-document benchmark`라고 표시한다.
- 한 화면에 너무 많은 로그를 넣지 말고, 증거가 되는 부분만 crop한다.
- `100%` 하나를 확대하는 식보다 **입력 품질별 변화와 실패 사례**가 보이게 한다.

## 사전 준비

Ubuntu/Linux 기준:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-kor
pip install -r requirements-dev.txt
```

macOS에서는 Homebrew Tesseract에 한국어 traineddata가 설치되어 있는지 먼저 확인한다.

## Capture 1 - 입력 품질 3단계 비교

실행:

```bash
python scripts/render_document_ai_samples.py \
  --output-dir artifacts/document_ai_capture \
  --sample 12
```

생성 파일:

- `01_clean_220dpi_png.png`
- `02_standard_170dpi_jpeg.jpg`
- `03_degraded_120dpi_jpeg.jpg`

캡처 방법:

- 세 이미지를 같은 크기로 나란히 보여준다.
- 제목은 `Clean / Standard / Degraded synthetic scan` 정도로만 붙인다.
- 실제 고객 스캔이라고 표현하지 않는다.

README 용도: Document AI 입력 예시.
포트폴리오 용도: Document AI 페이지 왼쪽 상단.

## Capture 2 - OCR → Structured Extraction + Quality

`02_standard_170dpi_jpeg_extraction.json`을 VS Code에서 연다.

화면에 다음 블록이 동시에 보이게 캡처한다.

- `expected`
- `ocr_quality.mean_confidence`
- `ocr_quality.low_confidence_fraction`
- `actual`
- `validation`

`ocr_text` 전체는 길면 접어도 된다.

이 화면의 목적은 “OCR을 호출했다”가 아니라 **ground truth와 실제 추출을 비교하고 quality signal까지 기록했다**는 것을 보여주는 것이다.

README 용도: structured extraction 증거.
포트폴리오 용도: 입력 이미지 옆 결과 카드.

## Capture 3 - Review-required 실패 사례

전체 benchmark 결과를 먼저 생성한다.

```bash
python scripts/evaluate_document_ai.py --n 60 --skip-checks \
  --output artifacts/document_ai_benchmark.json
```

그 뒤 sample index를 바꿔가며 실제 `validation.review_required=true` 사례 하나를 고른다.

```bash
python scripts/render_document_ai_samples.py \
  --output-dir artifacts/document_ai_capture_fail \
  --sample N
```

가장 추천하는 것은 degraded profile에서 실제 OCR 오독과 review가 함께 보이는 사례다.

캡처에는 다음이 보이게 한다.

- degraded scan 이미지
- `expected`
- `actual`
- `ocr_quality`
- `validation.review_required: true`
- `issues` (`low_ocr_mean_confidence`, `high_low_confidence_token_fraction`, missing/grounding issue 등 실제 발생 항목)

성공 사례보다 이 화면의 가치가 더 높다. **틀릴 수 있는 입력을 실제로 확인하고 review로 보낸 증거**이기 때문이다.

주의: 현 benchmark에서는 confidence gate가 모든 OCR 오독을 잡지 못한다. clean/standard의 high-confidence 오독은 상당수 통과한다. 캡처 설명도 “OCR 오류를 완전히 검출”이라고 쓰지 않는다.

## Capture 4 - 60-document × 3-profile benchmark

실행:

```bash
python scripts/evaluate_document_ai.py --n 60 --skip-checks
```

터미널 또는 `artifacts/document_ai_benchmark.json`에서 세 profile의 다음 값이 한 화면에 보이도록 캡처한다.

- `mean_request_char_accuracy`
- `field_f1_exact`
- `review_rate`
- `error_detection_recall`
- `mean_ocr_word_confidence`

필요하면 `mean_cer_no_space`와 `document_exact_match`도 포함한다.

현재 frozen headline:

| Profile | Request char acc. | Field F1 exact | Review | Error detection recall |
|---|---:|---:|---:|---:|
| Clean | 94.38% | 75.42% | 1.67% | 1.69% |
| Standard | 89.79% | 75.11% | 10.00% | 10.53% |
| Degraded | 93.53% | 62.44% | 58.33% | 58.33% |

`Request char accuracy`가 난이도에 따라 완벽히 단조 감소하지 않는 것도 실제 결과다. 캡처에서 숨기거나 숫자를 고치지 않는다.

README 용도: benchmark 근거.
포트폴리오 용도: 결과 카드/표.

## Capture 5 - Document AI → Evidence RAG 연결

먼저 `render_document_ai_samples.py`가 만든 입력 중 **validation을 통과하는** scan/image를 사용한다.

```bash
python scripts/document_ai.py \
  artifacts/document_ai_capture/02_standard_170dpi_jpeg.jpg \
  --rag
```

만약 해당 sample의 standard profile이 review로 막히면 sample index를 바꿔 validation-passing 사례를 선택한다.

캡처에는 다음 흐름이 보이면 된다.

- `document.mode: ocr`
- OCR engine / confidence
- extracted `serial`, `request`
- `validation.valid: true`
- RAG `evidence_count`
- evidence의 `source`, `serial`, `score`

`--rag-memo`는 Anthropic API를 실제 호출하므로 포트폴리오 캡처 목적만으로 실행하지 않아도 된다.

## Capture 6 - GitHub Actions CI

GitHub 저장소 → Actions → 최종 **main** CI run을 연다.

캡처할 항목:

- Python 3.9 green: core 563 + Evidence RAG
- Python 3.11 green: core 563 + Evidence RAG + Document AI
- `Document AI realistic synthetic scan 평가`
- 가능하면 `17 passed`가 보이는 dedicated check 로그

가능하면 두 job이 한 화면에 보이게 캡처한다. 세부 수치 JSON은 Capture 4에 맡기고, CI 캡처는 **재현성/회귀 검증 증거**로 쓴다.

## 선택 Capture 7 - 실제 코드 계약

포트폴리오 공간이 남을 때만 사용한다.

추천 코드:

- `app/document_ai/validation.py`: preregistered OCR confidence + grounding validation
- `app/document_ai/rag_bridge.py`: validation 통과 후에만 RAG 전달

코드 전체가 아니라 핵심 15~25줄만 캡처한다.

## 최종 권장 세트

포트폴리오에는 우선 **4장**만 사용한다.

1. Clean / Standard / Degraded 입력 3단계
2. OCR → Structured Extraction + OCR quality JSON
3. `review_required=true` 실제 실패 사례
4. 최종 GitHub Actions green

그리고 텍스트 표로 60×3 benchmark 수치를 넣는다. 포트폴리오 페이지가 답답하면 benchmark terminal 캡처는 README에만 둔다.

README에는 추가로 다음 두 장을 넣을 수 있다.

5. benchmark JSON/terminal 결과
6. Document AI → Evidence RAG 실제 출력

## README 삽입 위치

README `6. Architecture > OCR-aware Document AI intake` 아래에 다음 HTML comment가 미리 남아 있다.

- `CAPTURE_DOC_AI_INPUTS`
- `CAPTURE_DOC_AI_EXTRACTION`
- `CAPTURE_DOC_AI_FAILURE`

캡처 파일을 준비한 뒤 해당 comment 바로 아래에 Markdown image를 추가하면 된다. 지금은 깨진 이미지 링크를 만들지 않기 위해 placeholder comment만 둔다.

## 캡처 후

캡처한 PNG/JPG를 이 대화에 올리면 최종 포트폴리오 v4의 실제 자리로 교체한다. 이미지가 오기 전에는 포트폴리오에 가짜 실행 화면이나 생성 이미지를 넣지 않는다.
