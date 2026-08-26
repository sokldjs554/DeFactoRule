# Portfolio / README screenshot capture guide

이 문서는 실제 실행 화면을 직접 캡처하기 위한 가이드다. 저장소에는 임의의 목업 이미지를 커밋하지 않는다.

## 원칙

- 실제 실행 결과만 캡처한다.
- API key, 사용자 경로, 개인 정보가 화면에 나오지 않게 자른다.
- 숫자를 편집하거나 합성하지 않는다.
- synthetic scan은 반드시 `synthetic scanned-document benchmark`라고 표시한다.
- 한 화면에 너무 많은 로그를 넣지 말고, 증거가 되는 부분만 crop한다.

## 사전 준비

Ubuntu/Linux 기준:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-kor
pip install -r requirements-dev.txt
```

macOS에서는 Homebrew Tesseract에 한국어 traineddata가 설치되어 있는지 먼저 확인한다.

## Capture 1 — 입력 품질 3단계 비교

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
포트폴리오 용도: Document AI 페이지 왼쪽 1/2 영역.

## Capture 2 — OCR → Structured Extraction

`02_standard_170dpi_jpeg_extraction.json`을 VS Code에서 연다.

화면에 다음 블록이 동시에 보이게 캡처한다.

- `expected`
- `actual`
- `validation`

`ocr_text` 전체는 길면 접어도 된다.

README 용도: structured extraction 증거.
포트폴리오 용도: 입력 이미지 옆 결과 카드.

## Capture 3 — Fail-closed review 사례

60건 benchmark를 실행한다.

```bash
python scripts/evaluate_document_ai.py --n 60 --skip-checks \
  --output artifacts/document_ai_benchmark.json
```

먼저 aggregate 결과에서 `review_required`가 가장 많이 발생한 profile을 확인한다. 그 뒤 `scripts/render_document_ai_samples.py --sample N`의 N을 바꿔가며 실제 `validation.review_required=true` 사례 하나를 선택한다.

캡처에는 다음이 보이게 한다.

- degraded scan 이미지
- `actual` 추출값
- `validation.review_required: true`
- `issues`

포트폴리오에서 가장 중요한 실패 사례 캡처다. 성공 사례보다 이 화면의 가치가 더 높다.

## Capture 4 — 60-document benchmark 결과

실행:

```bash
python scripts/evaluate_document_ai.py --n 60 --skip-checks
```

터미널에서 세 profile의 다음 값이 보이도록 캡처한다.

- `mean_cer_no_space`
- `mean_request_char_accuracy`
- `field_f1_exact`
- `document_exact_match`
- `review_rate`
- `error_detection_recall`

100%인 보조 지표 하나만 확대하지 않는다. 세 난이도의 변화가 한 화면에서 보이게 하는 것이 목적이다.

README 용도: benchmark 근거.
포트폴리오 용도: 결과 페이지 또는 결과 카드.

## Capture 5 — Document AI → Evidence RAG 연결

검증된 scan/image 파일로 실행:

```bash
python scripts/document_ai.py \
  artifacts/document_ai_capture/02_standard_170dpi_jpeg.jpg \
  --rag
```

캡처에는 다음 흐름이 보이면 된다.

- OCR mode / engine
- extracted `serial`, `request`
- validation valid
- RAG `evidence_count`
- evidence의 `source`, `serial`, `score`

`--rag-memo`는 Anthropic API를 실제 호출하므로 포트폴리오 캡처 목적만으로 실행하지 않아도 된다.

## Capture 6 — GitHub Actions CI

GitHub 저장소 → Actions → 최종 main CI run을 연다.

캡처할 항목:

- Python 3.9 green
- Python 3.11 green
- `테스트`
- `Evidence RAG 오프라인 평가`
- `Document AI realistic synthetic scan 평가`

가능하면 두 job이 한 화면에 보이게 캡처한다.

## 선택 Capture 7 — 실제 코드 계약

포트폴리오 공간이 남을 때만 사용한다.

추천 코드:

- `app/document_ai/validation.py`: fail-closed validation
- `app/document_ai/rag_bridge.py`: validation 통과 후에만 RAG 전달

코드 전체가 아니라 핵심 15~25줄만 캡처한다.

## 최종 권장 세트

포트폴리오에는 4장만 우선 사용한다.

1. 입력 품질 3단계 비교
2. OCR → Structured Extraction JSON
3. `review_required=true` 실패 사례
4. GitHub Actions green

README에는 여기에 benchmark terminal 결과와 Document AI → Evidence RAG 결과를 추가해도 된다.

## 캡처 후

캡처한 PNG/JPG를 이 대화에 올리면 최종 포트폴리오 v4의 실제 자리로 교체한다. 이미지가 오기 전에는 포트폴리오에 가짜 실행 화면이나 생성 이미지를 넣지 않는다.
