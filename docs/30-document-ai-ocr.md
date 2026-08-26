# Document AI / OCR intake — final scoped extension

## 1. Goal

DeFactoRule originally assumed that source PDFs expose a usable text layer. This extension adds an **OCR-aware Document AI intake layer** in front of the frozen Evidence RAG / Agent workflow so image-only or text-poor inputs can be converted into structured, reviewable records.

This work does **not** change the frozen 168-row Router/S5 decision profile or the Evidence RAG retrieval audit. It is an optional upstream ingestion layer.

## 2. Architecture

```text
PDF / scan / image
        ↓
Document intake
  ├─ healthy native PDF text → PyMuPDF
  └─ image / text-poor PDF   → OCR adapter
                               └─ baseline: Tesseract 5.x + kor
        ↓
Structured extraction
  ├─ deterministic casebook-compatible parser (default)
  └─ optional Anthropic structured extraction
        ↓
serial / sector / decision / request + source quotes
        ↓
Deterministic validation
  ├─ serial format
  ├─ decision enum
  ├─ required-field presence
  └─ exact source-quote grounding
        ↓
  valid → accept / downstream Evidence RAG
  invalid → review_required
```

The OCR engine is behind an adapter boundary. Tesseract is a reproducible baseline, not a claim that production must use Tesseract.

## 3. Safety contract

OCR output is never treated as trusted structured data by itself.

- `serial`, `sector`, `decision`, `request` are the structured fields.
- Every non-null field must have a source quote.
- The quote must be present in the native/OCR source text after whitespace normalization.
- Invalid or missing serial, missing sector/request, or an unsupported decision label causes `review_required`.
- An ungrounded quote also causes `review_required`.
- The optional LLM extractor is instructed to return `null` rather than infer unsupported values and cannot make the downstream legal/precedent decision.

Thus the failure mode for extraction uncertainty is **review**, not silent acceptance.

## 4. Synthetic scanned-document benchmark

### Construction

Source: `data/eval/nonaction_test_clean.jsonl`.

Selection is deterministic: the first 20 rows that satisfy all of the following:

- `masked_leaks=0`
- serial is present
- request length is at most 700 characters

For each selected real financial-regulation request, a one-page form is generated with:

- serial
- sector
- decision label
- request text

The form is then rasterized into two profiles:

1. `clean_220dpi_png`
2. `degraded_120dpi_jpeg` (JPEG quality 45)

This is explicitly a **synthetic rasterization/degradation benchmark**. It is not a dataset of real customer scans, camera photos, or production documents.

### Metrics

- `mean_cer_no_space`: character error rate after whitespace removal over the full form.
- `mean_request_cer_no_space`: CER over the extracted request field.
- `scalar_field_exact`: exact-match rate over serial / sector / decision.
- `fully_valid`: documents whose structured extraction passes all deterministic checks.
- `review_required`: documents routed to review by the validation contract.

### Frozen baseline

Tesseract `5.3.4`, language `kor`, 20 documents per profile, no LLM/API calls.

| Profile | Full CER | Request CER | Scalar exact | Fully valid | Review required |
|---|---:|---:|---:|---:|---:|
| clean 220dpi PNG | **4.12%** | **4.51%** | **100.00%** | **20/20** | **0/20** |
| degraded 120dpi JPEG | **8.72%** | **18.81%** | **81.67%** | **14/20** | **6/20** |

Raw frozen artifact: `experiments/results/clean/document_ai_ocr.json`.

The degraded profile is intentionally not tuned after observing the result. The important operational behavior is that six degraded documents fail the structured validation contract and are routed to review rather than silently accepted.

## 5. Optional LLM extraction

`extract_fields_llm` reuses the project's Anthropic structured-output boundary. Its schema returns only document fields and source quotes; it does not return a final legal verdict or precedent-applicability decision.

This path is implemented and tested with a fake client for schema/contract behavior, but **live LLM extraction quality is not benchmarked in this phase**. Therefore the benchmark above is an OCR + deterministic extraction baseline only.

## 6. Integration with Evidence RAG

The intended service flow is:

```text
Document AI intake
  ↓
validated structured request
  ↓
Temporal-eligible Evidence RAG
  ↓
LLM evidence / deciding-factor analysis when needed
  ↓
deterministic grounding / safety gates
  ↓
decision, abstain, or human handoff
```

The existing RAG and decision metrics remain separate so document-ingestion quality cannot be confused with retrieval or decision accuracy.

## 7. Reproduction

Install Tesseract with Korean language data, then:

```bash
python scripts/evaluate_document_ai.py --n 20
```

By default this first runs the dedicated `checks/document_ai` suite and then executes the 20-document benchmark. Use `--skip-checks` only when intentionally running the benchmark without the contract checks.

For one document:

```bash
python scripts/document_ai.py path/to/document.pdf
```

`--llm` opts into Anthropic structured field extraction; the default makes no LLM call.

## 8. Limitations

- Synthetic scans, `n=20`; no claim of real customer scan performance.
- Rasterization is generated from existing text using PyMuPDF, not photographs or heterogeneous scanner hardware.
- Tesseract is a pretrained baseline; no OCR detector/recognizer training or fine-tuning was performed.
- No table-structure recognition benchmark.
- No image-VLM benchmark; optional LLM extraction currently consumes OCR/native text.
- Live LLM structured-extraction quality and API cost are not measured here.
- No customer documents, customer-domain review, production traffic, SLA, or monitoring evidence.

The correct portfolio claim is therefore **“implemented and evaluated an OCR-aware Document AI intake pipeline with fail-closed structured validation on a synthetic financial-document scan benchmark”**, not “production OCR solved.”
