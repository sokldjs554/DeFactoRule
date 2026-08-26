# Document AI / OCR intake - final scoped extension

## 1. Goal

DeFactoRule originally assumed that source PDFs expose a usable text layer. This extension adds an **OCR-aware Document AI intake layer** in front of the frozen Evidence RAG / Agent workflow so image-only or text-poor inputs can be converted into structured, reviewable records.

This work does **not** change the frozen 168-row Router/S5 decision profile or the Evidence RAG retrieval audit. It is an optional upstream ingestion layer.

The goal is not to claim that OCR is solved. The goal is to make document-ingestion quality observable, route obvious low-quality cases to review, and keep ingestion metrics separate from retrieval/decision metrics.

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
  ├─ serial format / required fields / decision enum
  ├─ exact quote grounding inside native/OCR text
  └─ preregistered OCR confidence gate
        ↓
  validated → downstream Evidence RAG
  review_required → human review, RAG not called
```

The OCR engine is behind an adapter boundary. Tesseract is a reproducible baseline, not a claim that production must use Tesseract.

`process_document_with_rag` is the explicit bridge: if document validation requires review, the RAG function is not invoked. A validated extraction supplies only the extracted request and serial to temporal Evidence RAG.

## 3. Validation contract and what it can actually prove

The structural validation layer checks:

- `serial`, `sector`, `decision`, `request` presence and schema constraints,
- source quote for each non-null field,
- quote existence inside the native/OCR text after whitespace normalization,
- Tesseract word-confidence signals for OCR inputs.

The OCR-quality policy was fixed **before observing the post-gate benchmark**:

- review if mean word confidence `< 80.0`, or
- review if more than `20%` of OCR tokens have confidence `< 60.0`.

The optional LLM extractor is instructed to return `null` rather than infer unsupported values and cannot make the downstream legal/precedent decision.

A critical distinction emerged during evaluation: **quote grounding against OCR text does not prove that OCR text itself matches the ground-truth document.** If Tesseract consistently misreads a character, the wrong string can still be perfectly grounded inside the OCR output. The confidence gate catches some degraded cases, but it is not a complete detector for transcription error.

Therefore the correct contract is:

> structural / grounding / low-confidence failures are routed to review; accepted OCR output is not claimed to be ground-truth perfect.

## 4. Realistic synthetic scanned-document benchmark

### Construction

Source: `data/eval/nonaction_test_clean.jsonl`.

Rather than cherry-picking the first easy rows, 60 eligible rows are **deterministically spread across the clean-test pool**. Eligibility is fixed as:

- `masked_leaks=0`,
- serial present,
- request length at most 700 characters.

Each selected real financial-regulation request is rendered as a one-page synthetic form containing:

- serial,
- sector,
- decision label,
- request text.

Three scan profiles are evaluated:

1. `clean_220dpi_png` - 220 dpi PNG, 11.5 pt, no skew.
2. `standard_170dpi_jpeg` - 170 dpi JPEG quality 70, 10.5 pt, gray text, 0.6° skew.
3. `degraded_120dpi_jpeg` - 120 dpi JPEG quality 40, 9.5 pt, lower contrast, 1.5° skew.

This is explicitly a **synthetic rasterization/degradation benchmark**. It is not a dataset of real customer scans, camera photos, heterogeneous scanner hardware, or production documents.

### Metrics

Headline metrics deliberately include the long `request` field rather than only easy scalar fields.

- `mean_cer_no_space`: full-page character error rate after whitespace removal.
- `mean_request_char_accuracy`: character accuracy of the extracted request.
- `field_f1_exact`: exact-match field F1 across serial / sector / decision / request.
- `document_exact_match`: fraction where all four fields are exact simultaneously.
- `review_rate`: fraction routed to human review by the structural / grounding / confidence contract.
- `error_detection_recall`: among documents with at least one ground-truth field mismatch, fraction routed to review.

`document_exact_match` is intentionally strict: one wrong character in the long request makes the whole document non-exact.

## 5. Frozen post-gate result

Tesseract `5.3.4`, language `kor`, `n=60` per profile, no LLM/API calls.

| Profile | Request char acc. | Field F1 exact | Document exact | Review rate | Error-detection recall |
|---|---:|---:|---:|---:|---:|
| clean 220dpi PNG | **94.38%** | **75.42%** | **1.67%** | **1.67%** | **1.69%** |
| standard 170dpi JPEG | **89.79%** | **75.11%** | **5.00%** | **10.00%** | **10.53%** |
| degraded 120dpi JPEG | **93.53%** | **62.44%** | **0.00%** | **58.33%** | **58.33%** |

Additional diagnostics:

| Profile | Full CER | Mean OCR word confidence | Mean low-conf-token fraction | Auto-accept |
|---|---:|---:|---:|---:|
| clean | 4.99% | 88.26 | 4.90% | 98.33% |
| standard | 5.32% | 88.06 | 5.77% | 90.00% |
| degraded | 6.36% | 86.85 | 7.06% | 41.67% |

Raw frozen artifact: `experiments/results/clean/document_ai_ocr.json`.

The non-monotonic request-character result is retained as measured: the degraded profile happened to produce a lower mean request CER than the standard profile even though its overall field F1 was substantially worse. No post-hoc metric editing or profile retuning was performed.

## 6. What the experiment changed

The first 60-document evaluation exposed an important blind spot: structural quote grounding alone auto-accepted most documents even when the exact ground-truth request differed by OCR characters.

The confidence thresholds above were then declared before the next evaluation and encoded as regression-tested constants. The post-gate evaluation showed:

- degraded scans are often rejected (`35/60` review, 58.33% error-detection recall),
- clean and standard OCR errors are frequently high-confidence errors and therefore remain difficult to detect (`1.69%` and `10.53%` error-detection recall),
- confidence is useful as one quality signal but is **not a substitute for an independent OCR/VLM verification channel**.

The thresholds were not retuned after observing these results.

This is the intended engineering conclusion, not a failed headline to hide: a production design would need stronger cross-checking, such as a second recognizer, image-aware VLM verification, or domain-specific field validation, before treating high-confidence OCR as trusted text.

## 7. Optional LLM extraction

`extract_fields_llm` reuses the project's Anthropic structured-output boundary. Its schema returns only document fields and source quotes; it does not return a final legal verdict or precedent-applicability decision.

This path is implemented and tested with a fake client for schema/contract behavior, but **live LLM extraction quality is not benchmarked in this phase**. Therefore the benchmark above is an OCR + deterministic extraction baseline only.

No API calls were made to generate the frozen OCR benchmark.

## 8. Integration with Evidence RAG

```text
Document AI intake
  ↓
structural / OCR-quality validation
  ├─ review_required → stop / human review
  └─ validated
       ↓
extracted request + serial
       ↓
Temporal-eligible Evidence RAG
       ↓
optional LLM evidence / deciding-factor analysis
       ↓
deterministic grounding / safety gates
       ↓
decision, abstain, or human handoff
```

The existing RAG and decision metrics remain separate so document-ingestion quality cannot be confused with retrieval or decision accuracy.

Because the benchmark shows that some OCR transcription errors survive the current confidence gate, `validated` here means **passed the implemented validation contract**, not “verified identical to ground truth.”

## 9. Reproduction

Install Tesseract with Korean language data, then:

```bash
python scripts/evaluate_document_ai.py --n 60
```

By default this first runs the dedicated `checks/document_ai` suite and then executes the 60-document, three-profile benchmark. Use `--skip-checks` only when intentionally running the benchmark without the contract checks.

Generate reproducible screenshots/examples without an LLM call:

```bash
python scripts/render_document_ai_samples.py \
  --output-dir artifacts/document_ai_capture \
  --sample 12
```

For one document:

```bash
python scripts/document_ai.py path/to/document.pdf
```

Add `--rag` to forward only a validation-passing extraction into Evidence RAG. `--llm` and `--rag-memo` are explicit opt-in Anthropic paths.

## 10. Limitations

- Synthetic scans, `n=60` per profile; no claim of real customer scan performance.
- Rasterization is generated from existing text using PyMuPDF, not photographs or heterogeneous scanner hardware.
- Tesseract is a pretrained baseline; no OCR detector/recognizer training or fine-tuning was performed.
- Exact field F1 is dominated partly by the strict long-request exact-match criterion; character accuracy is reported alongside it.
- OCR word confidence has weak discrimination for many high-confidence transcription errors in clean/standard scans.
- No table-structure recognition benchmark.
- No image-VLM benchmark; optional LLM extraction currently consumes OCR/native text.
- Live LLM structured-extraction quality and API cost are not measured here.
- No customer documents, customer-domain review, production traffic, SLA, or monitoring evidence.

The correct portfolio claim is therefore:

> **Implemented and evaluated an OCR-aware Document AI intake pipeline on a 60-document, three-profile synthetic financial scan benchmark; measured exact field quality and identified that OCR-confidence gating catches degraded inputs but is insufficient for many high-confidence transcription errors.**

It is not correct to claim “production OCR solved” or “all accepted OCR is ground-truth correct.”
