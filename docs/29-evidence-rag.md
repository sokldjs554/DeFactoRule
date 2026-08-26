# 29. Optional Evidence RAG

## 목적

DeFactoRule의 최종 clean Router/S5 operating profile은 이미 168건 기준으로 freeze되어 있다. RAG를 추가하면서 기존 지표를 다시 튜닝하거나 의사결정 경로를 바꾸면, 새 기능이 과거 결과의 의미를 뒤섞을 수 있다.

따라서 Evidence RAG는 **기존 decision Agent와 분리된 optional service layer**로 구현했다. 역할은 새 업무 요청에 대해 과거 선례 evidence를 검색하고, 필요할 때만 LLM이 근거 메모를 구조화하도록 하는 것이다. 최종 라벨은 생성하지 않는다.

## Workflow

```text
request_text + request_serial
        ↓
T-serial eligible precedent subset
        ↓
Hybrid retrieval — lexical + dense, RRF
        ↓
calibrated similarity floor 0.15
        ↓
provenance-preserving evidence context
  evidence_id / source / page / serial / outcome / score
        ↓
optional Anthropic structured evidence memo
        ↓
deterministic citation-ID + exact-quote validation
        ↓
validated memo OR fail-closed abstention
```

## Retrieval contract

- precedent corpus: `data/eval/nonaction_dev_clean.jsonl`
- retriever: `H(L+D)` — lexical + dense Reciprocal Rank Fusion
- temporal policy: `serial`
- candidate filtering occurs **before ranking**
- hybrid output keeps the lexical similarity scale, so the existing calibrated `SIMILARITY_FLOOR=0.15` can be reused without a new test-set threshold search
- evidence below 0.15 is omitted from the LLM context
- missing/invalid serial under the serial policy returns no evidence rather than falling back to future-unsafe retrieval

The 0.15 floor was not tuned on the RAG clean test. It is inherited from the project's pre-existing similarity calibration.

## Provenance contract

Every retrieved item carries:

- `evidence_id`
- `source`
- `page`
- `serial`
- `pair_index`
- `sector`
- historical `outcome`
- retrieval `score`
- precedent request text

`evidence_id` is stable as `P-{serial}-{pair_index}` for a given precedent row.

Historical outcome is included as provenance metadata. It is **not** an instruction to copy the historical decision.

## Generation contract

`POST /rag/evidence` defaults to `generate_memo=false`. In this mode the endpoint performs deterministic retrieval only and makes **0 LLM/API calls**.

When `generate_memo=true`, the LLM is constrained to produce:

- `summary`
- `claims[]`
  - `claim`
  - `evidence_id`
  - `quote`
- `uncertainty`
- `handoff_recommended`

The prompt and schema do not ask for a final `조치/비조치/기타` decision.

After generation, deterministic validation checks:

1. every `evidence_id` exists in the retrieved context;
2. every `quote` is a non-empty exact substring of the cited precedent request.

A fabricated citation or ungrounded quote marks the response invalid and the service fails closed with abstention. Unit tests exercise both the grounded path and a hallucinated-quote path with a fake structured-output client; CI does not call the external LLM API.

## Service boundary

Standalone service:

```bash
uvicorn app.rag.app:app --reload
```

Endpoints:

```text
GET  /health
POST /rag/evidence
```

Keeping this FastAPI app separate prevents the new RAG feature from silently changing the frozen Router/S5 aggregate evaluation.

## Offline clean-test retrieval evaluation

Frozen artifact: `experiments/results/clean/rag_retrieval.json`

Conditions:

- clean test: 168
- `k=5`
- T-serial before ranking
- hybrid lexical+dense retrieval
- similarity floor: 0.15
- API calls: 0

| Metric | Result |
|---|---:|
| test queries | 168 |
| queries with floor-passing evidence | 88 |
| evidence availability | **52.38%** |
| zero-evidence queries | 80 |
| mean retained evidence count | 1.137 |
| median retained top-1 similarity | 0.331 |
| temporal violations | **0** |
| duplicate evidence-ID queries | **0** |

Two additional values are retained only as diagnostics:

- top-1 historical-outcome agreement: 70.45%
- top-k contains same historical outcome: 78.41%

These are **not retrieval relevance metrics**. There are no human precedent-relevance judgments for the corpus, so they must not be described as Recall@K, nDCG, or RAG accuracy.

The earlier no-floor diagnostic returned evidence for 167/168 queries, but that happened because Top-K filled the context even with weak similarities. That version was rejected as an operating contract. The final service keeps only evidence at or above the pre-existing 0.15 calibrated floor and therefore reports 88/168 availability.

## What is and is not validated

Validated:

- temporal candidate filtering before ranking
- lexical+dense hybrid retrieval wiring
- relevance floor enforcement
- provenance ID preservation
- deterministic citation/quote validation
- hallucinated quote → fail-closed behavior
- clean 168 offline retrieval artifact reproducibility
- Python 3.9 / 3.11 CI

Not validated:

- human-labeled precedent relevance
- live LLM memo quality over a representative evaluation set
- end-to-end customer production quality
- RAG-driven improvement to the frozen 168-row decision profile

Therefore the accurate portfolio claim is:

> **Temporal-eligible hybrid Evidence RAG with provenance-preserving context and deterministic citation/quote validation was implemented as an optional service layer. On the clean 168-query offline retrieval audit, 88 queries (52.38%) had evidence above the inherited 0.15 similarity floor, with 0 temporal violations. The generation path is contract-tested, but live LLM memo quality is not claimed.**
