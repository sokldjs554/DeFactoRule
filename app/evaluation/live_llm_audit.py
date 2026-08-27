"""제출 전 live LLM contract audit — 계약 검증이지 성능 벤치마크가 아니다.

## 무엇을 검증하는가

frozen된 168행 decision profile(76/92/63/13)은 **건드리지 않는다.** 여기서 묻는
것은 하나다 — 현재 구현된 두 LLM 경로가 실제 Claude 호출에서도 선언한 계약대로
동작하는가.

    S5 arm   deciding-factor 구조 추출 -> 결정론적 Diff Coverage Gate
             (LLM은 verdict/applies/differs를 출력할 수 없다)
    RAG arm  evidence memo 생성 -> citation ID + exact quote 결정론 검증
             (검증 실패 memo는 fail-closed로 abstain 처리된다)

측정하는 것은 계약 준수율이다: API 성공, 스키마 통과, factor literal grounding,
citation ID 유효성, exact quote grounding, 그리고 **validator가 실제 Claude
오류를 차단한 사례**. 성능 지표(정확도 등)는 계산하지 않는다 — 이 audit의
결과를 보고 threshold/prompt/Router를 튜닝하지 않는다.

## 표본 선정 — cherry-picking이 불가능한 규칙

    S5 arm   frozen temporal Router에서 기권했고 top-1 선례가 similarity floor
             이상인 행 **전체**. 부분집합이 아니라 모집단이므로 고를 여지가 없다.
    RAG arm  T-serial + floor에서 evidence가 1건 이상 검색되는 행을 row key로
             정렬해 **stride 9로 기계적 추출** (index 0, 9, 18, ...).

seed 난수를 쓰지 않는다. 같은 데이터에서 항상 같은 표본이 나온다.

## 호출 규율

기본 실행은 dry-run이고 API를 부르지 않는다. `--go`에서만 호출하며, 상한은
선정된 표본 수와 정확히 같다. 재시도 0, 대체 모델 0. 각 호출 직후 checkpoint를
저장해 후처리 실패가 과금된 결과를 잃게 하지 않는다. 호출 전과 후에 frozen
profile을 재계산해 **한 건이라도 달라지면 실패로 처리한다** — 이 audit은
읽기 전용이어야 한다.
"""

from __future__ import annotations

import json
from collections import Counter

from app.agents.applicability import opposing_evidence
from app.agents.deciding_factor import Factor, evaluate_diff_coverage
from app.agents.deciding_factor_prompt import MAX_TOKENS as S5_MAX_TOKENS
from app.agents.deciding_factor_prompt import SYSTEM as S5_SYSTEM
from app.agents.deciding_factor_prompt import build_prompt as s5_prompt
from app.agents.deciding_factor_prompt import schema as s5_schema
from app.agents.workflow import VARIANTS, Workflow
from app.core.io import key_of, load_jsonl
from app.core.paths import EVAL, PROCESSED, RESULTS
from app.domain.similarity import DOUBT
from app.retrieval.lexical import LexicalRetriever

FROZEN_PROFILE = {"n": 168, "answered": 76, "abstained": 92, "correct": 63, "wrong": 13}
RAG_STRIDE = 9          # 88건 -> 10건. 표본 규모(총 20~30) 안에 들어오는 최소 보폭
OUT = RESULTS / "clean" / "live_llm_audit.json"
CHECKPOINT = RESULTS / "clean" / "live_llm_audit.checkpoint.json"

SELECTION_RULE = (
    "S5: frozen temporal Router 기권 & precedent_score >= DOUBT 인 행 전체(모집단). "
    f"RAG: T-serial+floor evidence >= 1 인 행을 row key 정렬 후 stride {RAG_STRIDE} 추출. "
    "난수/seed 없음."
)


def build_world():
    """final_freeze와 **같은 배선.** 다른 자산을 재면 다른 시스템을 감사하는 것이다."""
    dev = [r for r in load_jsonl(EVAL / "nonaction_dev_clean.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test_clean.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [
        c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
        for c in cases
    ]
    corpus = [text for text in corpus if text]
    rules = json.loads(
        (RESULTS / "clean" / "e6_rules_clean_runtime.json").read_text(encoding="utf-8")
    )["rules"]
    risk = json.loads(
        (RESULTS / "trap_risk_clean_temporal.json").read_text(encoding="utf-8")
    )
    fallback = Counter(r["label"] for r in dev).most_common(1)[0][0]
    flow = Workflow(
        LexicalRetriever().fit(dev, corpus), dev, rules, risk,
        policy=VARIANTS["router-temporal"], fallback=fallback,
    )
    by_key = {(r["source"], str(r["serial"])): r for r in dev}
    return flow, test, by_key


def profile_of(test: list[dict], states: list) -> dict:
    answered = [(r, s) for r, s in zip(test, states) if not s.abstained]
    correct = sum(1 for r, s in answered if s.decision == r["label"])
    return {
        "n": len(test), "answered": len(answered),
        "abstained": len(test) - len(answered),
        "correct": correct, "wrong": len(answered) - correct,
    }


def stride_pick(items: list, stride: int) -> list:
    """index 0부터 stride 간격으로 뽑는다. 규칙이 코드에 있으므로 재현된다."""
    return [items[i] for i in range(0, len(items), stride)]


def select_samples(flow, test: list[dict], by_key: dict) -> tuple[list[dict], dict]:
    from app.rag.evidence import EvidenceRetriever

    states = [flow.run(row) for row in test]
    profile = profile_of(test, states)

    s5_samples = []
    for row, state in zip(test, states):
        if not (state.abstained and state.precedent_score >= DOUBT):
            continue
        top = next((e for e in state.retrieved_evidence if e.rank == 0), None)
        precedent = by_key.get((top.source, str(top.serial)), {}) if top else {}
        s5_samples.append({
            "arm": "s5", "row": row, "state": state, "top": top,
            "opposing": opposing_evidence(state),
            "precedent_request": precedent.get("request", ""),
            "prompt": s5_prompt(row["request"], precedent.get("request", "")),
        })

    retriever = EvidenceRetriever()
    rag_eligible = []
    for row in sorted(test, key=key_of):
        hits = retriever.retrieve(row["request"], request_serial=str(row["serial"]), k=5)
        if hits:
            rag_eligible.append({"arm": "rag", "row": row, "hits": hits})
    rag_samples = stride_pick(rag_eligible, RAG_STRIDE)

    meta = {
        "selection_rule": SELECTION_RULE,
        "profile_before": profile,
        "s5_population": len(s5_samples),
        "rag_eligible_population": len(rag_eligible),
        "rag_stride": RAG_STRIDE,
        "rag_selected": len(rag_samples),
    }
    return s5_samples + rag_samples, meta


# ── 기록 ─────────────────────────────────────────────────────────────
def _s5_record(item: dict, raw: dict, latency: float) -> dict:
    """S5 한 건. LLM 출력은 gate에 넣어 보고, downstream은 frozen 정책 그대로다 —
    S5는 fail-closed safety veto로만 채택됐으므로 어떤 결과든 기권은 유지된다."""
    from app.infrastructure.anthropic_client import MODEL

    row, state, top = item["row"], item["state"], item["top"]
    data = raw.get("data")
    record = {
        "arm": "s5",
        "sample_id": str(row["serial"]),
        "row_key": list(key_of(row)),
        "route": state.route_reason,
        "retrieved_evidence_ids": [e.id for e in state.retrieved_evidence],
        "top_precedent": {"serial": str(top.serial), "label": top.label,
                          "score": round(top.score, 4)} if top else None,
        "opposing_count": len(item["opposing"]),
        "api_called": True,
        "api_success": "error" not in raw,
        "schema_valid": data is not None,
        "model": MODEL,
        "input_tokens": raw.get("input_tokens"),
        "output_tokens": raw.get("output_tokens"),
        "latency_s": round(latency, 3),
        "failure_reason": raw.get("error"),
        "n_factors": None,
        "n_factors_grounded": None,
        "n_factors_rejected": None,
        "gate_basis": None,
        "gate_fired_rule": None,
        "validator_blocked_recovery": None,
        "downstream": "abstain 유지 (S5는 safety veto로만 채택 — recovery 미채택)",
        "rejected_factor_texts": [],
        "model_output": data,
    }
    if data is None:
        return record

    shared = [_factor(x) for x in data.get("shared_factors", [])]
    differences = [
        _factor(x)
        for field in ("only_in_request", "only_in_precedent")
        for x in data.get(field, [])
    ]
    gate = evaluate_diff_coverage(
        row["request"], item["precedent_request"], shared, differences
    )
    all_factors = shared + differences
    by_id = {f.id: f for f in all_factors}
    record.update({
        "n_factors": len(all_factors),
        "n_factors_grounded": len(gate.grounded_shared_factor_ids)
        + len(gate.grounded_factor_ids),
        "n_factors_rejected": len(gate.rejected_factor_ids),
        "gate_basis": gate.basis,
        "gate_fired_rule": gate.fired_rule,
        # recovery 방향(applies 성격)이 gate에서 막혔는가 — fail-closed 확인
        "validator_blocked_recovery": gate.basis
        not in ("no_decisive_difference", "identical_after_metadata"),
        "rejected_factor_texts": [
            by_id[i].text[:120] for i in gate.rejected_factor_ids if i in by_id
        ],
    })
    return record


def _factor(item: dict) -> Factor:
    return Factor(
        id=str(item.get("id", "")), text=str(item.get("text", "")),
        side=item.get("side", "both"), axis=str(item.get("axis", "")),
        value_in_request=item.get("value_in_request"),
        value_in_precedent=item.get("value_in_precedent"),
        decisive=bool(item.get("decisive", False)),
        why_not_decisive=item.get("why_not_decisive"),
    )


def _rag_record(item: dict, response, latency: float) -> dict:
    from app.infrastructure.anthropic_client import MODEL

    row = item["row"]
    memo = response.memo
    validation = response.validation
    # memo도 validation도 없이 abstain이면 API/파싱 단계에서 실패한 것이다.
    # (검증 실패 abstain은 memo와 validation이 함께 남는다 — service 계약)
    api_error = None
    if memo is None and response.abstained and validation is None:
        api_error = response.abstain_reason
    claims = memo.claims if memo else []
    return {
        "arm": "rag",
        "sample_id": str(row["serial"]),
        "row_key": list(key_of(row)),
        "route": None,
        "retrieved_evidence_ids": [h.evidence_id for h in item["hits"]],
        "api_called": True,
        "api_success": api_error is None,
        "schema_valid": memo is not None,
        "model": MODEL,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_s": round(latency, 3),
        "failure_reason": api_error,
        "n_claims": len(claims),
        "invalid_citations": list(validation.invalid_citations) if validation else [],
        "ungrounded_quotes": list(validation.ungrounded_quotes) if validation else [],
        "citation_ids_valid": (not validation.invalid_citations) if validation else None,
        "quotes_grounded": (not validation.ungrounded_quotes) if validation else None,
        "validator_blocked": bool(
            validation and not validation.valid
        ),
        "downstream": (
            "usable memo" if validation and validation.valid
            else "abstain (fail-closed)" if memo is not None
            else "abstain (api/schema 실패)"
        ),
        "handoff_recommended": memo.handoff_recommended if memo else None,
        "memo": memo.model_dump() if memo else None,
    }


def aggregate(records: list[dict]) -> dict:
    """분모를 이름에 박는다. 100%가 나와도 n= 없이는 뜻이 없다."""
    s5 = [r for r in records if r["arm"] == "s5"]
    rag = [r for r in records if r["arm"] == "rag"]
    calls = len(records)
    ok = [r for r in records if r["api_success"]]
    schema_ok = [r for r in ok if r["schema_valid"]]

    s5_valid = [r for r in s5 if r["schema_valid"]]
    total_factors = sum(r["n_factors"] or 0 for r in s5_valid)
    grounded_factors = sum(r["n_factors_grounded"] or 0 for r in s5_valid)
    rejected_factors = sum(r["n_factors_rejected"] or 0 for r in s5_valid)

    rag_valid = [r for r in rag if r["schema_valid"]]
    total_claims = sum(r["n_claims"] for r in rag_valid)
    bad_citations = sum(len(r["invalid_citations"]) for r in rag_valid)
    bad_quotes = sum(len(r["ungrounded_quotes"]) for r in rag_valid)
    blocked = [r for r in records if r.get("validator_blocked")
               or (r["arm"] == "s5" and (r["n_factors_rejected"] or 0) > 0)]

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "api_calls_attempted": calls,
        "api_success": {"n": len(ok), "denominator": calls,
                        "rate": rate(len(ok), calls)},
        "schema_valid": {"n": len(schema_ok), "denominator": len(ok),
                         "rate": rate(len(schema_ok), len(ok))},
        "s5_factor_grounding": {
            "factors_total": total_factors,
            "factors_grounded": grounded_factors,
            "factors_rejected": rejected_factors,
            "denominator": f"factors emitted over n={len(s5_valid)} schema-valid S5 calls",
            "grounded_rate": rate(grounded_factors, total_factors),
        },
        "rag_citation_ids": {
            "claims_total": total_claims,
            "invalid": bad_citations,
            "denominator": f"claims over n={len(rag_valid)} schema-valid memos",
            "pass_rate": rate(total_claims - bad_citations, total_claims),
        },
        "rag_exact_quotes": {
            "claims_total": total_claims,
            "ungrounded": bad_quotes,
            "denominator": f"claims over n={len(rag_valid)} schema-valid memos",
            "pass_rate": rate(total_claims - bad_quotes, total_claims),
        },
        "validator_rejections": {
            "samples_with_any_rejection": len(blocked),
            "denominator": f"n={calls} audited samples",
            "rate": rate(len(blocked), calls),
        },
        "unsupported_output_found": {
            "s5_rejected_factors": rejected_factors,
            "rag_invalid_citations": bad_citations,
            "rag_ungrounded_quotes": bad_quotes,
            "total": rejected_factors + bad_citations + bad_quotes,
        },
    }


# ── 실행 ─────────────────────────────────────────────────────────────
def _load_checkpoint() -> list[dict]:
    if not CHECKPOINT.exists():
        return []
    payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise SystemExit(f"checkpoint 형식이 잘못됐습니다: {CHECKPOINT}")
    return records


def _save_checkpoint(records: list[dict]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps({"experiment": "live-llm-contract-audit", "records": records},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sample_key(record_or_item) -> tuple[str, str]:
    if isinstance(record_or_item, dict) and "plan" not in record_or_item:
        if "arm" in record_or_item and "sample_id" in record_or_item:
            return (record_or_item["arm"], record_or_item["sample_id"])
        return (record_or_item["arm"], str(record_or_item["row"]["serial"]))
    raise TypeError("sample key를 만들 수 없습니다")


def run_live(samples: list[dict], records: list[dict]) -> list[dict]:
    """미완료 표본만 부른다. 상한 = 표본 수. 재시도 없음."""
    import time as _time

    from app.infrastructure.anthropic_client import (
        FatalApiError,
        call_structured,
        connect,
    )
    from app.rag.schemas import RAGRequest
    from app.rag.service import run_rag

    done = {(_sample_key(r)) for r in records}
    pending = [s for s in samples if _sample_key(s) not in done]
    if not pending:
        print(f"이미 {len(records)}건 checkpoint 완료 — 추가 호출 0회")
        return records

    limit = len(samples)
    client = connect()
    print(f"checkpoint {len(records)}건 · 이번 실행 {len(pending)}건 · 상한 {limit}회")
    for item in pending:
        if len(records) >= limit:
            raise RuntimeError(f"호출 상한 {limit}회 초과 시도 — 중단")
        started = _time.perf_counter()
        try:
            if item["arm"] == "s5":
                raw = call_structured(client, S5_SYSTEM, item["prompt"],
                                      s5_schema(), S5_MAX_TOKENS)
                record = _s5_record(item, raw, _time.perf_counter() - started)
            else:
                row = item["row"]
                response = run_rag(
                    RAGRequest(request_text=row["request"],
                               request_serial=str(row["serial"]),
                               generate_memo=True),
                    client=client,
                )
                record = _rag_record(item, response, _time.perf_counter() - started)
        except FatalApiError as exc:
            print(f"  중단 — 계정 수준 오류: {exc}")
            break
        records.append(record)
        _save_checkpoint(records)
        mark = "✓" if record["api_success"] and record["schema_valid"] else "✗"
        print(f"  {mark} {record['arm']}:{record['sample_id']} · "
              f"{record.get('gate_basis') or record.get('downstream')} · "
              f"{record['latency_s']}s")
    return records


def main() -> None:
    import argparse

    from app.infrastructure.anthropic_client import MODEL, estimate_cost

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true",
                    help="실제로 호출한다. 없으면 선정과 검증만 하고 멈춘다.")
    args = ap.parse_args()

    flow, test, by_key = build_world()
    samples, meta = select_samples(flow, test, by_key)

    if meta["profile_before"] != FROZEN_PROFILE:
        raise SystemExit(
            f"frozen profile과 다릅니다 — audit을 시작하지 않습니다.\n"
            f"  기대 {FROZEN_PROFILE}\n  실측 {meta['profile_before']}"
        )

    print(f"live LLM contract audit — 표본 {len(samples)}건 "
          f"(S5 {meta['s5_population']} + RAG {meta['rag_selected']}"
          f"/{meta['rag_eligible_population']}) · 모델 {MODEL}")
    print(f"선정 규칙: {meta['selection_rule']}")
    print(f"frozen profile 일치 확인: {meta['profile_before']}")
    for item in samples:
        row = item["row"]
        if item["arm"] == "s5":
            top = item["top"]
            print(f"  s5  {row['serial']} · {item['state'].route_reason} · "
                  f"top {top.serial}/{top.label} {top.score:.3f} · "
                  f"opposing {len(item['opposing'])}")
        else:
            print(f"  rag {row['serial']} · evidence {len(item['hits'])}건")

    if not args.go:
        print("\ndry-run — API 0회. 실행하려면 --go 를 붙이세요.")
        return

    records = run_live(samples, _load_checkpoint())

    # 읽기 전용이었는지 — profile을 다시 계산해 확인한다.
    flow2, test2, _ = build_world()
    profile_after = profile_of(test2, [flow2.run(r) for r in test2])
    payload = {
        "experiment": "live-llm-contract-audit",
        "purpose": "LLM output contract + grounding/validation 경로의 live 검증. "
                   "성능 벤치마크가 아니며 frozen profile을 바꾸지 않는다.",
        "model": MODEL,
        "selection": meta,
        "frozen_profile": FROZEN_PROFILE,
        "profile_after_audit": profile_after,
        "profile_unchanged": profile_after == FROZEN_PROFILE,
        "api_calls": len(records),
        "retries": 0,
        "estimated_cost_usd": round(estimate_cost(records), 4),
        "aggregate": aggregate(records),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    agg = payload["aggregate"]
    print(f"\n호출 {len(records)}회 · 추정 비용 ${payload['estimated_cost_usd']:.4f}")
    print(f"API 성공 {agg['api_success']['n']}/{agg['api_success']['denominator']} · "
          f"스키마 통과 {agg['schema_valid']['n']}/{agg['schema_valid']['denominator']}")
    print(f"factor grounding {agg['s5_factor_grounding']['factors_grounded']}"
          f"/{agg['s5_factor_grounding']['factors_total']} · "
          f"인용 무결성 위반 {agg['unsupported_output_found']['total']}건")
    print(f"frozen profile 불변: {payload['profile_unchanged']}")
    if not payload["profile_unchanged"]:
        raise SystemExit("frozen profile이 변했습니다 — 원인 분석이 필요합니다.")
    print(f"-> {OUT}")
