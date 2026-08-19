"""회답 근거 구조화 — 당국이 실제로 무엇을 보고 판단했는가.

## 왜 이제 회답을 보는가

세 갈래가 같은 곳을 가리켰다. `조치` 여부를 가르는 신호가 **요청문 표면에 없다**.

    E5 검색   조치 14건 중 닮은 선례가 있는 것 1건 (7.1%)
    E6 규칙   조치 규칙 dev 100% -> test 20%
    E1 LLM    조치 재현율 0.286

요청문에 없는 것을 요청문에서 찾은 것이 여기까지의 한계였다. 그런데 **회답에는
당국이 왜 그렇게 판단했는지가 적혀 있다.** 그것을 구조화하면 "문서에 공표된 적
없는 판단 기준" 의 목록이 나온다 — 이 프로젝트가 처음부터 찾으려던 것이다.

## 순환을 어떻게 막는가

회답에는 결론도 함께 적혀 있다. 회답을 넣고 결론을 맞히면 100%가 나오고
아무것도 배우지 못한다. 그래서 다음 규율을 코드로 강제한다.

    추출   dev 의 회답에서만 기준을 꺼낸다
    적용   test 의 **요청문에만** 기준을 적용한다 — test 회답은 열지 않는다
    집계   기준별 가중치는 dev 에서만 정한다
    검증   기준 질문에 결론 표현이 들어 있으면 **버린다**

마지막 항목이 핵심이다. "이 사안은 조치 대상인가?" 같은 질문은 기준이 아니라
결론을 되묻는 것이다. 사람이 눈으로 거를 일이 아니라 코드가 막아야 한다.

## 명세 §9 의 분리

    LLM        회답을 읽고 판단 기준을 뽑는다 / 요청문이 그 기준을 충족하는지 답한다
    결정론 코드 순환 검사, 인용 대조, 가중치 산출, 최종 라벨 결정, 채점

**최종 라벨은 모델이 정하지 않는다.** 모델은 기준별 예/아니오까지만 말하고,
그것을 라벨로 바꾸는 것은 dev 에서 정한 가중치의 산술이다.

    python scripts/extract_criteria.py --dev data/eval/nonaction_dev.jsonl \\
        --cases data/processed/cases_nonaction.jsonl \\
        --output data/interim/criteria_raw.jsonl --dry-run
"""

from __future__ import annotations

import re

from app.core.text import clean_for_prompt, normalize_for_match
from app.domain.labels import NON_ACTIONS

# 기준 질문이 **이 사안의 결론**을 되묻고 있으면 기준이 아니다.
#
# 처음에는 '제재'·'결론'·'불이익'·'판단 결과' 를 낱말 단위로 걸렀는데, 정당한
# 사실 질문 아홉 개 중 넷을 잘라 냈다. "요청 행위가 소비자에게 불이익을
# 초래하는가" 는 되묻기가 아니라 **진짜 규제 판단 기준**이고, "요청인이 제재
# 이력을 보유하고 있는가" 는 요청인에 관한 사실이다.
#
# 걸러야 할 것은 낱말이 아니라 **묻는 대상**이다. 이 사안의 처분이 무엇이냐를
# 물으면 순환이고, 사안의 성질을 물으면 기준이다.
CONCLUSION_WORDS = re.compile(
    r"("
    r"비조치"                                              # 결론 라벨 그 자체
    r"|(?:이|본|해당|동)\s*(?:사안|건|요청|행위)[^?]{0,24}조치\s*(?:대상|여부|가능)"
    r"|조치\s*(?:대상|가능|필요)\s*(?:인가|인지|일까|에\s*해당)"
    r"|의견서를?\s*(?:받|발급|표명)"
    r"|당국이?\s*(?:어떻게|무엇을?|어떤)\s*(?:판단|결정|회신|조치)"
    r"|(?:결론|처분|회신\s*내용)(?:은|이|을|를)?\s*(?:무엇|어떻|어떤)"
    r"|어떤\s*(?:결론|처분)"
    r"|판단\s*결과(?:는|가)\s*(?:무엇|어떻)"
    r")"
)

MAX_CRITERIA_PER_CASE = 4

EXTRACT_SYSTEM = """\
당신은 금융규제 비조치의견서의 **판단이유**를 읽고, 당국이 실제로 무엇을 보고
판단했는지를 뽑아내는 분석가입니다.

목표는 결론을 요약하는 것이 아닙니다. 결론에 이르게 한 **판단 기준**을 꺼내는
것입니다. 사례집 어디에도 "이런 경우에는 이렇게 한다" 고 적혀 있지 않으므로,
개별 사례의 서술에서 그 기준을 되짚어야 합니다.

각 기준은 다음을 만족해야 합니다.

1. **요청문만 보고 답할 수 있는 예/아니오 질문**으로 쓸 것.
   판단이유나 결론을 봐야만 답할 수 있는 질문은 쓸모가 없습니다. 이 기준은
   나중에 회답이 없는 새 사안에 적용될 것이기 때문입니다.

2. **결론을 되묻지 말 것.**
   "조치 대상인가", "비조치 의견을 받을 수 있는가" 같은 질문은 기준이 아닙니다.
   무엇을 보고 그렇게 판단했는지를 물어야 합니다.

3. **판단이유 원문에서 그대로 인용**할 것. 요약하거나 다듬지 마십시오.
   글자 하나까지 원문과 같아야 하며, 대조에 실패하면 그 기준은 버려집니다.

4. 기준이 뚜렷하지 않으면 **적게 쓰십시오.** 억지로 채우지 마십시오.
   빈 목록도 정당한 답입니다.

한 사례에서 최대 {n} 개까지 뽑습니다."""

EXTRACT_SYSTEM = EXTRACT_SYSTEM.format(n=MAX_CRITERIA_PER_CASE)

APPLY_SYSTEM = """\
당신은 금융규제 비조치의견서의 **요청 내용**을 읽고, 주어진 판단 기준 각각에
대해 그 요청이 해당하는지 답합니다.

당신은 당국의 결론을 알지 못하며, 추측해서도 안 됩니다. 오직 요청문에 적힌
내용만 보고 각 기준에 답하십시오.

각 기준에 대해 셋 중 하나로 답합니다.

    yes      요청문의 내용이 그 기준에 해당한다
    no       해당하지 않는다
    unknown  요청문만으로는 알 수 없다

**모르면 unknown 을 쓰십시오.** 억지로 yes 나 no 를 고르면 그 답은 근거가
없습니다. unknown 은 실패가 아니라 정직한 답입니다."""


def extract_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "criteria": {
                # 길이 제약(maxItems)은 여기 쓸 수 없다. API 가 구조화 출력
                # 스키마에서 배열 길이 제약을 거부한다(400). 상한은 프롬프트로
                # 요청하고 **결정론적 코드가 강제한다** — 명세 §9 대로 계약
                # 강제는 모델이 아니라 코드의 일이다.
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "기준의 짧은 이름 (20자 이내)",
                        },
                        "question": {
                            "type": "string",
                            "description": "요청문만 보고 답할 수 있는 예/아니오 질문",
                        },
                        "quote": {
                            "type": "string",
                            "description": "판단이유에서 그대로 인용한 근거 구절",
                        },
                        "implies": {
                            "type": "string",
                            "enum": list(NON_ACTIONS),
                            "description": "이 기준에 해당할 때 당국이 낸 결론",
                        },
                    },
                    "required": ["name", "question", "quote", "implies"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["criteria"],
        "additionalProperties": False,
    }


def apply_schema(n: int) -> dict:
    """기준 n 개에 대한 답 스키마. n 은 프롬프트에만 반영되고 스키마 길이
    제약으로는 쓰지 않는다 — 아래 주석 참고."""
    del n
    return {
        "type": "object",
        "properties": {
            "answers": {
                # 위와 같은 이유로 길이 제약을 쓰지 않는다. 빠진 id 는 아래에서
                # unknown 으로 메우므로 개수가 모자라도 정렬이 어긋나지 않는다.
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "answer": {"type": "string", "enum": ["yes", "no", "unknown"]},
                    },
                    "required": ["id", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


def quote_is_grounded(quote: str, source: str) -> bool:
    """인용이 판단이유 원문에 글자 그대로 있는가.

    공백과 조판 잔재는 무시한다. 판단이유 98.8%에 깨진 글머리 기호(U+2244)나
    ZWNJ 가 섞여 있어서, 그것까지 따지면 **정상적인 인용이 전부 버려진다.**
    글자 자체는 건드리지 않으므로 뜻이 바뀐 인용은 여전히 걸러진다.
    """
    if not quote.strip():
        return False
    return normalize_for_match(quote) in normalize_for_match(source)


def question_is_circular(question: str) -> bool:
    """기준 질문이 결론을 되묻고 있는가.

    사람이 눈으로 거를 일이 아니다. 요청문 누출을 세 번 겪고서 배운 것이다.
    """
    return bool(CONCLUSION_WORDS.search(question or ""))


def validate_criterion(item: dict, reasoning: str) -> list[str]:
    """버려야 할 이유를 전부 모아 돌려준다. 비어 있으면 채택."""
    problems = []
    if not (item.get("name") or "").strip():
        problems.append("이름 없음")
    question = (item.get("question") or "").strip()
    if not question:
        problems.append("질문 없음")
    elif question_is_circular(question):
        problems.append("질문이 결론을 되묻는다")
    if not quote_is_grounded(item.get("quote", ""), reasoning):
        problems.append("인용이 원문에 없다")
    if item.get("implies") not in NON_ACTIONS:
        problems.append(f"알 수 없는 결론: {item.get('implies')}")
    return problems


def build_extract_prompt(case: dict) -> str:
    fields = case.get("fields", {})
    return (
        f"[요청대상행위]\n{clean_for_prompt(fields.get('요청대상행위'))}\n\n"
        f"[판단이유]\n{clean_for_prompt(fields.get('판단이유'))}\n\n"
        f"[당국의 결론]\n{case.get('decision', '')}"
    )


# ── 출력 예산 ─────────────────────────────────────────────────────
# 추정과 상한이 **같은 자리에서** 나와야 한다. 한동안 비용 추정은
# `60 × 기준수` 로 계산하면서 max_tokens 는 1200 으로 못 박혀 있었다. 기준이
# 15개일 때는 둘 다 맞았고, 88개가 되자 상한만 모자랐다. 잘린 JSON 은
# 파싱 실패로 나타나고, 그때는 이미 돈을 쓴 뒤다 (IN-11).
def apply_output_tokens(n_criteria: int) -> int:
    """기준 n 개에 답하는 데 드는 출력 토큰의 **예상치**.

    답 하나가 `{"id": 87, "answer": "unknown"}` 로 31자, 대략 12토큰이다.

    한동안 여유 있게 60 으로 잡았다. "비용을 낮게 부르는 쪽으로 틀리지 않기
    위해서" 였는데, 기준이 88개가 되자 그 여유가 5배로 불어나 **$14 라고
    부르는 값의 실제가 $5 언저리**가 됐다. 그렇게 부풀린 추정은 안전한 것이
    아니라 판단을 망친다 — 쓸 만한 실험을 비싸 보인다는 이유로 접게 만든다.

    그래서 실측값으로 잡는다. 대신 **적응형 사고 토큰은 여기 들어 있지
    않다.** 그 몫은 미리 알 수 없으므로 범위로 따로 보여 준다.
    """
    return 13 * n_criteria + 200


def apply_token_cap(n_criteria: int) -> int:
    """호출에 걸 상한. 답이 예상의 두 배로 길어져도 담고, 사고 몫을 더한다.

    상한은 넉넉해야 한다. 넘겨 잡아도 쓰지 않은 토큰은 청구되지 않지만,
    모자라면 응답 전체를 잃는다. 여유를 **곱셈으로** 두는 것이 중요하다 —
    고정값으로 두면 기준 수가 늘 때 여유만 줄어든다. 그것이 IN-11 이었다.
    """
    return 2 * apply_output_tokens(n_criteria) + 2000


def build_apply_prompt(request: str, criteria: list[dict]) -> str:
    lines = [f"[요청대상행위]\n{clean_for_prompt(request)}", "", "[판단 기준]"]
    for i, c in enumerate(criteria):
        lines.append(f"{i}. {c['question']}")
    return "\n".join(lines)


# ══ CLI ═══════════════════════════════════════════════════════════
def _estimate(prompts: list[str], out_tokens: int) -> tuple[int, float]:
    """한글은 대략 글자당 1토큰으로 잡는다. 실행 전 규모를 가늠하는 용도다."""
    from app.infrastructure.anthropic_client import PRICE_IN, PRICE_OUT

    in_tokens = sum(len(p) for p in prompts) + 700 * len(prompts)  # 시스템 프롬프트 몫
    cost = in_tokens / 1e6 * PRICE_IN + out_tokens * len(prompts) / 1e6 * PRICE_OUT
    return in_tokens, cost


def criteria_fingerprint(criteria: list[dict]) -> str:
    """기준 목록의 지문. 답 파일이 **어떤 질문들에 대한 답인지** 를 붙들어 둔다.

    답은 `["yes", "no", ...]` 처럼 순서로만 기준과 이어진다. `consolidate` 를
    다시 돌려 기준이 바뀌면 그 순서가 다른 질문을 가리키는데, 어디서도 오류가
    나지 않는다. 실제로 재현해 봤다 — 기준 5개짜리 답을 3개짜리 모델로 채점하니
    조용히 다른 답이 나왔다.

    질문 문장 자체로 지문을 만든다. id 나 개수만으로는 "질문 하나가 다른 것으로
    바뀐" 경우를 못 잡는다.
    """
    import hashlib

    joined = "\n".join(c["question"] for c in criteria)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def check_fingerprint(records: list[dict], criteria: list[dict], where: str) -> None:
    """답 파일이 지금의 기준 목록에 대한 답이 맞는지 확인한다."""
    want = criteria_fingerprint(criteria)
    seen = {r.get("criteria_fingerprint") for r in records if "answers" in r}
    unknown = None in seen or "" in seen
    wrong = {f for f in seen if f and f != want}
    if wrong:
        raise SystemExit(
            f"{where}: 답이 다른 기준 목록에 대한 것입니다.\n"
            f"  지금 기준 지문 {want} · 답 파일 지문 {', '.join(sorted(wrong))}\n\n"
            "  consolidate 를 다시 돌렸다면 apply 도 다시 돌려야 합니다.\n"
            "  답은 순서로만 기준과 이어지므로, 섞이면 조용히 틀립니다."
        )
    if unknown:
        lengths = {len(r["answers"]) for r in records if "answers" in r}
        print(f"  ⚠ 지문이 없는 옛 형식 답이 있습니다 (기준 지문 {want}).")
        print(f"     답 길이 {sorted(lengths)} · 지금 기준 {len(criteria)}개 — "
              "길이만 대조했습니다.")
        if lengths - {len(criteria)}:
            raise SystemExit(
                "  답 길이가 기준 수와 다릅니다. apply 를 다시 돌리세요."
            )


def relevance_preview(rows: list[dict], criteria: list[dict], floor: float = 0.15) -> str:
    """요청문과 기준 질문이 **얼마나 겹치는가** 를 호출 없이 재본다.

    기준은 사례별 판단이유에서 뽑았으므로 그 사례의 특수 사정을 그대로 담을 수
    있다. "요청이 부동산PF 사업장 재구조화와 관련되는가" 같은 기준은 PF 사안
    바깥에서는 언제나 아니오다. 그런 기준만 남으면 대부분의 사례에서 아무것도
    발화하지 않고, 예측은 기저율로 주저앉는다.

    글자 4-gram IDF 코사인으로 사례마다 가장 닮은 기준을 찾는다. 문턱은 TRAP
    지표에서 쓰는 값과 같은 0.15 다. **이것은 대리 측정이다** — 모델은 낱말이
    겹치지 않아도 뜻으로 판단할 수 있다. 그러니 여기서 나온 수는 발화율의
    상한이 아니라 눈금이다. 다만 눈금이 바닥이면 돈을 쓸 이유가 없다.
    """
    from app.evaluation.confusable import cosine, idf_table, weighted_vector

    questions = [c["question"] for c in criteria]
    texts = questions + [r.get("request", "") for r in rows]
    idf = idf_table(texts)
    q_vecs = [weighted_vector(q, idf) for q in questions]

    best = []
    for row in rows:
        vec = weighted_vector(row.get("request", ""), idf)
        best.append(max((cosine(vec, q) for q in q_vecs), default=0.0))

    over = [b for b in best if b >= floor]
    ordered = sorted(best)
    mid = ordered[len(ordered) // 2] if ordered else 0.0
    lines = [
        f"  기준과 낱말이 겹치는 사례: {len(over)}/{len(rows)} "
        f"({len(over) / max(1, len(rows)):.1%}, 문턱 {floor})",
        f"  최대 닮음의 중앙값 {mid:.3f} · 최댓값 {max(best, default=0):.3f}",
    ]
    by_label: dict[str, list[float]] = {}
    for row, b in zip(rows, best):
        if row.get("label"):
            by_label.setdefault(row["label"], []).append(b)
    for label, vals in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        hit = sum(1 for v in vals if v >= floor)
        lines.append(f"    {label:<4} {len(vals):>3}건 · 겹침 {hit}건 "
                     f"({hit / len(vals):.0%}) · 중앙값 {sorted(vals)[len(vals) // 2]:.3f}")
    if not over:
        lines.append("  ⚠ 겹치는 사례가 없다. 기준이 이 사례들에 대해 말할 것이 없다는 뜻이다.")
    return "\n".join(lines)


def cmd_extract(args) -> None:
    from pathlib import Path

    from app.core.io import key_of, load_jsonl, write_jsonl
    from app.infrastructure.anthropic_client import (
        FatalApiError,
        call_structured,
        connect,
        estimate_cost,
        preflight,
    )

    dev_keys = {key_of(r) for r in load_jsonl(Path(args.dev)) if r.get("label")}
    cases = [
        c for c in load_jsonl(Path(args.cases))
        if c.get("decision")
        and (c["source"], c["page"], c["serial"], 1) in dev_keys
        and (c["fields"].get("판단이유") or "").strip()
    ]
    if args.limit:
        cases = cases[: args.limit]
    prompts = [build_extract_prompt(c) for c in cases]

    _, cost = _estimate(prompts, out_tokens=900)
    print(f"dev 사례 {len(cases)}건에서 판단 기준을 뽑습니다.")
    print(f"  추정 비용 약 ${cost:.2f} (사례당 ${cost / max(1, len(cases)):.3f})")
    if args.dry_run:
        print("\n--dry-run 이므로 요청을 보내지 않습니다. 첫 사례의 프롬프트:\n")
        print("─" * 70)
        print(prompts[0][:1200] + ("…" if len(prompts[0]) > 1200 else ""))
        print("─" * 70)
        return

    out = Path(args.output)
    done = set()
    existing = []
    if args.resume and out.exists():
        existing = load_jsonl(out)
        done = {(r["source"], r["page"], r["serial"]) for r in existing if "error" not in r}
        print(f"  이어하기: {len(done)}건은 건너뜁니다.")

    client = connect()
    preflight(client, extract_schema())

    from app.core.audit import Discards

    records = list(existing)
    kept = 0
    discards = Discards("criteria-extract")
    try:
        for i, (case, prompt) in enumerate(zip(cases, prompts), 1):
            if (case["source"], case["page"], case["serial"]) in done:
                continue
            result = call_structured(client, EXTRACT_SYSTEM, prompt, extract_schema())
            record = {
                "source": case["source"], "page": case["page"], "serial": case["serial"],
                "sector": case.get("sector"), "decision": case["decision"],
            }
            if "error" in result:
                record.update(result)
            else:
                reasoning = case["fields"].get("판단이유") or ""
                proposed = result["data"].get("criteria", [])
                # 버린 것은 공용 장치가 이유와 함께 붙든다. 화면에만 찍으면
                # 나중에 왜 하나도 안 남았는지 알 방법이 없다 — API 오류에서
                # 상태 코드만 남겼던 것과 같은 실수다 (app/core/audit.py).
                case_discards = Discards("criteria-extract")
                accepted = [
                    item for item in proposed
                    if case_discards.keep_if(item, validate_criterion(item, reasoning))
                ]
                if len(accepted) > MAX_CRITERIA_PER_CASE:
                    for extra in accepted[MAX_CRITERIA_PER_CASE:]:
                        case_discards.drop(extra, ["사례당 상한 초과"])
                    accepted = accepted[:MAX_CRITERIA_PER_CASE]
                kept += len(accepted)
                for x in case_discards.records():
                    discards.drop(
                        {k: v for k, v in x.items() if k != "rejected_for"},
                        x["rejected_for"],
                    )
                record.update({
                    "criteria": accepted,
                    "rejected": case_discards.records(),
                    "discards": case_discards.summary(),
                    "proposed": len(proposed),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                })
            records.append(record)
            write_jsonl(out, records)
            if i % 20 == 0:
                print(f"  {i}/{len(cases)}")
    except FatalApiError as exc:
        print(f"\n중단 — 계정 수준 오류입니다. 남은 요청은 보내지 않았습니다.\n  {exc}")
        print(f"  여기까지 {len(records)}건이 {out} 에 저장됐습니다. --resume 으로 이어가세요.")

    errors = [r for r in records if "error" in r]
    print(f"\n{len(records)}건 처리 · 실패 {len(errors)}")
    print(f"  기준 채택 {kept}")
    print(discards.report(prefix="  "))
    if kept == 0:
        print("\n  ⚠ 채택된 기준이 하나도 없습니다. 위의 '버린 이유' 를 보세요.")
        print("     버려진 기준 자체는 각 레코드의 rejected 에 남아 있습니다.")
    print(f"  실제 비용 ${estimate_cost(records):.3f}")
    print(f"-> {out}")


def load_criteria(path) -> list[dict]:
    """기준 목록을 읽고 형식을 확인한다.

    잘못된 파일을 주면 traceback 대신 무엇이 잘못됐는지 말한다. 이 단계에서
    쓰는 파일이 세 종류(사례별 원본·통합 목록·평가셋)라 헷갈리기 쉽다.
    """
    import sys
    from pathlib import Path

    from app.core.io import load_jsonl

    if not Path(path).exists():
        sys.exit(
            f"기준 파일이 없습니다: {path}\n\n"
            "  통합 단계를 먼저 돌리세요 (API 를 쓰지 않습니다):\n"
            "    python3 scripts/criteria.py consolidate \\\n"
            "        --input data/interim/criteria_raw.jsonl \\\n"
            f"        --output {path}\n\n"
            "  그 앞 단계(extract)부터 확인하려면:\n"
            "    python3 scripts/criteria.py status"
        )
    rows = load_jsonl(Path(path))
    if not rows:
        sys.exit(f"기준 파일이 비어 있습니다: {path}")
    missing = [i for i, r in enumerate(rows) if "question" not in r]
    if missing:
        sys.exit(
            f"기준 파일 형식이 아닙니다: {path}\n"
            f"  {len(missing)}개 줄에 'question' 이 없습니다.\n"
            "  `scripts/criteria.py consolidate` 의 출력을 넣으세요."
        )
    return rows


def class_floors(distribution: dict, min_support: int) -> dict:
    """클래스별 지지도 문턱을 **같은 비율**로 환산한다.

    일률 문턱은 클래스마다 요구하는 증거의 비율을 다르게 만든다. dev 가
    비조치 58 · 조치 8 일 때 "2건 이상" 은 비조치에 3.4%, 조치에 25% 를
    요구하는 것이다. 소수 클래스 기준은 그래서 구조적으로 살아남지 못한다.

        >>> class_floors({"비조치": 58, "기타": 19, "조치": 8}, 2)
        {'비조치': 2, '기타': 1, '조치': 1}

    최소는 1 이다 — 0 으로 내려가면 문턱이 아니게 된다.
    """
    if not distribution:
        return {}
    biggest = max(distribution.values())
    return {
        label: max(1, round(min_support * n / biggest))
        for label, n in distribution.items()
    }


def cmd_consolidate(args) -> None:
    """사례별 추출 결과를 하나의 기준 목록으로 합친다. API 를 쓰지 않는다.

    같은 기준이 여러 사례에서 조금씩 다른 말로 나온다. 질문 텍스트의 IDF 가중
    코사인으로 묶고, 가장 많이 반복된 표현을 대표로 삼는다. **여러 사례에서
    반복해 나온 기준일수록 믿을 만하다** — 한 사례에서만 나왔다면 그 사례의
    특수 사정일 수 있다.

    ## 문턱은 개수가 아니라 비율이어야 한다

    dev 는 비조치 58 · 기타 19 · 조치 8 이다. 여기에 "2건 이상" 을 똑같이
    걸면 클래스마다 요구하는 **증거의 비율**이 달라진다.

        비조치  2/58 = 3.4%
        조치    2/8  = 25.0%

    조치 쪽에만 일곱 배 엄한 잣대를 대는 것이고, 그러면 소수 클래스 기준은
    구조적으로 살아남지 못한다. 규칙 학습기에서 라플라스를 통과 문턱으로 쓸 때
    똑같은 일이 있었다(EV-16 주변). 그래서 클래스별로 **같은 비율**이 되도록
    문턱을 환산한다.

        문턱(라벨) = max(1, round(min_support × n(라벨) / n(최다 라벨)))

    무리의 문턱은 그 무리를 만든 사례들의 최빈 결론으로 정한다. 낮춰서 살아남은
    무리가 몇 개인지 반드시 출력한다 — 조용히 완화하면 그것은 완화가 아니라
    은폐다.
    """
    from collections import Counter
    from pathlib import Path

    from app.core.io import load_jsonl, write_jsonl
    from app.evaluation.confusable import cosine, idf_table, weighted_vector

    all_records = load_jsonl(Path(args.input))
    raw = [r for r in all_records if "error" not in r]
    items = []
    for record in raw:
        for c in record.get("criteria", []):
            items.append({**c, "source": record["source"], "serial": record["serial"],
                          "decision": record["decision"], "sector": record.get("sector")})

    if not items:
        errors = [r for r in all_records if "error" in r]
        proposed = sum(r.get("proposed", 0) for r in raw)
        rejected = [x for r in raw for x in r.get("rejected", [])]
        print(f"채택된 기준이 하나도 없습니다.\n\n진단 — {args.input}")
        print(f"  레코드 {len(all_records)}건 · API 실패 {len(errors)}건")
        print(f"  모델이 제안한 기준 {proposed}개 · 검증 통과 0개")

        if errors:
            kinds = Counter(r["error"] for r in errors)
            print("\n  API 실패 종류: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common(3)))
            first = errors[0]
            print(f"    상세: {(first.get('error_detail') or '(없음)')[:200]}")
            print("\n  --resume 을 붙여 실패분만 다시 부르세요.")
        elif proposed == 0:
            print("\n  모델이 기준을 하나도 제안하지 않았습니다. 프롬프트가 너무 엄격하거나,")
            print("  판단이유가 기준이라 할 만한 것을 담고 있지 않을 수 있습니다.")
        elif rejected:
            why = Counter(w for x in rejected for w in x["rejected_for"])
            print("\n  버린 이유:")
            for reason, n in why.most_common():
                print(f"    {reason}: {n}")
            print("\n  버려진 기준 예시 (앞 3개):")
            for x in rejected[:3]:
                print(f"    이유 {x['rejected_for']}")
                print(f"      질문: {x.get('question', '')[:70]}")
                print(f"      인용: {x.get('quote', '')[:70]}")
        else:
            print("\n  이 파일은 버린 기준을 기록하지 않은 옛 형식입니다.")
            print("  `extract` 를 다시 돌리면(--resume 없이) 이유가 함께 남습니다.")
            print("  이미 성공한 호출을 아끼려면 --resume 을 쓰되, 그러면 이유는")
            print("  새로 부른 건에 대해서만 남습니다.")
        raise SystemExit(1)

    idf = idf_table([c["question"] for c in items])
    vecs = [weighted_vector(c["question"], idf) for c in items]

    groups: list[list[int]] = []
    assigned = [-1] * len(items)
    for i in range(len(items)):
        if assigned[i] >= 0:
            continue
        gid = len(groups)
        groups.append([i])
        assigned[i] = gid
        for j in range(i + 1, len(items)):
            if assigned[j] < 0 and cosine(vecs[i], vecs[j]) >= args.threshold:
                assigned[j] = gid
                groups[gid].append(j)

    # 클래스별 문턱을 같은 비율로 환산한다
    dev_dist = Counter(r["decision"] for r in raw)
    floor = class_floors(dev_dist, args.min_support)

    merged = []
    relaxed = 0
    for members in groups:
        group = [items[m] for m in members]
        dominant = Counter(c["decision"] for c in group).most_common(1)[0][0]
        needed = floor.get(dominant, args.min_support)
        if len(group) < needed:
            continue
        if len(group) < args.min_support:
            relaxed += 1
        implied = Counter(c["implies"] for c in group)
        merged.append({
            "id": len(merged),
            "question": Counter(c["question"] for c in group).most_common(1)[0][0],
            "name": Counter(c["name"] for c in group).most_common(1)[0][0],
            "support": len(group),
            "sources": len({c["serial"] for c in group}),
            "implies": implied.most_common(1)[0][0],
            "implies_distribution": dict(implied),
            "observed_decisions": dict(Counter(c["decision"] for c in group)),
            "quotes": [c["quote"] for c in group[:3]],
        })
    merged.sort(key=lambda c: -c["support"])
    for i, c in enumerate(merged):
        c["id"] = i
    write_jsonl(Path(args.output), merged)

    print(f"사례별 기준 {len(items)}개 -> {len(groups)}무리 -> 채택 {len(merged)}개")
    print("  클래스별 문턱(같은 비율로 환산): "
          + " · ".join(f"{lab} {floor[lab]}건 (dev {dev_dist[lab]}건)"
                       for lab in sorted(floor, key=lambda x: -dev_dist[x])))
    if relaxed:
        print(f"  이 중 {relaxed}개는 소수 클래스 문턱으로 살아남았다 "
              f"(일률 {args.min_support}건이었다면 탈락)")
    print(f"\n{'#':>3}  {'지지':>4}  {'사례':>4}  {'가리키는 결론':>12}  질문")
    for c in merged[: args.top]:
        print(f"{c['id']:>3}  {c['support']:>4}  {c['sources']:>4}  "
              f"{c['implies']:>12}  {c['question'][:56]}")
    print(f"\n-> {args.output}")


def cmd_apply(args) -> None:
    from pathlib import Path

    from app.core.io import load_jsonl, write_jsonl
    from app.infrastructure.anthropic_client import (
        FatalApiError,
        call_structured,
        connect,
        estimate_cost,
        preflight,
    )

    criteria = load_criteria(args.criteria)
    rows = [r for r in load_jsonl(Path(args.gold)) if (r.get("request") or "").strip()]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [build_apply_prompt(r["request"], criteria) for r in rows]

    n = len(criteria)
    _, low = _estimate(prompts, out_tokens=apply_output_tokens(n))
    _, high = _estimate(prompts, out_tokens=apply_token_cap(n))
    print(f"기준 {n}개를 사례 {len(rows)}건에 적용합니다.")
    print(f"  추정 비용 ${low:.2f} ~ ${high:.2f}")
    print("    아래쪽은 답 JSON 만 셌을 때, 위쪽은 상한까지 다 썼을 때입니다.")
    print("    적응형 사고 토큰은 미리 알 수 없어 이 범위로 갈음합니다.")
    print(f"    --limit 5 로 먼저 돌리면 실제 비용이 출력되고, {len(rows)}건으로 "
          "환산해 볼 수 있습니다.")
    if args.dry_run:
        print("\n낱말 겹침 미리보기 (호출 없음 · 대리 측정)")
        print(relevance_preview(rows, criteria))
        print("\n--dry-run 이므로 요청을 보내지 않습니다. 첫 사례의 프롬프트:\n")
        print("─" * 70)
        print(prompts[0][:1600] + ("…" if len(prompts[0]) > 1600 else ""))
        print("─" * 70)
        return

    out = Path(args.output)
    done = set()
    existing = []
    if args.resume and out.exists():
        existing = load_jsonl(out)
        done = {(r["source"], r["page"], r["serial"], r["pair_index"])
                for r in existing if "error" not in r}
        print(f"  이어하기: {len(done)}건은 건너뜁니다.")

    schema = apply_schema(len(criteria))
    fingerprint = criteria_fingerprint(criteria)
    client = connect()
    preflight(client, schema)

    records = list(existing)
    try:
        for i, (row, prompt) in enumerate(zip(rows, prompts), 1):
            key = (row["source"], row["page"], row["serial"], row.get("pair_index", 1))
            if key in done:
                continue
            result = call_structured(client, APPLY_SYSTEM, prompt, schema,
                                     max_tokens=apply_token_cap(len(criteria)),
                                     effort="low")
            record = {
                "source": row["source"], "serial": row["serial"], "page": row["page"],
                "pair_index": row.get("pair_index", 1),
                "criteria_fingerprint": fingerprint,
            }
            if "error" in result:
                record.update(result)
            else:
                answers = {a["id"]: a["answer"] for a in result["data"]["answers"]}
                record.update({
                    "answers": [answers.get(j, "unknown") for j in range(len(criteria))],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                })
            records.append(record)
            write_jsonl(out, records)
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}")
    except FatalApiError as exc:
        print(f"\n중단 — 계정 수준 오류입니다.\n  {exc}")
        print(f"  여기까지 {len(records)}건 저장. --resume 으로 이어가세요.")

    ok = [r for r in records if "error" not in r]
    print(f"\n{len(records)}건 처리 · 성공 {len(ok)} · 실패 {len(records) - len(ok)}")
    if ok:
        from collections import Counter
        dist = Counter(a for r in ok for a in r["answers"])
        total = sum(dist.values())
        print("  답 분포: " + ", ".join(f"{k} {v} ({v/total:.0%})" for k, v in dist.most_common()))
    print(f"  실제 비용 ${estimate_cost(records):.3f}")
    print(f"-> {out}")


def cmd_predict(args) -> None:
    """dev 답에서 가중치를 뽑고 test 답을 라벨로 바꾼다. API 를 쓰지 않는다.

    **최종 라벨은 여기서 정해진다.** 모델은 기준별 예/아니오까지만 말했다.
    """
    from pathlib import Path

    from app.core.io import key_of, load_jsonl, write_jsonl
    from app.rules.criteria_vote import confidence, fit, score

    criteria = load_criteria(args.criteria)
    n = len(criteria)

    dev_rows = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]
    dev_answer_rows = load_jsonl(Path(args.dev_answers))
    check_fingerprint(dev_answer_rows, criteria, "predict(dev)")
    dev_answers = {key_of(r): r["answers"] for r in dev_answer_rows if "answers" in r}
    covered = sum(1 for r in dev_rows if key_of(r) in dev_answers)
    if covered < len(dev_rows):
        print(f"  ⚠ dev {len(dev_rows)}건 중 답이 있는 것은 {covered}건입니다. "
              "가중치가 그만큼만 근거를 갖습니다.")

    model = fit(dev_answers, dev_rows, n)
    print(f"기준 {n}개 · dev {covered}건에서 가중치 산출\n")
    print(f"{'#':>3}  {'yes':>4}  {'가장 밀어주는 결론':>16}  {'가중치':>7}  질문")
    for c in criteria:
        w = model["criteria"][c["id"]]
        top, val = max(w["weights"].items(), key=lambda kv: kv[1])
        print(f"{c['id']:>3}  {w['n_yes']:>4}  {top:>16}  {val:>+7.2f}  {c['question'][:44]}")

    test_answers = load_jsonl(Path(args.test_answers))
    check_fingerprint(test_answers, criteria, "predict(test)")
    preds = []
    for r in test_answers:
        if "answers" not in r:
            continue
        result = score(model, r["answers"])
        preds.append({
            "source": r["source"], "serial": r["serial"], "page": r["page"],
            "pair_index": r.get("pair_index", 1),
            "predicted": result["predicted"],
            "confidence": confidence(result, args.high, args.medium),
            "margin": round(result["margin"], 4),
            "rule": "criteria:" + (",".join(map(str, result["fired"])) or "none"),
        })
    write_jsonl(Path(args.output), preds)

    from collections import Counter
    print(f"\n{len(preds)}건 -> {args.output}")
    for label, k in Counter(p["predicted"] for p in preds).most_common():
        print(f"  {label}: {k} ({k / len(preds):.1%})")
    conf = Counter(p["confidence"] for p in preds)
    print("  신뢰도: " + ", ".join(f"{k} {conf[k]}" for k in ("high", "medium", "low") if conf[k]))


def cmd_weights(args) -> None:
    """dev 답만으로 기준의 값어치를 잰다. API 를 쓰지 않고 test 를 열지 않는다.

    ## 왜 필요한가

    `apply` 를 dev 에만 돌린 뒤 test 에 돈을 더 쓸지 정해야 하는 자리가 있다.
    그런데 `predict` 는 test 답을 요구하므로 그 판단에 쓸 수 없다.

    여기서 보는 것은 셋이다.

        1. 기준별 가중치      — 어떤 기준이 어느 결론을 미는가
        2. 발화 커버리지      — 아무 기준도 안 걸리는 사례가 얼마나 되는가
        3. **LOO 교차검증**   — dev 에서 정직하게 잰 성능

    3 번이 핵심이다. 가중치를 dev 전체에서 뽑고 그 dev 를 다시 채점하면 자기
    답안지로 채점하는 것이다. 사례 하나를 빼고 나머지로 가중치를 뽑아 그
    하나를 맞히기를 85번 반복하면, test 를 열지 않고도 test 에서 무엇이
    나올지 가늠할 수 있다. 산술뿐이라 비용은 0 이다.
    """
    from collections import Counter
    from pathlib import Path

    from app.core.io import key_of, load_jsonl
    from app.domain.labels import NON_ACTIONS
    from app.evaluation.metrics import macro_f1
    from app.evaluation.validity import Claim
    from app.rules.criteria_vote import confidence, fit, score

    criteria = load_criteria(args.criteria)
    n = len(criteria)
    dev_rows = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]
    answer_rows = load_jsonl(Path(args.dev_answers))
    check_fingerprint(answer_rows, criteria, "weights")
    answers = {key_of(r): r["answers"] for r in answer_rows if "answers" in r}
    usable = [r for r in dev_rows if key_of(r) in answers]
    print(f"기준 {n}개 · dev {len(dev_rows)}건 중 답이 있는 것 {len(usable)}건\n")
    if not usable:
        raise SystemExit("dev 답이 하나도 없습니다. apply 를 먼저 돌리세요.")

    # ── 0. 답이 있는 것이 dev 를 대표하는가 ────────────────────────
    # apply 는 파일 순서대로 돈다. 중간에 멈추면 남은 것은 **무작위 표본이
    # 아니다.** 그 위에서 잰 수치를 dev 수치처럼 읽으면 EV-01·EV-08·EV-09 를
    # 네 번째로 반복하는 것이다.
    partial = len(usable) < len(dev_rows)
    full_dist = Counter(r["label"] for r in dev_rows)
    got_dist = Counter(r["label"] for r in usable)
    if partial:
        print("⚠ dev 를 다 돌리지 않았습니다. 답이 있는 것이 무작위 표본이 아닙니다.")
        print(f"  {'라벨':<6}{'dev 전체':>9}{'기대':>7}{'실제':>7}{'치우침':>9}")
        skewed = []
        for label in NON_ACTIONS:
            share = full_dist[label] / len(dev_rows)
            expected = share * len(usable)
            got = got_dist[label]
            ratio = got / expected if expected else 0.0
            if expected >= 1 and (ratio >= 1.5 or ratio <= 0.67):
                skewed.append(label)
            print(f"  {label:<6}{full_dist[label]:>9}{expected:>7.1f}{got:>7}{ratio:>8.2f}배")
        if skewed:
            print(f"  치우친 라벨: {', '.join(skewed)} — 아래 수치는 dev 값이 아닙니다.")
        print()

    # ── 1. 발화 커버리지 ──────────────────────────────────────────
    fired_counts = [sum(1 for a in answers[key_of(r)][:n] if a == "yes") for r in usable]
    silent = sum(1 for c in fired_counts if c == 0)
    print("발화 커버리지")
    print(f"  아무 기준도 걸리지 않은 사례: {silent}건 ({silent / len(usable):.1%})")
    print(f"  사례당 발화 기준 수: 평균 {sum(fired_counts) / len(usable):.1f} · "
          f"최대 {max(fired_counts)}")
    by_label: dict[str, list[int]] = {}
    for r, c in zip(usable, fired_counts):
        by_label.setdefault(r["label"], []).append(c)
    for label in NON_ACTIONS:
        got = by_label.get(label)
        if got:
            mute = sum(1 for x in got if x == 0)
            print(f"    {label:<4} {len(got):>3}건 · 침묵 {mute}건 · 평균 발화 "
                  f"{sum(got) / len(got):.1f}")

    # ── 2. 기준별 가중치 ──────────────────────────────────────────
    model = fit(answers, usable, n)
    print(f"\n기준별 가중치 (상위 {args.top}개, |가중치| 순)")
    print(f"{'#':>3}  {'yes':>4}  {'미는 결론':>10}  {'가중치':>7}  질문")
    ranked = sorted(
        criteria,
        key=lambda c: -max(model["criteria"][c["id"]]["weights"].values()),
    )
    for c in ranked[: args.top]:
        w = model["criteria"][c["id"]]
        top, val = max(w["weights"].items(), key=lambda kv: kv[1])
        print(f"{c['id']:>3}  {w['n_yes']:>4}  {top:>10}  {val:>+7.2f}  {c['question'][:46]}")

    pushing = {lab: sum(1 for c in criteria
                        if max(model["criteria"][c["id"]]["weights"].items(),
                               key=lambda kv: kv[1])[0] == lab)
               for lab in NON_ACTIONS}
    print("\n  가장 미는 결론별 기준 수: "
          + " · ".join(f"{k} {v}" for k, v in pushing.items()))

    # ── 3. LOO 교차검증 ───────────────────────────────────────────
    print(f"\nLOO 교차검증 — 사례 하나를 빼고 가중치를 다시 뽑기를 {len(usable)}번")
    pairs, conf_of = [], []
    for i, held in enumerate(usable):
        rest = usable[:i] + usable[i + 1:]
        m = fit(answers, rest, n)
        result = score(m, answers[key_of(held)])
        pairs.append((held["label"], result["predicted"]))
        conf_of.append(confidence(result, args.high, args.medium))

    macro, per = macro_f1(pairs, NON_ACTIONS)
    print(f"  매크로 F1 {macro:.3f}")
    print(f"  {'라벨':<6} {'정답수':>5} {'재현율':>7} {'정밀도':>7} {'F1':>7}")
    for label in NON_ACTIONS:
        row = per[label]
        print(f"  {label:<6} {row['support']:>5} {row['recall']:>7.3f} "
              f"{row['precision']:>7.3f} {row['f1']:>7.3f}")

    # 수치는 무효 기준을 통과해야 판정이 된다. 점추정을 적고 옆에 작은 글씨로
    # 주의를 다는 방식은 이미 실패했다 — 사람은 굵은 숫자를 읽는다 (EV-22).
    # 무효/한계 기준은 `app/evaluation/validity.py` 한 곳에서만 정한다.
    claim = Claim(
        name="조치 재현율",
        numerator=round(per["조치"]["support"] * per["조치"]["recall"]),
        denominator=per["조치"]["support"],
        sample=Counter(r["label"] for r in usable),
        population=Counter(r["label"] for r in dev_rows),
        threshold=0.286,
    )
    print("\n  사전 등록한 기준: 조치 재현율 > 0.286 (E1/E3 LLM)")
    print(claim.render(indent="  "))
    print("  주의 — 이것은 dev 추정치다. test 수치가 아니다.")

    dist = Counter(conf_of)
    print("\n  신뢰도 분포: "
          + ", ".join(f"{k} {dist[k]}" for k in ("high", "medium", "low") if dist[k]))


def cmd_status(args) -> None:
    """파이프라인 어디까지 왔는지 보여준다. 파일만 읽는다.

    단계가 여섯이고 그중 셋이 돈을 쓰므로, "지금 뭘 돌려야 하나" 를 매번
    문서에서 찾게 하면 안 된다.
    """
    from pathlib import Path

    from app.core.io import load_jsonl

    steps = [
        ("1 extract",     args.raw,        "dev 회답에서 기준 추출 (API · 약 $2.5)"),
        ("2 consolidate", args.criteria,   "기준 통합 (무료)"),
        ("3 apply dev",   args.dev_answers, "dev 요청문에 적용 (API · 약 $1.0)"),
        ("4 apply test",  args.test_answers, "test 요청문에 적용 (API · 약 $2.0)"),
        ("5 predict",     args.predictions, "답을 라벨로 (무료)"),
    ]
    print("Phase 5 진행 상황\n")
    nxt = None
    for name, path, what in steps:
        path = Path(path)
        if not path.exists():
            print(f"  [ ] {name:<14} {what}")
            nxt = nxt or (name, path)
            continue
        try:
            rows = load_jsonl(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] {name:<14} 읽을 수 없음 — {exc}")
            continue
        errors = sum(1 for r in rows if "error" in r)
        extra = f" · 실패 {errors}" if errors else ""
        print(f"  [x] {name:<14} {len(rows)}건{extra}  ({path})")
        if errors:
            nxt = nxt or (name + " (--resume)", path)

    if nxt:
        print(f"\n  다음: {nxt[0]}")
        print("  각 단계에 --dry-run 을 붙이면 요청을 보내지 않고 비용만 확인합니다.")
    else:
        print("\n  모든 단계가 끝났습니다. 채점:")
        print("    python3 scripts/evaluate.py --gold data/eval/nonaction_test.jsonl \\")
        print(f"        --pred {args.predictions} --labels nonaction --name criteria")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="dev 회답에서 판단 기준을 뽑는다")
    ex.add_argument("--dev", required=True)
    ex.add_argument("--cases", required=True)
    ex.add_argument("--output", required=True)
    ex.add_argument("--limit", type=int, default=0)
    ex.add_argument("--resume", action="store_true")
    ex.add_argument("--dry-run", action="store_true")
    ex.set_defaults(func=cmd_extract)

    co = sub.add_parser("consolidate", help="사례별 기준을 하나의 목록으로 합친다 (API 없음)")
    co.add_argument("--input", required=True)
    co.add_argument("--output", required=True)
    co.add_argument("--threshold", type=float, default=0.55)
    co.add_argument("--min-support", type=int, default=2)
    co.add_argument("--top", type=int, default=25)
    co.set_defaults(func=cmd_consolidate)

    ap_ = sub.add_parser("apply", help="기준을 요청문에 적용한다 (회답은 보지 않는다)")
    ap_.add_argument("--gold", required=True)
    ap_.add_argument("--criteria", required=True)
    ap_.add_argument("--output", required=True)
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--resume", action="store_true")
    ap_.add_argument("--dry-run", action="store_true")
    ap_.set_defaults(func=cmd_apply)

    we = sub.add_parser("weights", help="dev 답만으로 기준의 값어치를 잰다 (API 없음)")
    we.add_argument("--criteria", required=True)
    we.add_argument("--dev", required=True, help="dev 평가셋 (라벨)")
    we.add_argument("--dev-answers", required=True)
    we.add_argument("--top", type=int, default=25)
    we.add_argument("--high", type=float, default=1.0)
    we.add_argument("--medium", type=float, default=0.4)
    we.set_defaults(func=cmd_weights)

    pr = sub.add_parser("predict", help="답을 라벨로 바꾼다 — 결정론 (API 없음)")
    pr.add_argument("--criteria", required=True)
    pr.add_argument("--dev", required=True, help="dev 평가셋 (라벨)")
    pr.add_argument("--dev-answers", required=True)
    pr.add_argument("--test-answers", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--high", type=float, default=1.0, help="margin 문턱 (dev 에서 정할 것)")
    pr.add_argument("--medium", type=float, default=0.4)
    pr.set_defaults(func=cmd_predict)

    st = sub.add_parser("status", help="파이프라인 진행 상황 (파일만 읽는다)")
    st.add_argument("--raw", default="data/interim/criteria_raw.jsonl")
    st.add_argument("--criteria", default="data/eval/criteria.jsonl")
    st.add_argument("--dev-answers", default="data/interim/answers_dev.jsonl")
    st.add_argument("--test-answers", default="data/interim/answers_test.jsonl")
    st.add_argument("--predictions", default="data/processed/pred_nonaction_criteria.jsonl")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
