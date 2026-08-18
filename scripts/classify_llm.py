"""회답의 결론을 LLM 으로 판정한다.

규칙 baseline 은 864건 중 41% 만 판정하고 나머지는 읽지 못한다. 그 59% 가
LLM 이 맡을 자리다. 결론절의 표현이 사례마다 달라 열거로는 덮이지 않기 때문이다.

설계 원칙 — LLM 에게 시키는 것과 시키지 않는 것을 나눈다.

  LLM     회답 자연어 → 결론 라벨 (긍정/부정/조건부/판단유보)
          그리고 그 판정의 근거가 된 **원문 구절**을 그대로 인용

  코드    라벨 유효성 검사, 인용 구절이 원문에 실재하는지 대조,
          신뢰도 임계값 미달 시 판단유보 처리, 결과 집계

인용 구절을 원문과 대조하는 것이 핵심이다. 모델이 그럴듯한 요약을 지어내면
대조에서 걸린다. 이 검증 없이 라벨만 받으면 환각을 발견할 방법이 없다.

    export ANTHROPIC_API_KEY=...   # 또는 `ant auth login`
    # 법령해석 회답 -> 결론
    python scripts/classify_llm.py --task verdict \
        --input data/processed/qa_pairs.jsonl \
        --output data/processed/pred_llm.jsonl --limit 50

    # 비조치 요청 -> 당국 결론 예측 (순환 없는 평가셋)
    python scripts/classify_llm.py --task nonaction \
        --input data/eval/nonaction_test.jsonl \
        --output data/processed/pred_nonaction_llm.jsonl --limit 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from base_rates import describe_overall, describe_sector
from labels import GUIDELINE, NON_ACTIONS, VERDICTS

BASE_RATES_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "dev_base_rates.json"

MODEL = "claude-opus-5"

# 계정 수준 오류 — 다음 요청도 똑같이 실패한다. 건별로 잡고 계속 돌면
# 남은 전부를 헛되이 던지게 된다. 즉시 중단한다.
FATAL_MARKERS = (
    "credit balance is too low",
    "invalid x-api-key",
    "authentication_error",
    "permission_error",
    "Your account has been disabled",
)


class FatalApiError(RuntimeError):
    """계속 시도해도 소용없는 오류."""

# 비조치의견서 과제의 지침. 법령해석과 달리 **결론이 아니라 요청 자체를 보고
# 당국이 어떻게 답했을지 예측**하는 것이므로 성격이 완전히 다르다.
NONACTION_GUIDELINE = """\
신청인이 하려는 행위만 보고, 금융당국이 어떤 결론을 냈을지 예측합니다.
회답이나 판단 이유는 주어지지 않습니다.

비조치  당국이 해당 행위에 대해 제재하지 않겠다고 회신한 경우
조치    제재 대상이거나 규정 위반에 해당한다고 회신한 경우
기타    비조치도 조치도 아닌 형태로 회신한 경우
        (해석만 제시, 소관이 아님, 별도 절차 안내 등)

이 과제는 어렵습니다. 세 부류의 요청문은 형태가 거의 같고
("~하는 것이 ~에 해당하는지 여부"), 결론은 요청문의 표면이 아니라 법적 분석의
결과입니다. 근거가 약하면 억지로 확신하지 말고 `confidence` 를 낮추십시오."""

SYSTEM = f"""당신은 금융규제 법령해석 회신문을 읽고 결론을 분류합니다.

{GUIDELINE}

판정의 근거가 된 원문 구절을 `evidence` 에 **그대로** 옮깁니다. 요약하거나
바꿔 쓰지 마십시오. 옮긴 구절은 원문과 글자 단위로 대조됩니다.

근거가 될 만한 결론절을 찾지 못하면 `verdict` 를 "판단유보" 로 두고
`evidence` 를 빈 문자열로 두십시오. 추측하지 마십시오."""

NONACTION_SYSTEM = f"""당신은 금융규제 비조치의견서 신청 내용을 읽고, 당국이 어떤
결론을 냈을지 예측합니다.

{NONACTION_GUIDELINE}

판단의 단서가 된 요청문 구절을 `evidence` 에 **그대로** 옮깁니다. 요약하거나 바꿔
쓰지 마십시오. 옮긴 구절은 원문과 글자 단위로 대조됩니다.

단서를 찾지 못하면 `evidence` 를 빈 문자열로 두고 `confidence` 를 "low" 로
두십시오. 지어내지 마십시오."""


def _schema(labels: tuple[str, ...], evidence_hint: str) -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": list(labels)},
                "evidence": {"type": "string", "description": evidence_hint},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["verdict", "evidence", "confidence"],
            "additionalProperties": False,
        },
    }


# 과제마다 지침·스키마·입력 필드가 다르다. 한 곳에 모아 두어 실험이 섞이지 않게 한다.
TASKS = {
    "verdict": {
        "system": None,  # 아래에서 SYSTEM 을 넣는다
        "labels": VERDICTS,
        "evidence_hint": "판정 근거가 된 회답 원문 구절 (그대로 인용)",
        "input_fields": ("question", "answer"),
        "titles": ("질의", "회답"),
        "source_field": "answer",
    },
    "nonaction": {
        "system": NONACTION_SYSTEM,
        "labels": NON_ACTIONS,
        "evidence_hint": "판단 단서가 된 요청문 구절 (그대로 인용)",
        "input_fields": ("request",),
        "titles": ("요청대상행위",),
        "source_field": "request",
        "context": None,
    },
    # ── E4 프롬프트 변형 ────────────────────────────────────────
    # E3 에서 모델이 소수 클래스를 과잉 예측한다는 것을 확인했다. 기저율을 알려주면
    # 줄어드는가? 기저율은 dev 에서만 뽑는다 — test 에서 뽑으면 정답 누출이다.
    #
    # 두 변형을 따로 둔 이유는 교란을 가르기 위해서다. sector 변형은 업권을
    # 알려주는 것과 그 업권의 기저율을 알려주는 것이 섞여 있다. prior 변형은
    # 업권을 밝히지 않고 전체 분포만 주므로 기저율 효과만 잰다.
    "nonaction_prior": {
        "system": NONACTION_SYSTEM,
        "labels": NON_ACTIONS,
        "evidence_hint": "판단 단서가 된 요청문 구절 (그대로 인용)",
        "input_fields": ("request",),
        "titles": ("요청대상행위",),
        "source_field": "request",
        "context": "overall",
    },
    "nonaction_sector": {
        "system": NONACTION_SYSTEM,
        "labels": NON_ACTIONS,
        "evidence_hint": "판단 단서가 된 요청문 구절 (그대로 인용)",
        "input_fields": ("request",),
        "titles": ("요청대상행위",),
        "source_field": "request",
        "context": "sector",
    },
}

def normalize(text: str) -> str:
    """대조용 정규화. 공백과 조판 잔재만 걷어낸다 — 글자는 건드리지 않는다."""
    return re.sub(r"\s+", "", text)


def evidence_is_grounded(evidence: str, answer: str) -> bool:
    """인용 구절이 원문에 실재하는지 확인한다."""
    if not evidence.strip():
        return True  # 인용을 포기한 것은 환각이 아니다
    return normalize(evidence) in normalize(answer)


def build_prompt(row: dict, task: dict) -> str:
    parts = []
    for field, title in zip(task["input_fields"], task["titles"]):
        parts.append(f"[{title}]\n{(row.get(field) or '').strip()}")
    # 기저율은 사용자 메시지에 붙인다. 사안마다 달라지므로 시스템 프롬프트에
    # 넣으면 캐시되는 앞부분이 매번 깨진다.
    table = task.get("base_rates")
    mode = task.get("context")
    if table and mode == "overall":
        parts.append(describe_overall(table))
    elif table and mode == "sector":
        parts.append(describe_sector(table, row.get("sector")))
    return "\n\n".join(parts)


def classify_one(client, row: dict, task: dict) -> dict:
    import anthropic

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=task["system"],
            thinking={"type": "adaptive"},
            output_config={
                "format": _schema(task["labels"], task["evidence_hint"]),
                "effort": "medium",
            },
            messages=[{"role": "user", "content": build_prompt(row, task)}],
        )
    except anthropic.APIStatusError as exc:
        # 상태 코드만 남기면 원인을 알 수 없다. 실제로 그래서 400 이 39건 났을 때
        # 아무것도 진단하지 못했다. 메시지와 본문을 함께 남긴다.
        detail = getattr(exc, "message", "") or str(exc)
        body = getattr(exc, "body", None)
        blob = f"{detail} {body}"
        if any(m.lower() in blob.lower() for m in FATAL_MARKERS):
            raise FatalApiError(detail.strip()) from exc
        return {
            "error": f"{type(exc).__name__}: {exc.status_code}",
            "error_detail": detail[:600],
            "error_body": json.dumps(body, ensure_ascii=False)[:600] if body else None,
            "prompt_chars": len(build_prompt(row, task)),
        }
    except anthropic.APIConnectionError as exc:
        return {"error": f"connection: {exc}"}

    if response.stop_reason == "refusal":
        return {"error": "refusal"}

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "unparseable_output", "raw": text[:400]}

    grounded = evidence_is_grounded(
        parsed.get("evidence", ""), row.get(task["source_field"], "")
    )
    return {
        "predicted": parsed["verdict"],
        "evidence": parsed["evidence"],
        "confidence": parsed["confidence"],
        "evidence_grounded": grounded,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="verdict",
        help="verdict: 회답 -> 결론 / nonaction: 요청 -> 당국 결론 예측",
    )
    ap.add_argument("--doc-type", default="interpretation")
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전체")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="출력 파일에 이미 성공으로 남은 건은 건너뛴다. 실패분만 다시 부를 때 쓴다.",
    )
    args = ap.parse_args()

    task = dict(TASKS[args.task])
    if task["system"] is None:
        task["system"] = SYSTEM
    if task.get("context"):
        if not BASE_RATES_PATH.exists():
            sys.exit(
                f"기저율 파일이 없습니다: {BASE_RATES_PATH}\n"
                "python3 scripts/base_rates.py --dev data/eval/nonaction_dev.jsonl "
                "--output data/eval/dev_base_rates.json"
            )
        task["base_rates"] = json.loads(BASE_RATES_PATH.read_text(encoding="utf-8"))
        if task["base_rates"].get("source") != "dev":
            sys.exit("기저율이 dev 에서 나온 것이 아닙니다. 정답 누출 위험.")

    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic 패키지가 없습니다: pip install anthropic")

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # 자격증명 없음
        sys.exit(
            f"Anthropic 클라이언트를 만들 수 없습니다: {exc}\n"
            "ANTHROPIC_API_KEY 를 설정하거나 `ant auth login` 을 실행하세요."
        )

    # 170건을 던지기 전에 계정이 살아 있는지 한 번만 확인한다.
    # 최소 토큰이라 비용은 사실상 0 이고, 크레딧이 없으면 여기서 바로 멈춘다.
    try:
        client.messages.create(
            model=MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
    except anthropic.APIStatusError as exc:
        detail = getattr(exc, "message", "") or str(exc)
        blob = f"{detail} {getattr(exc, 'body', None)}"
        if any(m.lower() in blob.lower() for m in FATAL_MARKERS):
            sys.exit(
                f"사전 점검 실패 — 요청을 하나도 보내지 않았습니다.\n  {detail.strip()}\n\n"
                "  Anthropic Console 의 Plans & Billing 에서 크레딧을 확인하세요.\n"
                "  충전 직후에는 반영에 몇 분 걸릴 수 있습니다."
            )
        print(f"  ⚠ 사전 점검에서 예상치 못한 오류: {type(exc).__name__} {exc.status_code}")

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_field = task["source_field"]
    targets = [r for r in rows if (r.get(source_field) or "").strip()]
    if args.task.startswith("nonaction") and task.get("context") == "sector":
        missing = sum(1 for r in targets if not r.get("sector"))
        if missing:
            print(f"  ⚠ 업권 정보가 없는 {missing}건은 전체 기저율로 대체됩니다.")
    if args.task == "verdict":
        # 법령해석 쌍 파일에는 두 문서 종류가 섞여 있다
        targets = [r for r in targets if r.get("doc_type") == args.doc_type]
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        sys.exit(f"입력에서 '{source_field}' 필드를 가진 행을 찾지 못했습니다: {args.input}")

    from collections import Counter

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: dict[tuple, dict] = {}
    if args.resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" not in rec:
                done[(rec["source"], rec["page"], rec["serial"], rec["pair_index"])] = rec
        print(f"  이어하기: 성공 {len(done)}건은 건너뜁니다.")

    results = []
    fatal = False
    with out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(targets, 1):
            key = (row["source"], row["page"], row["serial"], row.get("pair_index", 1))
            if key in done:
                fh.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                fh.flush()
                results.append(done[key])
                continue
            try:
                result = classify_one(client, row, task)
            except FatalApiError as exc:
                # 남은 요청을 던지지 않고 지금까지의 결과를 지킨 채 멈춘다.
                remaining = len(targets) - i + 1
                print(f"\n중단 — 계정 수준 오류입니다. 남은 {remaining}건은 보내지 않았습니다.")
                print(f"  {exc}")
                print(f"\n  여기까지 {len(results)}건이 {out} 에 저장됐습니다.")
                print("  문제를 해결한 뒤 같은 명령에 --resume 을 붙이면 이어서 진행합니다.")
                fatal = True
                break
            record = {
                "source": row["source"],
                "serial": row["serial"],
                "page": row["page"],
                "pair_index": row.get("pair_index", 1),
                **result,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()  # 중간에 끊겨도 여기까지는 남는다
            results.append(record)
            if i % 25 == 0:
                print(f"  {i}/{len(targets)}")

    if fatal:
        written = {
            (r["source"], r["page"], r["serial"], r["pair_index"]) for r in results
        }
        leftover = [rec for k, rec in done.items() if k not in written]
        if leftover:
            with out.open("a", encoding="utf-8") as fh:
                for rec in leftover:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.extend(leftover)
            print(f"  이전 성공 {len(leftover)}건도 함께 보존했습니다.")

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    print(f"\n{len(results)}건 처리 · 성공 {len(ok)} · 실패 {len(errors)}")
    if ok:
        for label, n in Counter(r["predicted"] for r in ok).most_common():
            print(f"  {label}: {n} ({n / len(ok):.1%})")
        ungrounded = [r for r in ok if not r["evidence_grounded"]]
        print(f"\n인용 미대조 {len(ungrounded)} ({len(ungrounded) / len(ok):.1%})")
        cost_in = sum(r["input_tokens"] for r in ok) / 1e6 * 5.0
        cost_out = sum(r["output_tokens"] for r in ok) / 1e6 * 25.0
        print(f"추정 비용 ${cost_in + cost_out:.3f}")
    if errors:
        for kind, n in Counter(r["error"] for r in errors).most_common():
            print(f"  오류 {kind}: {n}")
        print("\n  실패 사례 상세 (앞 3건)")
        for r in errors[:3]:
            print(f"    [{r['serial']}] {r.get('error_detail', '(상세 없음)')[:200]}")
        by_sector = Counter(
            next((t.get("sector") for t in targets
                  if t["serial"] == r["serial"] and t["page"] == r["page"]), "?")
            for r in errors
        )
        print("\n  실패의 업권 분포: " + ", ".join(f"{k} {v}" for k, v in by_sector.most_common()))
        print("\n  실패분만 다시 부르려면 같은 명령에 --resume 을 붙이세요.")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
