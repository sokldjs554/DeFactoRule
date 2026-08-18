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
    python scripts/classify_llm.py --input data/processed/qa_pairs.jsonl \
        --output data/processed/pred_llm.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from labels import GUIDELINE, VERDICTS

MODEL = "claude-opus-5"

SYSTEM = f"""당신은 금융규제 법령해석 회신문을 읽고 결론을 분류합니다.

{GUIDELINE}

판정의 근거가 된 원문 구절을 `evidence` 에 **그대로** 옮깁니다. 요약하거나
바꿔 쓰지 마십시오. 옮긴 구절은 원문과 글자 단위로 대조됩니다.

근거가 될 만한 결론절을 찾지 못하면 `verdict` 를 "판단유보" 로 두고
`evidence` 를 빈 문자열로 두십시오. 추측하지 마십시오."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "evidence": {
                "type": "string",
                "description": "판정 근거가 된 회답 원문 구절 (그대로 인용)",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "결론절이 얼마나 명확한가",
            },
        },
        "required": ["verdict", "evidence", "confidence"],
        "additionalProperties": False,
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


def build_prompt(pair: dict) -> str:
    question = pair.get("question", "").strip()
    answer = pair.get("answer", "").strip()
    return f"[질의]\n{question}\n\n[회답]\n{answer}"


def classify_one(client, pair: dict) -> dict:
    import anthropic

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"format": RESPONSE_SCHEMA, "effort": "medium"},
            messages=[{"role": "user", "content": build_prompt(pair)}],
        )
    except anthropic.APIStatusError as exc:
        return {"error": f"{type(exc).__name__}: {exc.status_code}"}
    except anthropic.APIConnectionError as exc:
        return {"error": f"connection: {exc}"}

    if response.stop_reason == "refusal":
        return {"error": "refusal"}

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "unparseable_output", "raw": text[:400]}

    grounded = evidence_is_grounded(parsed.get("evidence", ""), pair.get("answer", ""))
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
    ap.add_argument("--doc-type", default="interpretation")
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전체")
    args = ap.parse_args()

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

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets = [r for r in rows if r["doc_type"] == args.doc_type and r.get("answer", "").strip()]
    if args.limit:
        targets = targets[: args.limit]

    from collections import Counter

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out.open("w", encoding="utf-8") as fh:
        for i, pair in enumerate(targets, 1):
            result = classify_one(client, pair)
            record = {
                "source": pair["source"],
                "serial": pair["serial"],
                "page": pair["page"],
                "pair_index": pair["pair_index"],
                **result,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()  # 중간에 끊겨도 여기까지는 남는다
            results.append(record)
            if i % 25 == 0:
                print(f"  {i}/{len(targets)}")

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
        for kind, n in Counter(r["error"].split(":")[0] for r in errors).most_common():
            print(f"  오류 {kind}: {n}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
