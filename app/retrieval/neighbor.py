"""최근접 선례 기준선 — 검색만으로 어디까지 되는가.

이 도메인에서 가장 자연스러운 전략은 **닮은 선례를 찾아 그 결론을 따라가는
것**이다. 규제 실무가 실제로 그렇게 움직이고, RAG 로 이 문제를 풀겠다는
접근도 결국 이것이다. 그러므로 LLM 을 논하기 전에 이 선을 먼저 그어야 한다.

dev 를 선례 창고로 삼고, test 사례마다 가장 닮은 선례 하나의 결론을 그대로
예측으로 낸다. 유사도는 문자 4-gram 의 IDF 가중 코사인이다.

신뢰도 구간은 **dev 안의 leave-one-out 으로만** 정했다. test 를 보고 정하면
그 자체가 누출이다. 실측은 이렇다.

    유사도 ≥ 0.60   n=32   1-NN 적중률 93.8%   -> high
    0.15 ~ 0.60     n= 5   적중률이 흔들림      -> medium
    < 0.15          n=48   적중률 50.0%        -> low

낮은 구간의 50%는 "동전 던지기" 가 아니라 "닮은 선례가 아예 없다" 는 뜻이다.
그 구간에서 이 기준선이 내놓는 답은 근거가 없다.

    python scripts/baseline_neighbor.py --dev data/eval/nonaction_dev.jsonl \\
        --gold data/eval/nonaction_test.jsonl \\
        --output data/processed/pred_nonaction_neighbor.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.core.io import load_jsonl, write_jsonl
from app.core.paths import PROCESSED
from app.domain import similarity as domain_similarity
from app.evaluation.confusable import cosine, idf_table, weighted_vector

# dev leave-one-out 에서 정했다. test 는 열지 않았다.
# 도메인 문턱을 그대로 쓴다. 여기서 따로 정하면 또 어긋난다.
HIGH = domain_similarity.TRUST
MEDIUM = domain_similarity.DOUBT


def band(similarity: float) -> str:
    if similarity >= HIGH:
        return "high"
    if similarity >= MEDIUM:
        return "medium"
    return "low"


def predict(rows: list[dict], precedents: list[dict], idf: dict) -> list[dict]:
    pvecs = [weighted_vector(p["request"], idf) for p in precedents]
    out = []
    for row in rows:
        v = weighted_vector(row["request"], idf)
        best_i, best = -1, -1.0
        for i, pv in enumerate(pvecs):
            s = cosine(v, pv)
            if s > best:  # 동점은 앞선 선례가 이긴다 — 난수 없이 재현된다
                best, best_i = s, i
        nb = precedents[best_i]
        out.append({
            "source": row["source"],
            "serial": row["serial"],
            "page": row["page"],
            "pair_index": row.get("pair_index", 1),
            "predicted": nb["label"],
            "confidence": band(best),
            "rule": f"nearest:{nb['source']}#{nb['serial']}",
            "similarity": round(best, 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dev", required=True, help="선례 창고")
    ap.add_argument("--output", required=True)
    ap.add_argument("--corpus", default=str(PROCESSED / "cases_nonaction.jsonl"))
    args = ap.parse_args()

    rows = [r for r in load_jsonl(Path(args.gold)) if r.get("request")]
    dev = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]

    corpus = Path(args.corpus)
    texts = (
        [(c["fields"].get("요청대상행위") or "") for c in load_jsonl(corpus)]
        if corpus.exists()
        else [r["request"] for r in rows + dev]
    )
    idf = idf_table(texts)

    preds = predict(rows, dev, idf)
    write_jsonl(Path(args.output), preds)

    print(f"최근접 선례: {len(preds)}건 -> {args.output}")
    print(f"  선례 창고 {len(dev)}건 · IDF {len(texts)}건")
    for label, n in Counter(p["predicted"] for p in preds).most_common():
        print(f"  {label}: {n} ({n / len(preds):.1%})")
    conf = Counter(p["confidence"] for p in preds)
    print("  신뢰도: " + ", ".join(f"{k} {conf[k]}" for k in ("high", "medium", "low") if conf[k]))


if __name__ == "__main__":
    main()
