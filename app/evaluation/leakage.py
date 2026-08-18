"""정답 누출 감사 — 어떤 낱말이 결론을 그대로 비추는가.

요청대상행위에서 이미 한 번 겪었다. '비조치' 라는 낱말이 요청문에 그대로 남아
있었고, 낱말만 지웠더니 '를요청' 이라는 흔적이 다시 신호가 됐고, 마스크 토큰
'[결론표현]' 자체가 세 번째 누출이 됐다. 세 번 다 **눈으로 찾았다.**

회답 본문(판단이유)을 쓰기 시작하면 이 문제가 훨씬 심해진다. 판단 근거에는
"따라서 비조치 의견을 표명한다" 같은 문장이 당연히 들어 있다. 그것을 그대로
모델에 넣고 결론을 맞히면 100% 가 나오고, 아무것도 배우지 못한다.

그래서 감사를 도구로 만든다. 어떤 항목이든 겨눠서 **클래스를 강하게 가리키는
표현**을 뽑아 준다.

## 지표

낱말 t 에 대해

    지지도    t 가 나타나는 사례 수
    쏠림      t 가 있을 때 가장 흔한 클래스의 비율
    상승      쏠림 − 그 클래스의 전체 기저율

상승이 큰 것이 위험하다. 기저율이 74%인 클래스에서 쏠림 80%는 별 뜻이 없지만,
기저율 8%인 클래스에서 쏠림 80%는 그 낱말이 답을 알려준다는 뜻이다.

    python scripts/leakage_audit.py --input data/processed/cases_nonaction.jsonl \\
        --field 판단이유 --label-key decision
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.core.io import load_jsonl, write_json
from app.evaluation.comparison import holm

# 낱말 경계가 불안정한 문서군이라 문자 n-gram 을 본다.
NGRAM_LENGTHS = (2, 3, 4, 5, 6)

# 처음에는 5로 뒀는데, 지지도 5~6 짜리가 상위를 전부 채웠다. 소수 클래스
# 기저율이 8.6% 라 6건이 우연히 한 클래스로 몰리는 일이 드물지 않고, 무엇보다
# **같은 구절의 조각들이 따로 세어졌다** — '가능하거'·'능하거나'·'거나현' 이
# 같은 문서 6건에서 온 하나의 발견인데 열 줄을 차지했다.
MIN_SUPPORT = 12

JUNK = re.compile(r"\s+")


def squeeze(text: str) -> str:
    return JUNK.sub("", text or "")


def terms(text: str) -> set:
    s = squeeze(text)
    out = set()
    for n in NGRAM_LENGTHS:
        for i in range(len(s) - n + 1):
            out.add(s[i : i + n])
    return out


def binomial_tail(hits: int, trials: int, p: float) -> float:
    """P(X >= hits), X ~ Binom(trials, p). 정확 계산이다.

    "지지도 6건이 전부 조치" 가 놀라운 일인지 아닌지를 기저율에 비추어 잰다.
    기저율 8.6% 에서 6/6 은 4e-7 이지만, 후보가 1만 개면 그중 몇 개는 그냥
    나온다. 그래서 Holm 보정을 함께 건다.
    """
    return sum(
        math.comb(trials, k) * (p ** k) * ((1 - p) ** (trials - k))
        for k in range(hits, trials + 1)
    )


def field_text(row: dict, field: str) -> str:
    if "fields" in row:
        return row["fields"].get(field) or ""
    return row.get(field) or ""


def audit(rows: list[dict], field: str, label_key: str, min_support: int = MIN_SUPPORT) -> dict:
    """표현별 클래스 쏠림을 재고, **덮는 문서 집합이 같은 표현은 하나로 묶는다.**

    묶지 않으면 겹치는 n-gram 이 같은 발견을 열 번 보고한다. 규칙 학습기에서
    같은 처리를 했고 이유도 같다 — 목록을 사람이 읽을 수 있어야 한다.
    """
    labeled = [r for r in rows if r.get(label_key)]
    base = Counter(r[label_key] for r in labeled)
    total = len(labeled)

    by_term: dict[str, set] = defaultdict(set)
    for i, row in enumerate(labeled):
        for t in terms(field_text(row, field)):
            by_term[t].add(i)

    # 같은 문서 집합을 덮는 표현들을 한 무리로
    groups: dict[frozenset, list[str]] = defaultdict(list)
    for term, docs in by_term.items():
        if len(docs) >= min_support:
            groups[frozenset(docs)].append(term)

    findings = []
    for docs, forms in groups.items():
        dist = Counter(labeled[i][label_key] for i in docs)
        label, hits = dist.most_common(1)[0]
        support = len(docs)
        prior = base[label] / total
        findings.append({
            "term": min(forms, key=lambda t: (len(t), t)),
            "surface_forms": len(forms),
            "support": support,
            "label": label,
            "concentration": hits / support,
            "prior": prior,
            "lift": hits / support - prior,
            "p_binomial": binomial_tail(hits, support, prior),
        })

    if findings:
        for f, (p_adj, keep) in zip(findings, holm([x["p_binomial"] for x in findings])):
            f["p_holm"] = p_adj
            f["significant"] = bool(keep)
    findings.sort(key=lambda f: (f["p_binomial"], -f["lift"], -f["support"]))
    return {
        "field": field,
        "n": total,
        "min_support": min_support,
        "base_rates": {k: v / total for k, v in base.items()},
        "n_candidates": len(by_term),
        "n_groups": len(groups),
        "findings": findings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--label-key", default="decision")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--report")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    result = audit(rows, args.field, args.label_key, args.min_support)

    print(f"'{args.field}' 항목 · 라벨 있는 사례 {result['n']}건")
    print("  기저율: " + ", ".join(
        f"{k} {v:.1%}" for k, v in sorted(result["base_rates"].items(), key=lambda kv: -kv[1])))
    print(f"  후보 표현 {result['n_candidates']:,}개 -> 덮는 문서가 같은 것끼리 묶어 "
          f"{result['n_groups']}무리 (지지도 {args.min_support} 이상)\n")

    risky = [f for f in result["findings"] if f["significant"]]
    print(f"Holm 보정 후 유의한 무리 {len(risky)}개 — 상위 {min(args.top, len(risky))}개\n")
    if risky:
        print(f"{'표현':>12}  {'형태':>4}  {'지지도':>5}  {'결론':>5}  "
              f"{'쏠림':>6}  {'기저율':>6}  {'상승':>7}  {'p(Holm)':>8}")
        for f in risky[: args.top]:
            print(f"{f['term']:>12}  {f['surface_forms']:>4}  {f['support']:>5}  "
                  f"{f['label']:>5}  {f['concentration']:>6.1%}  {f['prior']:>6.1%}  "
                  f"{f['lift']:>+7.1%}  {f['p_holm']:>8.4f}")
    else:
        print("  없음 — 이 항목은 단일 표현으로 결론을 비추지 않는다.")

    print("\n  '형태' = 같은 문서 집합을 덮는 표면형의 개수. 겹치는 n-gram 을 묶은 결과다.")
    print("  상승 = 그 표현이 있을 때의 쏠림 − 해당 결론의 전체 기저율")
    print("  기저율 72%인 결론에서 쏠림 80%는 별 뜻이 없다. 기저율 8.6%인 결론에서")
    print("  쏠림 100%면 그 표현이 답을 알려주고 있다는 뜻이다.")

    if args.report:
        write_json(args.report, {**result, "findings": result["findings"][:300]})
        print(f"\n-> {args.report}")


if __name__ == "__main__":
    main()
