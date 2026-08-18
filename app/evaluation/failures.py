"""LLM 예측의 오류를 뜯어본다.

점수는 무엇을 고쳐야 할지 알려주지 않는다. 어디서 어떻게 틀렸는지를 봐야 다음
수를 정할 수 있다. 네 가지를 본다.

  신뢰도 보정   모델이 high 라고 한 것이 실제로 더 맞는가
  인용 미대조   원문에 없는 구절을 근거로 든 사례
  업권별 성능   특정 업권에서만 무너지는가
  혼동 사례     어떤 요청문에서 라벨이 갈리는가

    python scripts/error_analysis.py --gold data/eval/nonaction_test.jsonl \\
        --pred data/processed/pred_nonaction_llm.jsonl --labels nonaction
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.core.io import key_of, load_jsonl
from app.domain.labels import LABEL_SETS
from app.evaluation.metrics import macro_f1


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def overlap_ratio(evidence: str, source: str) -> float:
    """인용 구절 중 원문에 연속으로 존재하는 최장 부분의 비율.

    참/거짓만으로는 "원문에 아예 없는 문장"과 "떨어진 조각을 이어붙인 것"이
    구별되지 않는다. 후자는 내용을 지어낸 것이 아니라 **연속성을 지어낸** 것이고,
    성격이 전혀 다르므로 정도를 함께 본다.
    """
    a, b = squeeze(evidence), squeeze(source)
    if not a:
        return 1.0
    for length in range(len(a), 0, -1):
        for i in range(len(a) - length + 1):
            if a[i : i + length] in b:
                return length / len(a)
    return 0.0


def section(title: str) -> None:
    print(f"\n{'─' * 62}\n{title}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", choices=sorted(LABEL_SETS), default="verdict")
    ap.add_argument("--show", type=int, default=4, help="사례를 몇 건 보여줄지")
    args = ap.parse_args()

    labels = LABEL_SETS[args.labels]
    gold = {key_of(r): r for r in load_jsonl(Path(args.gold)) if r.get("label")}
    pred = {key_of(r): r for r in load_jsonl(Path(args.pred))}
    keys = [k for k in gold if k in pred]
    rows = [(gold[k], pred[k]) for k in keys]
    correct = [(g, p) for g, p in rows if g["label"] == p.get("predicted")]

    print(f"채점 대상 {len(rows)}건 · 정답 {len(correct)}건 ({len(correct) / len(rows):.1%})")

    # ── 신뢰도 보정 ─────────────────────────────────────────────
    section("신뢰도 보정 — high 라고 한 것이 실제로 더 맞는가")
    buckets: dict[str, list[bool]] = defaultdict(list)
    for g, p in rows:
        buckets[p.get("confidence", "?")].append(g["label"] == p.get("predicted"))
    print(f"{'신뢰도':>8}  {'건수':>5}  {'정확도':>7}")
    for level in ("high", "medium", "low", "?"):
        hits = buckets.get(level)
        if not hits:
            continue
        print(f"{level:>8}  {len(hits):>5}  {sum(hits) / len(hits):>7.1%}")
    ordered = [
        (lv, sum(v) / len(v)) for lv in ("high", "medium", "low") if (v := buckets.get(lv))
    ]
    if len(ordered) >= 2:
        monotone = all(
            ordered[i][1] >= ordered[i + 1][1] for i in range(len(ordered) - 1)
        )
        print(
            "\n  신뢰도가 높을수록 정확한가: "
            + ("예 — 보정이 살아 있다" if monotone else "아니다 — 신뢰도를 걸러내기에 쓸 수 없다")
        )

    # 신뢰도로 판단유보를 만들면 어떻게 되는가
    # 낮은 신뢰도를 판단유보로 돌리면 어떤 운영점이 생기는가.
    # 이 프로젝트의 주제가 "언제 판단하면 안 되는가" 이므로 이 표가 핵심이다.
    print(f"\n{'채택 범위':>16}  {'커버리지':>8}  {'정확도':>7}  {'매크로 F1':>9}")
    for name, accept in (
        ("high 만", {"high"}),
        ("high + medium", {"high", "medium"}),
        ("전부", {"high", "medium", "low", "?"}),
    ):
        kept = [(g, p) for g, p in rows if p.get("confidence") in accept]
        if not kept:
            continue
        pairs = [(g["label"], p.get("predicted")) for g, p in kept]
        acc = sum(1 for a, b in pairs if a == b) / len(pairs)
        macro, _ = macro_f1(pairs, labels)
        print(
            f"{name:>16}  {len(kept) / len(rows):>8.1%}  {acc:>7.1%}  {macro:>9.3f}"
        )

    # ── 인용 대조 ───────────────────────────────────────────────
    section("인용 미대조 — 원문에 없는 구절을 근거로 든 사례")
    bad = [(g, p) for g, p in rows if p.get("evidence_grounded") is False]
    print(f"{len(bad)}건 ({len(bad) / len(rows):.1%})")
    if bad:
        bad_correct = sum(1 for g, p in bad if g["label"] == p["predicted"])
        print(f"  이 중 라벨은 맞은 것 {bad_correct}건")
        ratios = [overlap_ratio(p.get("evidence", ""), g["request"]) for g, p in bad]
        print(
            "  원문과의 최장 연속 일치 비율: "
            + ", ".join(f"{r:.0%}" for r in sorted(ratios, reverse=True))
        )
        invented = sum(1 for r in ratios if r < 0.3)
        print(
            f"  거의 통째로 지어낸 것(30% 미만) {invented}건, "
            f"조각을 이어붙인 것 {len(ratios) - invented}건"
        )
        for g, p in bad[: args.show]:
            print(f"\n  [{g['serial']}] 정답 {g['label']} / 예측 {p['predicted']}")
            print(f"    모델 인용: {p.get('evidence', '')[:110]}")
            print(f"    원문 앞부분: {g['request'][:110]}")

    # ── 업권별 ──────────────────────────────────────────────────
    section("업권별 정확도")
    by_sector: dict[str, list[bool]] = defaultdict(list)
    for g, p in rows:
        by_sector[g.get("sector") or "미분류"].append(g["label"] == p.get("predicted"))
    for sector, hits in sorted(by_sector.items(), key=lambda kv: -len(kv[1])):
        if len(hits) < 3:
            continue
        print(f"  {sector:>12}  {len(hits):>3}건  {sum(hits) / len(hits):>6.1%}")

    # ── 혼동 사례 ───────────────────────────────────────────────
    section("가장 잦은 혼동의 실제 사례")
    confusion: Counter = Counter()
    for g, p in rows:
        if g["label"] != p.get("predicted"):
            confusion[(g["label"], p.get("predicted"))] += 1
    for (truth, guess), n in confusion.most_common(2):
        print(f"\n▸ {truth} → {guess} ({n}건)")
        shown = 0
        for g, p in rows:
            if g["label"] == truth and p.get("predicted") == guess:
                print(f"  [{g['serial']}] conf={p.get('confidence')}")
                print(f"    {g['request'][:150].replace(chr(10), ' ')}")
                shown += 1
                if shown >= args.show:
                    break


if __name__ == "__main__":
    main()
