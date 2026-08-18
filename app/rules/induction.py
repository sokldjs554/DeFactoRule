"""사실상 규칙 역추출 — dev 에서 해석 가능한 규칙 목록을 배운다.

이 프로젝트의 이름이 뜻하는 바가 여기 있다. 사례집 어디에도 "이런 경우에는
비조치" 라고 적혀 있지 않다. 결론만 있다. 그 결론들에서 **규칙을 되짚어**
꺼내는 것이 목표다.

  손으로 만든 어휘 규칙(app/rules/nonaction.py)   사람이 dev 를 읽고 지어냈다
  이 학습기                                        같은 dev 에서 기계가 찾는다

둘을 나란히 두는 이유는, 사람이 눈으로 찾은 것이 기계가 찾은 것보다 나은지
아닌지를 재기 위해서다. 둘 다 dev 만 본다.

## 방법

순차 피복(sequential covering)이다. CN2 계열의 가장 단순한 형태다.

  1. 아직 덮이지 않은 dev 사례에서, 한 라벨을 가장 순수하게 집어내는
     조건 결합을 빔 탐색으로 찾는다
  2. 그 규칙이 덮은 사례를 빼고 1을 반복한다
  3. 아무것도 문턱을 못 넘으면 멈추고, 나머지는 기본 라벨(다수)로 둔다

조건(atom)은 세 종류뿐이다. 전부 사람이 읽을 수 있어야 한다.

  문자 n-gram 포함     띄어쓰기가 무너진 문서가 많아 낱말 단위가 불안정하다
  업권 일치            사례집이 실제로 업권으로 묶여 있다
  길이 구간            장문 요청과 단문 요청은 성격이 다르다

## 과적합을 숨기지 않는다

dev 는 85건뿐이고 `조치` 는 그중 8건이다. 8건에서 찾은 규칙이 test 에서도
통할 것이라고 기대할 이유가 없다. 그래서 규칙마다 **dev 정밀도와 test
정밀도를 나란히** 적는다. 그 간극이 이 실험의 결과다 — 규칙이 잘 나오는 것이
결과가 아니라, 나온 규칙이 전이되는지가 결과다.

    python scripts/induce_rules.py --dev data/eval/nonaction_dev.jsonl \\
        --test data/eval/nonaction_test.jsonl --report experiments/results/e6_rules.json
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.core.io import load_jsonl, write_json, write_jsonl
from app.domain.labels import NON_ACTIONS

# ── 탐색 설정 ────────────────────────────────────────────────────
NGRAM_LENGTHS = (3, 4, 5, 6)
MIN_DF = 4            # 이보다 드문 n-gram 은 조건 후보로 쓰지 않는다
MAX_DF_RATIO = 0.60   # 이보다 흔하면 상투구다
MIN_SUPPORT = 4       # 규칙이 덮어야 할 최소 dev 건수
MIN_PRECISION = 0.80  # dev 라플라스 정밀도 문턱
MAX_DEPTH = 3         # 조건 결합의 최대 길이
BEAM = 12

JUNK = re.compile(r"\s+")


def squeeze(text: str) -> str:
    return JUNK.sub("", text or "")


def length_bucket(text: str) -> str:
    n = len(squeeze(text))
    if n < 200:
        return "짧음(<200자)"
    if n < 600:
        return "보통(200-600자)"
    return "긺(>=600자)"


@dataclass(frozen=True)
class Atom:
    kind: str   # ngram | sector | length
    value: str

    def holds(self, row: dict) -> bool:
        if self.kind == "ngram":
            return self.value in squeeze(row["request"])
        if self.kind == "sector":
            return (row.get("sector") or "미분류") == self.value
        return length_bucket(row["request"]) == self.value

    def describe(self) -> str:
        return {
            "ngram": f"본문에 '{self.value}' 포함",
            "sector": f"업권이 '{self.value}'",
            "length": f"길이 {self.value}",
        }[self.kind]


@dataclass
class Rule:
    atoms: tuple
    label: str
    dev_support: int = 0
    dev_correct: int = 0
    test_support: int = 0
    test_correct: int = 0
    order: int = 0
    notes: list = field(default_factory=list)

    def fires(self, row: dict) -> bool:
        return all(a.holds(row) for a in self.atoms)

    def describe(self) -> str:
        return " AND ".join(a.describe() for a in self.atoms) + f" -> {self.label}"

    @property
    def dev_precision(self) -> float:
        return self.dev_correct / self.dev_support if self.dev_support else 0.0

    @property
    def test_precision(self) -> float | None:
        return self.test_correct / self.test_support if self.test_support else None


def mine_atoms(rows: list[dict]) -> list[Atom]:
    """조건 후보를 dev 에서만 캔다."""
    n = len(rows)
    df: Counter = Counter()
    for row in rows:
        text = squeeze(row["request"])
        seen = set()
        for size in NGRAM_LENGTHS:
            for i in range(len(text) - size + 1):
                seen.add(text[i : i + size])
        df.update(seen)

    atoms = [
        Atom("ngram", g)
        for g, c in sorted(df.items())
        if MIN_DF <= c <= MAX_DF_RATIO * n
    ]
    atoms += [Atom("sector", s) for s in sorted({r.get("sector") or "미분류" for r in rows})]
    atoms += [Atom("length", b) for b in sorted({length_bucket(r["request"]) for r in rows})]
    return atoms


def maximal_form(term: str, texts: list[str], limit: int = 40) -> str:
    """지지도를 잃지 않는 한 좌우로 늘린다.

    문자 n-gram 은 어절 경계를 예사로 가로지른다. '것이전' 은 규칙처럼 보이지만
    실제로는 "…하는 **것이 전**자금융감독규정 제15조…" 라는 상투 인용구의 조각이다.
    지지도가 그대로인 한 늘려 보면 정체가 드러난다.

        '것이전'    -> '하는것이전자금융감독규정제1'   상투구
        '위반인지'  -> '위반인지여부'                  진짜 구절
        '망연계'    -> '망연계솔루션'                  특정 제품 (2건)

    표면 통계로는 이 셋을 가를 수 없었다 — 지지도도 출처 수도 업권 수도 쪽
    범위도 같았다. 늘려 놓으면 사람이 읽고 가릴 수 있다.
    """
    hits = [t for t in texts if term in t]
    base = len(hits)
    if not base:
        return term
    cur = term
    while len(cur) < limit:
        rights = {t[i + len(cur)] for t in hits
                  for i in [t.find(cur)] if 0 <= i < len(t) - len(cur)}
        grown = next((cur + c for c in sorted(rights)
                      if sum(1 for t in hits if cur + c in t) == base), None)
        if grown is None:
            lefts = {t[t.find(cur) - 1] for t in hits if t.find(cur) > 0}
            grown = next((c + cur for c in sorted(lefts)
                          if sum(1 for t in hits if c + cur in t) == base), None)
        if grown is None:
            break
        cur = grown
    return cur


def coverage_masks(rows: list[dict], atoms: list[Atom]) -> dict:
    """조건마다 '어느 사례를 덮는가' 를 정수 비트마스크로 만든다.

    순진하게 매번 문자열을 훑으면 조건 2,000개 × 빔 12 × 깊이 3 에서 분 단위가
    된다. 마스크로 바꾸면 결합이 AND 한 번이다.

    **덮는 집합이 완전히 같은 조건은 하나로 합친다.** 겹치는 n-gram 이
    ('에따른', '에따', '따른' …) 같은 규칙을 여러 벌 만들어 내는데, 그중
    가장 짧은 것 하나만 남긴다. 속도보다 해석 가능성 때문에 중요하다 —
    같은 규칙이 표현만 바꿔 여러 번 나오면 규칙 목록을 읽을 수 없다.
    """
    by_mask: dict[int, Atom] = {}
    for atom in atoms:
        mask = 0
        for i, row in enumerate(rows):
            if atom.holds(row):
                mask |= 1 << i
        if not mask:
            continue
        keep = by_mask.get(mask)
        if keep is None or (len(atom.value), atom.value) < (len(keep.value), keep.value):
            by_mask[mask] = atom

    # n-gram 대표는 최대 확장형으로 바꾼다. dev 피복은 그대로이므로 학습에는
    # 영향이 없고, 규칙을 읽을 수 있게 된다. test 에서는 더 좁게 걸리는데
    # 그것은 손해가 아니라 정직함이다 — 원래 그 조각은 더 긴 구절의 일부였다.
    texts = [squeeze(r["request"]) for r in rows]
    out = {}
    for mask, atom in by_mask.items():
        if atom.kind == "ngram":
            atom = Atom("ngram", maximal_form(atom.value, texts))
        out[atom] = mask
    return out


def popcount(x: int) -> int:
    # int.bit_count() 는 3.10+ 다. 3.9 를 지원하는 동안은 이쪽을 쓴다.
    return bin(x).count("1")


def laplace(correct: int, covered: int, n_classes: int) -> float:
    """CN2 의 라플라스 추정. 3건 전부 맞은 규칙이 30건 중 28건보다 낫게 보이는 것을 막는다.

    **문턱으로는 쓰지 않는다.** laplace(4, 4, 3) = 0.714 라서, 문턱을 0.80 으로
    두면 dev 에 8건뿐인 `조치` 클래스는 완벽한 규칙을 찾아도 통과할 수 없다.
    소수 클래스를 구조적으로 배제하는 문턱은 문턱이 아니라 버그다.
    순위 매기기에만 쓰고, 통과 여부는 원 정밀도와 최소 피복으로 판단한다.
    """
    return (correct + 1) / (covered + n_classes)


def best_rule(
    rows: list[dict],
    masks: dict,
    live: int,
    labels: tuple,
    min_support: int,
    min_precision: float,
    max_depth: int,
) -> Rule | None:
    """빔 탐색으로 조건 결합 하나를 찾는다. 동점은 사전순으로 갈라 재현된다."""
    label_masks = {}
    for lab in labels:
        m = 0
        for i, row in enumerate(rows):
            if row["label"] == lab:
                m |= 1 << i
        label_masks[lab] = m

    beam: list[tuple] = [((), live)]
    best: Rule | None = None
    best_score = -1.0

    for _ in range(max_depth):
        scored = []
        for prefix, pmask in beam:
            for atom, amask in masks.items():
                if atom in prefix:
                    continue
                cmask = pmask & amask
                covered = popcount(cmask)
                if covered < min_support:
                    continue
                hits = [(popcount(cmask & label_masks[lab]), lab) for lab in labels]
                correct, label = max(hits, key=lambda h: (h[0], h[1]))
                combo = prefix + (atom,)
                scored.append((
                    laplace(correct, covered, len(labels)),
                    covered,
                    tuple(sorted(a.value for a in combo)),
                    combo, cmask, label, correct,
                ))
        if not scored:
            break
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        for score, covered, _, combo, _cmask, label, correct in scored[:BEAM]:
            if correct / covered >= min_precision and score > best_score:
                best, best_score = (
                    Rule(atoms=combo, label=label,
                         dev_support=covered, dev_correct=correct),
                    score,
                )
        beam = [(x[3], x[4]) for x in scored[:BEAM]]
    return best


def induce(
    rows: list[dict],
    labels: tuple = NON_ACTIONS,
    min_support: int = MIN_SUPPORT,
    min_precision: float = MIN_PRECISION,
    max_depth: int = MAX_DEPTH,
) -> tuple[list[Rule], str]:
    """순차 피복. 규칙 목록과 기본 라벨을 돌려준다."""
    masks = coverage_masks(rows, mine_atoms(rows))
    live = (1 << len(rows)) - 1
    rules: list[Rule] = []

    while live:
        rule = best_rule(rows, masks, live, labels,
                         min_support, min_precision, max_depth)
        if rule is None:
            break
        rule.order = len(rules) + 1
        rules.append(rule)
        covered = live
        for atom in rule.atoms:
            covered &= masks[atom]
        live &= ~covered

    leftover = [r for i, r in enumerate(rows) if live >> i & 1]
    default = Counter(r["label"] for r in (leftover or rows)).most_common(1)[0][0]
    return rules, default


def apply_rules(rules: list[Rule], default: str, row: dict) -> tuple[str, str, str]:
    """(라벨, 근거 규칙, 신뢰도). 먼저 걸리는 규칙이 이긴다."""
    for rule in rules:
        if rule.fires(row):
            conf = "high" if rule.dev_precision >= 0.9 else "medium"
            return rule.label, f"rule{rule.order}", conf
    return default, "default", "low"


def measure_on(rules: list[Rule], rows: list[dict]) -> None:
    """test 에서 각 규칙이 몇 건을 덮고 몇 건을 맞혔는지 채운다.

    규칙 목록의 순서를 존중한다 — 앞선 규칙이 덮은 사례는 뒤 규칙에 가지 않는다.
    그래야 실제 적용 방식과 같은 숫자가 나온다.
    """
    for rule in rules:
        rule.test_support = rule.test_correct = 0
    for row in rows:
        for rule in rules:
            if rule.fires(row):
                rule.test_support += 1
                rule.test_correct += rule.label == row["label"]
                break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", help="test 예측을 JSONL 로 저장")
    ap.add_argument("--report")
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    args = ap.parse_args()

    dev = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]
    test = [r for r in load_jsonl(Path(args.test)) if r.get("label")]

    print(f"dev {len(dev)}건에서 규칙을 찾는다 "
          f"(최소 피복 {args.min_support} · 최소 정밀도 {args.min_precision} "
          f"· 최대 결합 {args.max_depth})")
    print("  dev 라벨 분포: " + ", ".join(
        f"{k} {v}" for k, v in Counter(r["label"] for r in dev).most_common()))

    rules, default = induce(dev, min_support=args.min_support,
                            min_precision=args.min_precision,
                            max_depth=args.max_depth)
    measure_on(rules, test)

    print(f"\n규칙 {len(rules)}개 · 기본 라벨 '{default}'\n")
    if not rules:
        print("  문턱을 넘는 규칙이 하나도 없다.")
    print(
        f"{'#':>2}  {'라벨':>5}  {'dev 피복':>8}  {'dev 정밀도':>10}  "
        f"{'test 피복':>9}  {'test 정밀도':>11}"
    )
    for rule in rules:
        tp = rule.test_precision
        tp_s = f"{tp:>11.1%}" if tp is not None else f"{'—':>11}"
        print(f"{rule.order:>2}  {rule.label:>5}  {rule.dev_support:>8}  "
              f"{rule.dev_precision:>10.1%}  {rule.test_support:>9}  {tp_s}")
        print(f"      {rule.describe()}")

    # ── 전이 여부 ────────────────────────────────────────────────
    fired = [r for r in rules if r.test_support > 0]
    if fired:
        dev_avg = sum(r.dev_precision for r in fired) / len(fired)
        test_avg = sum(r.test_precision for r in fired) / len(fired)
        print(f"\n{'─' * 74}")
        print(f"test 에서 실제로 발화한 규칙 {len(fired)}/{len(rules)}개")
        print(f"  평균 정밀도  dev {dev_avg:.1%}  ->  test {test_avg:.1%}"
              f"  (간극 {dev_avg - test_avg:+.1%}p)")
        dead = len(rules) - len(fired)
        if dead:
            print(f"  ⚠ {dead}개 규칙은 test 에서 한 번도 발화하지 않았다 — dev 전용 우연이다.")

        # ── 클래스별 전이 ────────────────────────────────────────
        print(f"\n{'라벨':>6}  {'규칙':>4}  {'dev 정밀도':>10}  {'test 정밀도':>11}  {'간극':>8}")
        by_label: dict[str, list] = {}
        for rule in fired:
            by_label.setdefault(rule.label, []).append(rule)
        for label in NON_ACTIONS:
            group = by_label.get(label)
            if not group:
                print(f"{label:>6}  {0:>4}  {'—':>10}  {'—':>11}  {'규칙 없음':>8}")
                continue
            d = sum(r.dev_correct for r in group) / sum(r.dev_support for r in group)
            t = sum(r.test_correct for r in group) / sum(r.test_support for r in group)
            print(f"{label:>6}  {len(group):>4}  {d:>10.1%}  {t:>11.1%}  {d - t:>+8.1%}p")
        print("\n  같은 dev 정밀도라도 클래스마다 전이가 다르다. 소수 클래스 규칙이")
        print("  무너진다면, 그 클래스의 신호가 본문 표면에 없다는 뜻이다.")

    if args.output:
        preds = []
        for row in test:
            label, rule, conf = apply_rules(rules, default, row)
            preds.append({
                "source": row["source"], "serial": row["serial"], "page": row["page"],
                "pair_index": row.get("pair_index", 1),
                "predicted": label, "rule": rule, "confidence": conf,
            })
        write_jsonl(Path(args.output), preds)
        print(f"\n-> {args.output} ({len(preds)}건)")
        print("  신뢰도: " + ", ".join(
            f"{k} {v}" for k, v in Counter(p["confidence"] for p in preds).most_common()))

    if args.report:
        write_json(args.report, {
            "settings": {
                "ngram_lengths": list(NGRAM_LENGTHS), "min_df": MIN_DF,
                "max_df_ratio": MAX_DF_RATIO, "min_support": args.min_support,
                "min_precision": args.min_precision, "max_depth": args.max_depth,
                "beam": BEAM,
            },
            "default_label": default,
            "rules": [{
                "order": r.order, "label": r.label, "description": r.describe(),
                "atoms": [{"kind": a.kind, "value": a.value} for a in r.atoms],
                "dev_support": r.dev_support, "dev_precision": r.dev_precision,
                "test_support": r.test_support, "test_precision": r.test_precision,
            } for r in rules],
        })
        print(f"-> {args.report}")


if __name__ == "__main__":
    main()


# ══ 교차검증 기반 규칙 선별 ═══════════════════════════════════════
#
# dev 순도로 규칙을 고르면 우연이 신호를 이긴다. 실측으로 확인했다.
#
#   '것이전'   dev 5/5 (100%) · laplace 0.750 -> test 20.0%   우연
#   '위반인지' dev 4/5 ( 80%) · laplace 0.625 -> test 47.4%   신호
#
# 순차 피복은 laplace 가 높은 쪽을 먼저 집으므로 '것이전' 을 골랐다. dev 안에서
# 완벽한 규칙을 찾는 것은 표본이 작을수록 쉽고, 그 쉬움이 곧 함정이다.
#
# 그래서 **dev 안에서 다시 나눠** 규칙이 보류 조각에서도 버티는지 본다.
# test 는 여전히 건드리지 않는다.

FOLDS = 5


def fold_of(index: int, folds: int) -> int:
    """난수 없이 조각을 나눈다. 정렬이 고정되어 있으므로 재현된다."""
    return index % folds


def cross_validate(
    rows: list[dict],
    labels: tuple = NON_ACTIONS,
    folds: int = FOLDS,
    min_support: int = MIN_SUPPORT,
    min_precision: float = MIN_PRECISION,
    max_depth: int = MAX_DEPTH,
) -> dict:
    """조각마다 나머지에서 배우고 그 조각에서 잰다.

    같은 조건 결합(atom tuple)이 여러 조각에서 반복해서 나오고, 보류 조각에서도
    정밀도를 지키면 그 규칙은 믿을 만하다. 한 조각에서만 나왔다면 그 조각의
    우연이다.
    """
    stats: dict[tuple, dict] = {}
    for f in range(folds):
        train = [r for i, r in enumerate(rows) if fold_of(i, folds) != f]
        held = [r for i, r in enumerate(rows) if fold_of(i, folds) == f]
        if not train or not held:
            continue
        learned, _ = induce(train, labels, min_support, min_precision, max_depth)
        for rule in learned:
            key = tuple(sorted((a.kind, a.value) for a in rule.atoms))
            slot = stats.setdefault(key, {
                "label": rule.label, "folds": 0,
                "oof_support": 0, "oof_correct": 0,
                "describe": rule.describe(),
            })
            slot["folds"] += 1
            for row in held:
                if rule.fires(row):
                    slot["oof_support"] += 1
                    slot["oof_correct"] += row["label"] == rule.label
    for slot in stats.values():
        slot["oof_precision"] = (
            slot["oof_correct"] / slot["oof_support"] if slot["oof_support"] else None
        )
    return stats


def induce_validated(
    rows: list[dict],
    labels: tuple = NON_ACTIONS,
    folds: int = FOLDS,
    min_support: int = MIN_SUPPORT,
    min_precision: float = MIN_PRECISION,
    max_depth: int = MAX_DEPTH,
    min_oof_support: int = 3,
    min_oof_precision: float = 0.50,
) -> tuple[list[Rule], str, dict]:
    """전체 dev 에서 규칙을 배우되, 교차검증을 통과한 것만 남긴다.

    통과 조건은 두 가지다. 보류 조각에서 **실제로 발화했고**(min_oof_support),
    거기서도 정밀도를 지켰을 것(min_oof_precision).

    문턱을 0.50 으로 둔 이유는, 소수 클래스 기저율이 8~9% 라 정밀도 50% 도
    기저율 대비 여섯 배이기 때문이다. dev 순도 100% 를 요구하는 것과는 정반대
    방향의 기준이다 — 완벽함이 아니라 **버티는가**를 본다.
    """
    stats = cross_validate(rows, labels, folds, min_support, min_precision, max_depth)
    trusted = {
        key for key, s in stats.items()
        if s["oof_support"] >= min_oof_support
        and s["oof_precision"] is not None
        and s["oof_precision"] >= min_oof_precision
    }

    full, default = induce(rows, labels, min_support, min_precision, max_depth)
    kept = []
    for rule in full:
        key = tuple(sorted((a.kind, a.value) for a in rule.atoms))
        if key in trusted:
            rule.notes.append(
                f"교차검증 보류 정밀도 {stats[key]['oof_precision']:.1%} "
                f"({stats[key]['oof_correct']}/{stats[key]['oof_support']})"
            )
            rule.order = len(kept) + 1
            kept.append(rule)
    return kept, default, stats
