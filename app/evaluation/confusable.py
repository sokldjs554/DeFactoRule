"""표면선례 함정 정확도 (TRAP) — 이 프로젝트를 위해 만든 지표.

매크로 F1 도 AURC 도 기성품이고, 둘 다 "문서에 적혀 있지 않은 판단 기준을
복원했는가" 라는 질문에는 답하지 못한다.

착상은 이렇다. 이 도메인에서 가장 손쉬운 전략은 **닮은 선례를 그대로 따라가는
것**이다. 실제로 요청문이 거의 같은 사안들이 사례집에 반복해서 나온다. 그런데
그중 일부는 결론이 갈린다 — 표면에 적혀 있지 않은 무언가 때문에.

그래서 test 사례마다 dev 에서 **가장 닮은 선례**를 찾고 두 무리로 가른다.

    순응(AGREE)   최근접 선례의 결론이 정답과 같다 — 따라가면 맞는다
    함정(TRAP)    최근접 선례의 결론이 정답과 다르다 — 따라가면 틀린다

TRAP 위에서의 정확도가 이 지표다. 표면 유사도를 그대로 베끼는 전략은 여기서
**구조적으로 0%** 다. 그러므로 TRAP 정확도는 "표면 너머를 읽었는가" 의 직접
측정이 된다.

최근접 선례는 **dev 에서만** 찾는다. test 끼리 이웃을 찾으면 정답끼리
정보를 주고받게 된다.

유사도는 문자 4-gram 의 IDF 가중 코사인이다. 단순 자카드를 쓰면 이 문서군에
공통으로 깔린 조판 상투구 때문에 전부 비슷해 보인다. IDF 는 라벨을 쓰지
않으므로 코퍼스 전체(1,095건)에서 뽑아도 누출이 아니다.

    python scripts/confusable.py --dev data/eval/nonaction_dev.jsonl \\
        --gold data/eval/nonaction_test.jsonl
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

from app.core.io import key_of, load_jsonl, write_json
from app.core.paths import PROCESSED

NGRAM = 4

# 이보다 안 닮았으면 "선례를 따라간다" 는 전제 자체가 없다. 그런 사례는
# 두 무리 어디에도 넣지 않고 따로 센다.
SIMILARITY_FLOOR = 0.25

JUNK = re.compile(r"[\s​﻿]+")


def normalize(text: str) -> str:
    """공백만 걷어낸다. 글자는 건드리지 않는다."""
    return JUNK.sub("", text or "")


def ngrams(text: str, n: int = NGRAM) -> Counter:
    s = normalize(text)
    if len(s) < n:
        return Counter([s] if s else [])
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def idf_table(texts: list[str], n: int = NGRAM) -> dict[str, float]:
    """문서빈도 역수. 라벨을 쓰지 않으므로 전체 코퍼스에서 뽑아도 된다."""
    df: Counter = Counter()
    for t in texts:
        df.update(set(ngrams(t, n)))
    total = max(1, len(texts))
    return {g: math.log(total / (1 + c)) + 1.0 for g, c in df.items()}


def weighted_vector(text: str, idf: dict[str, float], n: int = NGRAM) -> dict[str, float]:
    vec = {g: c * idf.get(g, 1.0) for g, c in ngrams(text, n).items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {g: v / norm for g, v in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(g, 0.0) for g, v in a.items())


def nearest(
    rows: list[dict], neighbors: list[dict], idf: dict[str, float]
) -> list[dict]:
    """각 행에 대해 이웃 목록에서 가장 닮은 하나를 찾는다.

    동점은 이웃의 등장 순서로 가른다 — 난수를 쓰지 않아 재현된다.
    """
    nvecs = [weighted_vector(r["request"], idf) for r in neighbors]
    out = []
    for row in rows:
        v = weighted_vector(row["request"], idf)
        best_i, best_sim = -1, -1.0
        for i, nv in enumerate(nvecs):
            sim = cosine(v, nv)
            if sim > best_sim:
                best_i, best_sim = i, sim
        out.append({
            "row": row,
            "neighbor": neighbors[best_i] if best_i >= 0 else None,
            "similarity": best_sim if best_i >= 0 else 0.0,
        })
    return out


def partition(links: list[dict], floor: float = SIMILARITY_FLOOR) -> dict[str, list]:
    """순응 / 함정 / 이웃 없음 으로 가른다."""
    groups = {"agree": [], "trap": [], "unanchored": []}
    for link in links:
        nb = link["neighbor"]
        if nb is None or link["similarity"] < floor:
            groups["unanchored"].append(link)
        elif nb["label"] == link["row"]["label"]:
            groups["agree"].append(link)
        else:
            groups["trap"].append(link)
    return groups


def anchoring_by_class(links: list[dict], floor: float) -> dict[str, dict]:
    """클래스별로 '닮은 선례가 존재하는가' 를 센다.

    이 프로젝트에서 가장 중요한 진단이 여기서 나온다. 전체 앵커링 비율만 보면
    괜찮아 보이는데, 클래스별로 쪼개면 소수 클래스에 선례가 아예 없을 수 있다.
    그러면 검색 기반 접근은 **평균은 좋고 정작 필요한 곳에서는 무력하다.**
    """
    out: dict[str, dict] = {}
    for link in links:
        label = link["row"]["label"]
        slot = out.setdefault(label, {"n": 0, "anchored": 0, "trap": 0, "max_sim": 0.0})
        slot["n"] += 1
        slot["max_sim"] = max(slot["max_sim"], link["similarity"])
        if link["neighbor"] is not None and link["similarity"] >= floor:
            slot["anchored"] += 1
            if link["neighbor"]["label"] != label:
                slot["trap"] += 1
    for slot in out.values():
        slot["anchor_rate"] = slot["anchored"] / slot["n"] if slot["n"] else 0.0
    return out


def accuracy(links: list[dict], pred: dict) -> tuple[float, int, int]:
    scored = [
        link for link in links if key_of(link["row"]) in pred
        and pred[key_of(link["row"])].get("predicted") is not None
    ]
    hit = sum(
        1 for link in scored
        if pred[key_of(link["row"])]["predicted"] == link["row"]["label"]
    )
    return (hit / len(scored) if scored else 0.0), hit, len(scored)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, help="test 평가셋")
    ap.add_argument("--dev", required=True, help="선례를 찾을 dev 집합")
    ap.add_argument("--pred", action="append", metavar="NAME=PATH",
                    help="생략하면 data/processed 의 예측 전부")
    ap.add_argument("--corpus", default=str(PROCESSED / "cases_nonaction.jsonl"))
    ap.add_argument("--floor", type=float, default=SIMILARITY_FLOOR)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--report")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(Path(args.gold)) if r.get("label")]
    dev = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]

    corpus = Path(args.corpus)
    if corpus.exists():
        texts = [(c["fields"].get("요청대상행위") or "") for c in load_jsonl(corpus)]
        source = f"{corpus.name} {len(texts)}건"
    else:
        texts = [r["request"] for r in rows + dev]
        source = f"평가셋 {len(texts)}건 (코퍼스 없음)"
    idf = idf_table(texts)

    links = nearest(rows, dev, idf)
    groups = partition(links, args.floor)
    n_anchored = len(groups["agree"]) + len(groups["trap"])

    print(f"IDF 출처: {source} · 문자 {NGRAM}-gram · 유사도 하한 {args.floor}\n")
    print(f"test {len(rows)}건을 최근접 dev 선례({len(dev)}건) 기준으로 가른다\n")
    print(f"  순응 AGREE    {len(groups['agree']):>4}  선례를 따라가면 맞는다")
    print(f"  함정 TRAP     {len(groups['trap']):>4}  선례를 따라가면 틀린다")
    print(f"  이웃 없음      {len(groups['unanchored']):>4}  유사도 {args.floor} 미만")
    if n_anchored:
        print(f"\n  1-NN 이 순응 집합에서 내는 정확도: {len(groups['agree']) / n_anchored:.1%}")

    # ── 가장 중요한 진단 ────────────────────────────────────────
    table = anchoring_by_class(links, args.floor)
    print(f"\n{'─' * 74}\n클래스별로 선례가 존재하는가 — 검색 접근의 사각지대\n")
    print(
        f"{'라벨':>8}  {'건수':>4}  {'선례 있음':>9}  {'비율':>7}  "
        f"{'그중 함정':>9}  {'최대 유사도':>10}"
    )
    for label, slot in sorted(table.items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"{label:>8}  {slot['n']:>4}  {slot['anchored']:>9}  "
            f"{slot['anchor_rate']:>7.1%}  {slot['trap']:>9}  {slot['max_sim']:>10.3f}"
        )
    blind = [lab for lab, slot in table.items() if slot["anchored"] == 0]
    if blind:
        print(
            f"\n  ⚠ {', '.join(blind)} 클래스는 닮은 선례가 **하나도** 없다.\n"
            "     검색 기반 접근은 이 클래스에서 원리적으로 무력하다 — 평균 성능은\n"
            "     멀쩡해 보이는데 정작 판별이 필요한 곳에서 아무것도 못 한다."
        )

    if not groups["trap"]:
        raise SystemExit("\n함정 사례가 없습니다. 하한을 낮추거나 평가셋을 확인하세요.")

    specs = args.pred or [
        f"{p.stem.replace('pred_nonaction_', '')}={p}"
        for p in sorted(PROCESSED.glob("pred_nonaction_*.jsonl"))
    ]
    models = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"형식은 NAME=PATH 입니다: {spec}")
        models[name] = {key_of(r): r for r in load_jsonl(Path(path))}

    print(f"\n{'─' * 74}")
    print(f"{'모델':>10}  {'전체':>16}  {'순응':>16}  {'함정 TRAP':>16}  {'격차':>7}")
    results = {}
    for name, pred in models.items():
        overall, oh, on = accuracy(links, pred)
        agree, ah, an = accuracy(groups["agree"], pred)
        trap, th, tn = accuracy(groups["trap"], pred)
        results[name] = {
            "overall": overall, "overall_n": on,
            "agree": agree, "agree_n": an,
            "trap": trap, "trap_n": tn, "trap_hits": th,
            "gap": agree - trap,
        }
        print(
            f"{name:>10}  {overall:>7.1%} ({oh:>3}/{on:<3})  "
            f"{agree:>7.1%} ({ah:>3}/{an:<3})  "
            f"{trap:>7.1%} ({th:>3}/{tn:<3})  {agree - trap:>+7.1%}"
        )

    print("\n  TRAP = 가장 닮은 선례를 따라가면 틀리는 사례에서의 정확도")
    print("  선례를 그대로 베끼는 전략(1-NN)은 여기서 구조적으로 0% 다.")
    print("  격차가 크다는 것은 그 모델이 표면 유사도에 기대고 있다는 뜻이다.")
    print("\n  ⚠ 다수 클래스만 찍는 전략은 TRAP 에서 0% 가 아니다. 함정의 정답이")
    print("     다수 클래스인 경우가 있기 때문이다. TRAP 만으로 '표면 너머를")
    print("     읽었다' 고 말할 수 없고, majority 기준선과 함께 읽어야 한다.")

    n_trap = len(groups["trap"])
    if n_trap < 30:
        print(
            f"\n  ⚠ 함정 사례가 {n_trap}건뿐이다. 이 수로는 모델 순위를 매길 수 없다.\n"
            "     한 건이 뒤집히면 지표가 "
            f"{1 / n_trap:.1%}p 움직인다. 아래 숫자는 순위가 아니라\n"
            "     '어느 모델이 선례를 그대로 베끼는가' 의 정성적 신호로만 읽을 것."
        )

    if args.examples:
        print(f"\n{'─' * 74}\n함정 사례 — 닮은 선례와 결론이 갈린 지점\n")
        for link in sorted(groups["trap"], key=lambda x: -x["similarity"])[: args.examples]:
            row, nb = link["row"], link["neighbor"]
            print(f"  유사도 {link['similarity']:.3f}")
            print(f"    정답  [{row['label']}] {row.get('sector') or '?'} · "
                  f"{normalize(row['request'])[:60]}")
            print(f"    선례  [{nb['label']}] {nb.get('sector') or '?'} · "
                  f"{normalize(nb['request'])[:60]}")
            marks = []
            for name, pred in models.items():
                p = (pred.get(key_of(row)) or {}).get("predicted", "—")
                marks.append(f"{'○' if p == row['label'] else '×'} {name}:{p}")
            print("      " + "  ".join(marks) + "\n")

    if args.report:
        write_json(args.report, {
            "metric": "TRAP",
            "ngram": NGRAM,
            "similarity_floor": args.floor,
            "idf_source": source,
            "counts": {k: len(v) for k, v in groups.items()},
            "anchoring_by_class": table,
            "one_nn_overall": len(groups["agree"]) / n_anchored if n_anchored else 0.0,
            "models": results,
        })
        print(f"-> {args.report}")


if __name__ == "__main__":
    main()
