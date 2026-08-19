"""선례를 따르면 틀릴 확률을 dev 에서 보정한다 — `trap_risk`.

## 왜 이것이 Router 의 심장인가

E5 가 보인 것: 최근접 선례를 따르는 전략은 순응 구간에서 정확도 1.000,
함정 구간에서 0.000 이다. 그러므로 이 도메인의 핵심 결정은 "무엇이 답인가" 가
아니라 **"지금 선례를 믿어도 되는가"** 다.

그 판단을 하려면 추론 시점에 알 수 있는 것만으로 위험을 추정해야 한다.
정답은 조건에 넣을 수 없다. 쓸 수 있는 것은 **유사도**와 **선례가 가리키는
라벨** 둘이다.

## leave-one-out 이 아니면 표가 거짓말을 한다

dev 안에서 최근접 선례를 찾을 때 자기 자신을 빼지 않으면 유사도 1.0 이 나오고
오류율은 0 이 된다. 그 표를 test 에 적용하면 "언제나 선례를 믿어라" 가 된다.
**LOO 는 선택이 아니라 정확성의 요건이다.**

## 실측이 설계를 바꿨다

처음 설계는 (유사도 구간 × 선례 라벨) 2차원 표를 쓰려 했다. 실제로 만들어 보니
칸 9개 중 5개가 1~2건이었다. 그 표로 문턱을 정하면 한두 건이 정책을 결정한다.

    유사도만            [0.00,0.15) 0.500 [0.364, 0.636]   n=48
                       [0.60,1.01) 0.062 [0.017, 0.201]   n=32   <- 구간이 안 겹친다
    선례 라벨만          비조치 0.214 · 기타 0.611 · 조치 0.455
    둘을 겹치면          칸 9개 중 5개가 n<=2

그래서 **유사도를 주 신호로 쓰고, 선례 라벨은 겹치지 않는 곳에서만 보조로
쓴다.** 자유 변수를 줄이는 쪽이 dev 85건에서 과적합을 덜 한다.

또 하나 — 유사도 분포가 **양극단에 몰려 있다.** 85건 중 80건이 0.15 미만이거나
0.60 이상이고, 가운데는 5건뿐이다. 그러므로 중간 구간의 문턱을 미세하게
조정하는 것은 의미가 없다. 세 구간(믿음 / 판단 유보 / 못 믿음)으로 족하다.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain import similarity as domain_similarity
from app.evaluation.confusable import cosine, idf_table, weighted_vector
from app.evaluation.metrics import wilson_interval

# 문턱은 도메인 한 곳에서만 정한다. 근거는 그 파일에 적혀 있고,
# 그 근거를 만드는 것이 바로 이 모듈이다.
TRUST = domain_similarity.TRUST
DOUBT = domain_similarity.DOUBT

BAND_TRUST, BAND_MIDDLE, BAND_DOUBT = "trust", "middle", "doubt"


def band_of(similarity: float) -> str:
    if similarity >= TRUST:
        return BAND_TRUST
    if similarity >= DOUBT:
        return BAND_MIDDLE
    return BAND_DOUBT


def loo_links(rows: list[dict], idf: dict[str, float]) -> list[dict]:
    """각 행의 최근접 선례를 **자기 자신을 빼고** 찾는다.

    동점은 앞선 것이 이긴다 — 난수를 쓰지 않아 재현된다.
    """
    vecs = [weighted_vector(r["request"], idf) for r in rows]
    out = []
    for i, row in enumerate(rows):
        best_j, best = -1, -1.0
        for j in range(len(rows)):
            if i == j:
                continue
            score = cosine(vecs[i], vecs[j])
            if score > best:
                best, best_j = score, j
        neighbor = rows[best_j] if best_j >= 0 else None
        out.append({
            "similarity": best if neighbor else 0.0,
            "band": band_of(best if neighbor else 0.0),
            "neighbor_label": neighbor["label"] if neighbor else None,
            "true_label": row["label"],
            "wrong": bool(neighbor) and neighbor["label"] != row["label"],
        })
    return out


def risk_table(links: list[dict]) -> dict:
    """구간별 '선례를 따르면 틀릴 확률'. 구간마다 신뢰구간을 함께 낸다."""
    table = {}
    for band in (BAND_TRUST, BAND_MIDDLE, BAND_DOUBT):
        group = [x for x in links if x["band"] == band]
        wrong = sum(1 for x in group if x["wrong"])
        lo, hi = wilson_interval(wrong, len(group))
        table[band] = {
            "n": len(group), "wrong": wrong,
            "risk": wrong / len(group) if group else None,
            "ci95": [lo, hi],
        }
    return table


def bands_are_separable(table: dict) -> tuple[bool, str]:
    """믿음 구간과 못 믿음 구간의 신뢰구간이 갈리는가.

    갈리지 않으면 유사도로 위험을 가를 수 없다는 뜻이고, 그러면 Router 의
    R5·R8 이 근거를 잃는다. **설계의 전제가 여기 걸려 있다.**
    """
    trust, doubt = table[BAND_TRUST], table[BAND_DOUBT]
    if not trust["n"] or not doubt["n"]:
        return False, "구간 하나가 비어 있다"
    if trust["ci95"][1] < doubt["ci95"][0]:
        return True, (f"믿음 상한 {trust['ci95'][1]:.3f} < 못믿음 하한 "
                      f"{doubt['ci95'][0]:.3f} — 겹치지 않는다")
    return False, (f"믿음 상한 {trust['ci95'][1]:.3f} · 못믿음 하한 "
                   f"{doubt['ci95'][0]:.3f} — 겹친다")


def by_neighbor_label(links: list[dict]) -> dict:
    """선례가 가리키는 라벨별 위험. 보조 신호로 쓸 수 있는지 보는 용도."""
    out = {}
    labels = {x["neighbor_label"] for x in links if x["neighbor_label"]}
    for label in sorted(labels):
        group = [x for x in links if x["neighbor_label"] == label]
        wrong = sum(1 for x in group if x["wrong"])
        lo, hi = wilson_interval(wrong, len(group))
        out[label] = {"n": len(group), "wrong": wrong,
                      "risk": wrong / len(group), "ci95": [lo, hi]}
    return out


def joint_cells(links: list[dict]) -> dict:
    """(구간 × 선례 라벨) 칸별 건수. **왜 2차원 표를 안 쓰는지**의 근거."""
    cells: dict[str, int] = {}
    for x in links:
        if x["neighbor_label"]:
            cells[f"{x['band']}|{x['neighbor_label']}"] = (
                cells.get(f"{x['band']}|{x['neighbor_label']}", 0) + 1)
    return cells


def calibrate(dev_rows: list[dict], idf: dict[str, float]) -> dict:
    """dev 에서 보정표를 만든다. test 는 열지 않는다."""
    links = loo_links(dev_rows, idf)
    table = risk_table(links)
    separable, detail = bands_are_separable(table)
    cells = joint_cells(links)
    return {
        "n_dev": len(dev_rows),
        "thresholds": {"trust": TRUST, "doubt": DOUBT},
        "overall_risk": sum(1 for x in links if x["wrong"]) / len(links),
        "by_band": table,
        "by_neighbor_label": by_neighbor_label(links),
        "joint_cell_counts": cells,
        "sparse_cells": sum(1 for v in cells.values() if v <= 2),
        "bands_separable": separable,
        "separability_detail": detail,
    }


def risk_of(table: dict, similarity: float) -> float:
    """보정표를 적용한다. 빈 구간은 전체 위험으로 되돌린다."""
    cell = table["by_band"].get(band_of(similarity))
    if not cell or cell["risk"] is None:
        return table["overall_risk"]
    return cell["risk"]


def main() -> None:
    import argparse

    from app.core.io import load_jsonl
    from app.core.paths import EVAL, PROCESSED, RESULTS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(EVAL / "nonaction_dev.jsonl"))
    ap.add_argument("--idf-source", default=str(PROCESSED / "cases_nonaction.jsonl"))
    ap.add_argument("--output", default=str(RESULTS / "trap_risk.json"))
    args = ap.parse_args()

    dev = [r for r in load_jsonl(Path(args.dev)) if r.get("label")]
    cases = load_jsonl(Path(args.idf_source))
    texts = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
             for c in cases]
    idf = idf_table([t for t in texts if t])

    report = calibrate(dev, idf)
    report["idf_source"] = f"{Path(args.idf_source).name} {len(cases)}건"

    print(f"dev {report['n_dev']}건 · LOO · 선례를 따랐을 때 전체 오류율 "
          f"{report['overall_risk']:.3f}\n")
    print(f"  {'구간':<8}{'건수':>5}{'오류':>5}{'위험':>8}   95% CI")
    for band in (BAND_TRUST, BAND_MIDDLE, BAND_DOUBT):
        c = report["by_band"][band]
        risk = f"{c['risk']:.3f}" if c["risk"] is not None else "—"
        print(f"  {band:<8}{c['n']:>5}{c['wrong']:>5}{risk:>8}   "
              f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]")

    print(f"\n  구간 분리 {'✅' if report['bands_separable'] else '❌'} — "
          f"{report['separability_detail']}")
    print(f"  2차원 칸 {len(report['joint_cell_counts'])}개 중 "
          f"{report['sparse_cells']}개가 2건 이하 — 그래서 유사도 1차원으로 간다")

    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
