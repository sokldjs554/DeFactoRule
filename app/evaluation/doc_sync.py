"""문서의 파생 수치를 산출물에서 다시 써 넣는다.

## 왜 필요한가

문서 수치를 손으로 고치면 반드시 밀린다. 실제로 `docs/15` 는 규칙 학습기를
고친 뒤 갱신을 놓쳐 **판정이 뒤집힌 문장**을 담은 채 남아 있었고, README 는
레지스트리를 46건·테스트를 154개로 적고 있었다(실제 53건·252개).

회귀 테스트(`test_documented_numbers.py`)가 어긋남을 잡아 주지만, 잡힌 뒤
**고치는 일이 손 작업이면 결국 같은 자리로 돌아온다.** 그래서 고치는 쪽도
자동으로 만든다.

    python3 scripts/sync_docs.py --check   무엇이 어긋났는지만 본다
    python3 scripts/sync_docs.py           문서를 다시 써 넣는다

원본은 항상 산출물이다. 문서는 그것을 비추는 것일 뿐이다.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict

from app.core.paths import ROOT

COUNT_UNITS = {"건", "개", "쌍", "회"}
LAYER_ORDER = ("extraction", "labeling", "retrieval", "evaluation",
               "agent", "infrastructure")


def registry() -> list[dict]:
    path = ROOT / "data/failures/registry.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def collected_tests() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    match = re.search(r"(\d+) tests? collected", out.stdout)
    if not match:
        raise RuntimeError("테스트 수를 셀 수 없습니다:\n" + out.stdout[-400:])
    return int(match.group(1))


def load_report(name: str) -> dict | None:
    path = ROOT / "experiments" / "results" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def corpus_counts() -> tuple[int, int]:
    """(사례 수, 질의–회답 쌍 수). 산출물이 없으면 (0, 0)."""
    from app.core.io import load_jsonl
    from app.core.paths import PROCESSED

    try:
        cases = len(load_jsonl(PROCESSED / "cases_nonaction.jsonl")) + len(
            load_jsonl(PROCESSED / "cases_interpretation.jsonl"))
        pairs = len(load_jsonl(PROCESSED / "qa_pairs.jsonl"))
    except FileNotFoundError:
        return 0, 0
    return cases, pairs


def significant(d: dict) -> int:
    return sum(1 for c in d["comparisons"] if c["significant_holm"])


def verdict_of(c: dict) -> str:
    if c["significant_holm"]:
        return "**유의**"
    return "보정 후 탈락" if c["significant_raw"] else "판정 보류"


# ── 표 생성 ──────────────────────────────────────────────────────
def point_table(data: dict, header: str) -> str:
    lines = [f"| 모델 | {header} |", "|---|---|"]
    lines += [f"| `{k}` | {v:.3f} |" for k, v in sorted(data.items(), key=lambda kv: -kv[1])]
    return "\n".join(lines)


def aurc_table(rc: dict) -> str:
    lines = ["| 모델 | AURC | 최저 커버리지에서의 위험 | 운영점 |", "|---|---|---|---|"]
    for k, v in sorted(rc["aurc"].items(), key=lambda kv: kv[1]):
        pts = rc["curves"][k]
        top = min(pts, key=lambda p: p["coverage"])
        cell = "— (한 점)" if len(pts) == 1 else f"{top['coverage']:.1%} 에서 {top['risk']:.1%}"
        lines.append(f"| `{k}` | {v:.3f} | {cell} | {len(pts)} |")
    return "\n".join(lines)


def comparison_table(data: dict) -> str:
    lines = ["| 비교 | 차이 | 95% CI | p | p(Holm) | 판정 |", "|---|---|---|---|---|---|"]
    for c in sorted(data["comparisons"], key=lambda x: x["p_approx"]):
        lo, hi = c["ci95"]
        lines.append(
            f"| {c['a']} − {c['b']} | {c['diff']:+.3f} | {lo:+.3f} – {hi:+.3f} | "
            f"{c['p_approx']:.3f} | {c['p_holm']:.3f} | {verdict_of(c)} |"
        )
    return "\n".join(lines)


def anchoring_block(trap: dict) -> str:
    """검색의 사각지대 — 클래스별로 닮은 선례가 있는 비율."""
    rows = ["| 정답 | test 건수 | dev 에 닮은 선례가 있는 건수 | 비율 |", "|---|---|---|---|"]
    for label, v in sorted(trap["anchoring_by_class"].items(), key=lambda kv: kv[1]["anchor_rate"]):
        rows.append(f"| `{label}` | {v['n']} | {v['anchored']} | **{v['anchor_rate']:.1%}** |")
    return "\n".join(rows)


def trap_block(trap: dict) -> str:
    """순응/함정 구간별 정확도. 표면 유사도를 베끼는 전략은 함정에서 0% 다."""
    c = trap["counts"]
    head = (
        f"순응 {c['agree']}건 · 함정 {c['trap']}건 · 선례 없음 {c['unanchored']}건 "
        f"(닮음 문턱 {trap['similarity_floor']}, 문자 {trap['ngram']}-gram IDF 코사인)"
    )
    rows = [
        f"| 모델 | 전체 | 순응 {c['agree']}건 | 함정 {c['trap']}건 (TRAP) | 격차 |",
        "|---|---|---|---|---|",
    ]
    for name, v in sorted(trap["models"].items(), key=lambda kv: -kv[1]["trap"]):
        rows.append(
            f"| `{name}` | {v['overall']:.3f} | {v['agree']:.3f} | "
            f"**{v['trap']:.3f}** | {v['gap']:.3f} |"
        )
    return head + "\n\n" + "\n".join(rows)


def registry_tables() -> tuple[str, str]:
    reg = registry()
    by_layer: dict[str, Counter] = defaultdict(Counter)
    for c in reg:
        by_layer[c["layer"]][c["category"]] += 1
    tax = ["| 계층 | 건수 | 범주 |", "|---|---|---|"]
    for layer in LAYER_ORDER:
        counts = by_layer.get(layer)
        if counts:
            cats = " · ".join(f"{k} {v}" for k, v in counts.most_common())
            tax.append(f"| {layer} | {sum(counts.values())} | {cats} |")

    def cell(m: dict) -> str:
        unit = (m.get("unit") or "").strip()
        ratio = unit not in COUNT_UNITS

        def num(v):
            if not isinstance(v, (int, float)):
                return str(v)
            return f"{v:.3f}" if (ratio and isinstance(v, float)) else f"{v:g}"

        if (m.get("note") or "").startswith("복구가 아니라"):
            return f"{num(m['before'])} (통과 문턱 {num(m['after'])})"
        body = f"{num(m['before'])} → {num(m['after'])}"
        return f"{body}{unit}" if unit in COUNT_UNITS else body

    def label(m: dict) -> str:
        unit = (m.get("unit") or "").strip()
        name = m["name"]
        return f"{name} ({unit})" if unit and unit not in COUNT_UNITS and unit not in name else name

    metrics = ["| ID | 지표 | 전 → 후 | 종류 |", "|---|---|---|---|"]
    for c in sorted((x for x in reg if x.get("metric")),
                    key=lambda x: (x["id"][:2], int(x["id"].split("-")[1]))):
        m = c["metric"]
        metrics.append(f"| {c['id']} | {label(m)} | {cell(m)} | {m['kind']} |")
    return "\n".join(tax), "\n".join(metrics)


# ── 써 넣기 ──────────────────────────────────────────────────────
def replace_marked(text: str, tag: str, body: str) -> str:
    begin, end = f"<!-- {tag}:시작 -->", f"<!-- {tag}:끝 -->"
    if begin not in text or end not in text:
        raise LookupError(f"{tag} 구획 표시가 없습니다")
    head = text[: text.index(begin) + len(begin)]
    tail = text[text.index(end):]
    return f"{head}\n{body}\n{tail}"


def sync(check_only: bool = False) -> list[str]:
    """바뀐 파일 목록을 돌려준다. check_only 면 쓰지 않는다."""
    changed = []
    reg = registry()
    n_cases = len(reg)
    n_probes = sum(1 for c in reg if c.get("probe"))
    n_metrics = sum(1 for c in reg if c.get("metric"))
    f1, rc = load_report("e7_all_models.json"), load_report("e7_risk_coverage.json")

    # README
    path = ROOT / "README.md"
    text = original = path.read_text(encoding="utf-8")
    text = re.sub(r"<!--TESTS-->\d+<!--/TESTS-->",
                  f"<!--TESTS-->{collected_tests()}<!--/TESTS-->", text)
    cases, pairs = corpus_counts()
    if cases:
        text = re.sub(r"\d{1,3}(?:,\d{3})*건 — W1 게이트 통과",
                      f"{cases:,}건 — W1 게이트 통과", text)
        text = re.sub(r"\d{1,3}(?:,\d{3})*쌍", f"{pairs:,}쌍", text)
    text = re.sub(r"실패 케이스 \d+건", f"실패 케이스 {n_cases}건", text)
    if f1 and rc:
        text = re.sub(r"F1 \d+/21 · \*\*AURC \d+/21 유의\*\*",
                      f"F1 {significant(f1)}/21 · **AURC {significant(rc)}/21 유의**", text)
    if f1:
        text = replace_marked(
            text, "README_F1", point_table(f1["point"], "매크로 F1 (커버리지 100%)"))
    trap = load_report("trap.json")
    if trap:
        text = replace_marked(text, "README_BLIND", anchoring_block(trap))
        text = replace_marked(text, "README_TRAP", trap_block(trap))
    if text != original:
        changed.append("README.md")
        if not check_only:
            path.write_text(text, encoding="utf-8")

    # docs/12
    path = ROOT / "docs/12-failure-registry.md"
    text = original = path.read_text(encoding="utf-8")
    tax, metrics = registry_tables()
    text = re.sub(r"^# 실패 케이스 레지스트리 — \d+건",
                  f"# 실패 케이스 레지스트리 — {n_cases}건", text, count=1, flags=re.M)
    text = re.sub(r"\d+건 중 \d+건에 probe 가 있다",
                  f"{n_cases}건 중 {n_probes}건에 probe 가 있다", text)
    text = re.sub(r"\d+건에 수치가 있다", f"{n_metrics}건에 수치가 있다", text)
    text = replace_marked(text, "TAXONOMY", tax)
    text = replace_marked(text, "METRICS", metrics)
    if text != original:
        changed.append("docs/12-failure-registry.md")
        if not check_only:
            path.write_text(text, encoding="utf-8")

    # docs/15
    if f1 and rc:
        path = ROOT / "docs/15-full-comparison.md"
        text = original = path.read_text(encoding="utf-8")
        text = replace_marked(text, "POINT_F1", point_table(f1["point"], "매크로 F1"))
        text = replace_marked(text, "CMP_F1", comparison_table(f1))
        text = replace_marked(text, "AURC", aurc_table(rc))
        text = replace_marked(text, "CMP_AURC", comparison_table(rc))
        if text != original:
            changed.append("docs/15-full-comparison.md")
            if not check_only:
                path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="쓰지 않고 어긋남만 본다")
    args = ap.parse_args()

    changed = sync(check_only=args.check)
    if not changed:
        print("문서가 산출물과 일치합니다.")
        return
    verb = "갱신 필요" if args.check else "갱신함"
    print(f"{verb}: {len(changed)}개 파일")
    for name in changed:
        print(f"  {name}")
    if args.check:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
