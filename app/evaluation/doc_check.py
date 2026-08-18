"""문서에 적힌 수치를 산출물과 대조한다.

테스트와 probe 가 **같은 구현**을 쓰도록 여기 둔다. 둘이 따로 구현되면 한쪽만
고쳐지고, 그 순간 "검사가 통과한다" 는 사실이 아무것도 보장하지 않게 된다.
결측 검사(`check_completeness`) 에서 같은 이유로 같은 처리를 했다.

## 무엇이 있었나

`docs/15` 에 손으로 옮겨 적은 수치가 있었다. 그 뒤 규칙 학습기를 고쳐
`induced` 의 예측이 바뀌었고, `docs/14` 는 갱신했지만 `docs/15` 는 놓쳤다.

    AURC 보정 후 유의    문서 12쌍    실제 10쌍
    llm − induced        문서 "유의"   실제 "보정 후 탈락"

**판정이 뒤집힌 문장이 저장소에 남아 있었다.** "숫자를 정직하게 다룬다" 가 이
프로젝트의 주장인데 그 주장 자체가 무너지는 자리다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.paths import ROOT

MINUS = "−"  # 문서에서 쓰는 빼기 기호
DOC15 = "docs/15-full-comparison.md"
REGISTRY = "data/failures/registry.jsonl"


def section(doc: str, tag: str) -> str:
    """`<!-- TAG:시작 -->` 과 `<!-- TAG:끝 -->` 사이만 꺼낸다.

    구획 없이 정규식만 쓰면 매크로 F1 표와 AURC 표가 모양이 같아 뒤엣것이
    앞엣것을 덮는다. 첫 판이 실제로 그렇게 잘못 통과할 뻔했다.
    """
    begin, end = f"<!-- {tag}:시작 -->", f"<!-- {tag}:끝 -->"
    if begin not in doc or end not in doc:
        raise LookupError(f"{tag} 구획 표시가 없습니다")
    return doc[doc.index(begin) + len(begin) : doc.index(end)]


def verdict_of(comparison: dict) -> str:
    if comparison["significant_holm"]:
        return "**유의**"
    return "보정 후 탈락" if comparison["significant_raw"] else "판정 보류"


def compare_table(doc: str, tag: str, data: dict) -> list[str]:
    """비교 표 한 개를 보고서와 대조하고, 어긋난 것을 전부 돌려준다."""
    body = section(doc, tag)
    rows = {
        (a, b): (diff, holm, verdict.strip())
        for a, b, diff, _ci, _p, holm, verdict in re.findall(
            rf"^\| (\w+) {MINUS} (\w+) \| ([+\-]\d\.\d{{3}}) \| "
            rf"([^|]+) \| (\d\.\d{{3}}) \| (\d\.\d{{3}}) \| ([^|]+) \|",
            body, re.M,
        )
    }
    problems = []
    for c in data["comparisons"]:
        key = (c["a"], c["b"])
        if key not in rows:
            problems.append(f"{tag}: {c['a']} {MINUS} {c['b']} 행 없음")
            continue
        diff, holm, verdict = rows[key]
        if diff != f"{c['diff']:+.3f}":
            problems.append(f"{tag} {key} 차이: 문서 {diff} · 실제 {c['diff']:+.3f}")
        if holm != f"{c['p_holm']:.3f}":
            problems.append(f"{tag} {key} p(Holm): 문서 {holm} · 실제 {c['p_holm']:.3f}")
        want = verdict_of(c)
        if verdict != want:
            problems.append(f"{tag} {key} 판정: 문서 '{verdict}' · 실제 '{want}' ★뒤집힘")
    return problems


def check_documented_numbers(root: Path | None = None) -> tuple[bool, str]:
    """문서 수치가 산출물과 맞는지. (통과 여부, 설명) 을 돌려준다."""
    root = Path(root or ROOT)
    results = root / "experiments" / "results"
    doc_path = root / DOC15
    if not doc_path.exists():
        return True, "docs/15 가 없습니다 — 건너뜀"
    reports = {
        "CMP_F1": results / "e7_all_models.json",
        "CMP_AURC": results / "e7_risk_coverage.json",
    }
    if not all(p.exists() for p in reports.values()):
        return True, "E7 보고서가 없습니다 — 건너뜀"

    doc = doc_path.read_text(encoding="utf-8")
    problems: list[str] = []
    checked = 0
    for tag, path in reports.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        checked += len(data["comparisons"])
        problems += compare_table(doc, tag, data)

    # 레지스트리 건수가 문서와 맞는가
    reg = root / REGISTRY
    if reg.exists():
        n = sum(1 for line in reg.read_text(encoding="utf-8").splitlines() if line.strip())
        readme = (root / "README.md").read_text(encoding="utf-8")
        found = re.search(r"실패 케이스 (\d+)건", readme)
        if not found or int(found.group(1)) != n:
            problems.append(
                f"README 실패 케이스 수: 문서 {found.group(1) if found else '없음'} · 실제 {n}"
            )
        checked += 1

    if problems:
        return False, f"{len(problems)}건 어긋남 — " + " / ".join(problems[:3])
    return True, f"비교 {checked}건 대조 · 어긋남 없음"
