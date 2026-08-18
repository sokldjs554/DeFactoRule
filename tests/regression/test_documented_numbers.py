"""문서에 적힌 수치가 지금도 재현되는가.

## 왜 이 파일이 있는가

`docs/15` 에 손으로 옮겨 적은 수치가 있었다. 그 뒤 규칙 학습기를 고쳐
`induced` 의 예측이 바뀌었고, `docs/14` 는 갱신했지만 `docs/15` 는 놓쳤다.
결과는 단순한 오타가 아니었다.

    AURC 보정 후 유의    문서 12쌍   실제 10쌍
    llm − induced        문서 "유의"  실제 "보정 후 탈락"

**판정이 뒤집힌 문장이 저장소에 남아 있었다.** 이 프로젝트가 내세우는 것이
"숫자를 정직하게 다룬다" 인데, 그 주장 자체가 무너지는 자리다.

개수 주장도 같이 밀렸다 — README 은 레지스트리를 46건, 테스트를 154개로
적고 있었는데 실제로는 53건과 243개였다.

## 무엇을 하는가

문서의 수치를 **산출물에서 다시 계산해** 대조한다. 어긋나면 어느 파일의
무엇이 틀렸는지 말하고 실패한다. 고치는 방법은 문서를 손대는 것이 아니라
생성 스크립트를 다시 돌리는 것이다.

이 검사가 잡지 못하는 것도 분명히 해 둔다 — 문서의 **서술**이 수치와 맞는지는
사람이 읽어야 한다. 여기서 보는 것은 숫자뿐이다.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
MINUS = "−"  # 문서에서 쓰는 빼기 기호


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} 이 없습니다")
    return path.read_text(encoding="utf-8")


def report(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} 이 없습니다. E7 을 먼저 돌리세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def significant(d: dict) -> int:
    return sum(1 for c in d["comparisons"] if c["significant_holm"])


def section(doc: str, tag: str) -> str:
    """문서에서 `<!-- TAG:시작 -->` 과 `<!-- TAG:끝 -->` 사이만 꺼낸다.

    구획 없이 정규식만 쓰면 매크로 F1 표와 AURC 표가 모양이 같아 뒤엣것이
    앞엣것을 덮는다. 실제로 첫 판에서 그렇게 잘못 통과할 뻔했다.
    """
    begin, end = f"<!-- {tag}:시작 -->", f"<!-- {tag}:끝 -->"
    assert begin in doc and end in doc, f"문서에 {tag} 구획 표시가 없습니다"
    return doc[doc.index(begin) + len(begin) : doc.index(end)]


# ── 개수 ─────────────────────────────────────────────────────────
def test_readme_test_count_matches_reality():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    match = re.search(r"(\d+) tests? collected", out.stdout)
    assert match, out.stdout[-500:]
    actual = int(match.group(1))

    claimed = re.search(r"<!--TESTS-->(\d+)<!--/TESTS-->", read("README.md"))
    assert claimed, "README 에 <!--TESTS--> 표시가 없습니다"
    assert int(claimed.group(1)) == actual, (
        f"README 은 테스트 {claimed.group(1)}개라고 적었는데 실제는 {actual}개입니다."
    )


def test_registry_count_matches_everywhere():
    n = sum(1 for line in (ROOT / "data/failures/registry.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())

    readme = re.search(r"실패 케이스 (\d+)건", read("README.md"))
    assert readme and int(readme.group(1)) == n, (
        f"README 의 실패 케이스 수가 {readme.group(1) if readme else '없음'} — 실제 {n}건"
    )

    doc = read("docs/12-failure-registry.md")
    title = re.search(r"^# 실패 케이스 레지스트리 — (\d+)건", doc, re.M)
    assert title and int(title.group(1)) == n, (
        f"docs/12 제목이 {title.group(1) if title else '없음'} — 실제 {n}건"
    )
    body = re.search(r"(\d+)건 중 (\d+)건에 probe 가 있다", doc)
    assert body and int(body.group(1)) == n, "docs/12 본문의 총 건수가 어긋납니다"

    probes = sum(
        1 for line in (ROOT / "data/failures/registry.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("probe")
    )
    assert int(body.group(2)) == probes, (
        f"docs/12 는 probe {body.group(2)}건이라 적었는데 실제 {probes}건"
    )


def test_corpus_counts_in_readme():
    from app.core.io import load_jsonl
    from app.core.paths import PROCESSED

    cases = len(load_jsonl(PROCESSED / "cases_nonaction.jsonl")) + len(
        load_jsonl(PROCESSED / "cases_interpretation.jsonl"))
    pairs = len(load_jsonl(PROCESSED / "qa_pairs.jsonl"))
    readme = read("README.md")
    assert f"{cases:,}건" in readme, f"README 에 코퍼스 {cases:,}건 표기가 없습니다"
    assert f"{pairs:,}쌍" in readme, f"README 에 {pairs:,}쌍 표기가 없습니다"


# ── E7 수치 ──────────────────────────────────────────────────────
def test_readme_significance_counts():
    f1, rc = report("e7_all_models.json"), report("e7_risk_coverage.json")
    claimed = re.search(r"F1 (\d+)/21 · \*\*AURC (\d+)/21 유의\*\*", read("README.md"))
    assert claimed, "README 의 E7 행을 찾지 못했습니다"
    assert int(claimed.group(1)) == significant(f1), "README 의 F1 유의 쌍 수가 어긋납니다"
    assert int(claimed.group(2)) == significant(rc), (
        f"README 은 AURC {claimed.group(2)}쌍이라 적었는데 실제 {significant(rc)}쌍입니다."
    )


@pytest.mark.parametrize("key,name,tag", [
    ("point", "e7_all_models.json", "POINT_F1"),
    ("aurc", "e7_risk_coverage.json", "AURC"),
])
def test_doc15_point_values(key, name, tag):
    """docs/15 의 모델별 점추정이 보고서와 같은가."""
    data = report(name)[key]
    doc = section(read("docs/15-full-comparison.md"), tag)
    found = dict(re.findall(r"^\| `(\w+)` \| (\d\.\d{3}) \|", doc, re.M))
    missing = set(data) - set(found)
    assert not missing, f"docs/15 에 없는 모델: {sorted(missing)}"
    for model, value in data.items():
        assert found[model] == f"{value:.3f}", (
            f"docs/15 의 {model} = {found[model]}, 실제 {value:.3f}. "
            "문서를 다시 생성하세요."
        )


@pytest.mark.parametrize("name,tag", [("e7_all_models.json", "CMP_F1"),
                                      ("e7_risk_coverage.json", "CMP_AURC")])
def test_doc15_comparison_rows(name, tag):
    """비교 표의 차이·p·판정이 보고서와 같은가. 판정이 뒤집힌 채 남는 것을 막는다."""
    data = report(name)
    doc = section(read("docs/15-full-comparison.md"), tag)
    rows = {
        (a, b): (diff, p, holm, verdict.strip())
        for a, b, diff, _ci, p, holm, verdict in re.findall(
            rf"^\| (\w+) {MINUS} (\w+) \| ([+\-]\d\.\d{{3}}) \| "
            rf"([^|]+) \| (\d\.\d{{3}}) \| (\d\.\d{{3}}) \| ([^|]+) \|",
            doc, re.M,
        )
    }
    for c in data["comparisons"]:
        key = (c["a"], c["b"])
        assert key in rows, f"docs/15 에 {c['a']} {MINUS} {c['b']} 행이 없습니다"
        diff, p, holm, verdict = rows[key]
        assert diff == f"{c['diff']:+.3f}", f"{key} 차이: 문서 {diff}, 실제 {c['diff']:+.3f}"
        assert holm == f"{c['p_holm']:.3f}", f"{key} p(Holm): 문서 {holm}, 실제 {c['p_holm']:.3f}"
        expected = ("**유의**" if c["significant_holm"]
                    else "보정 후 탈락" if c["significant_raw"] else "판정 보류")
        assert verdict == expected, (
            f"{key} 판정: 문서 '{verdict}', 실제 '{expected}'. 판정이 뒤집혀 있습니다."
        )


# ── 다른 실험 문서 ───────────────────────────────────────────────
def test_headline_model_scores_still_reproduce():
    """E1·E4·E5 가 내세운 매크로 F1 이 지금도 나오는가."""
    from app.core.io import key_of, load_jsonl
    from app.core.paths import EVAL, PROCESSED
    from app.domain.labels import NON_ACTIONS
    from app.evaluation.metrics import macro_f1

    gold_path = EVAL / "nonaction_test.jsonl"
    if not gold_path.exists():
        pytest.skip("평가셋이 없습니다")
    gold = {key_of(r): r for r in load_jsonl(gold_path) if r.get("label")}

    claims = {"llm": (0.587, "docs/07"), "prior": (0.504, "docs/10"),
              "neighbor": (0.538, "docs/13"), "keyword": (0.494, "docs/06"),
              "majority": (0.284, "docs/06")}
    for model, (want, where) in claims.items():
        path = PROCESSED / f"pred_nonaction_{model}.jsonl"
        if not path.exists():
            pytest.skip(f"{path.name} 이 없습니다")
        pred = {key_of(r): r for r in load_jsonl(path)}
        pairs = [(gold[k]["label"], pred[k]["predicted"]) for k in gold if k in pred]
        got, _ = macro_f1(pairs, NON_ACTIONS)
        assert abs(got - want) < 0.0005, (
            f"{where} 는 {model} 매크로 F1 {want} 라고 적었는데 실제 {got:.3f} 입니다."
        )


# ── 동기화 도구 ──────────────────────────────────────────────────
def test_docs_are_in_sync_with_artifacts():
    """`scripts/sync_docs.py --check` 가 통과하는가.

    어긋남을 잡는 것만으로는 부족하다 — 고치는 일이 손 작업이면 결국 같은
    자리로 돌아온다. 그래서 고치는 쪽도 명령 하나로 만들었고, 그 명령이
    '고칠 것이 없다' 고 말하는지를 여기서 본다.
    """
    from app.evaluation.doc_sync import sync

    changed = sync(check_only=True)
    assert not changed, (
        f"문서가 산출물과 어긋납니다: {changed}\n"
        "  python3 scripts/sync_docs.py 로 갱신하세요."
    )


def test_doc_check_and_test_share_one_implementation():
    """probe 와 회귀 테스트가 같은 구현을 쓰는가.

    둘이 따로 구현되면 한쪽만 고쳐지고, 그 순간 '검사가 통과한다' 는 사실이
    아무것도 보장하지 않게 된다.
    """
    from app.evaluation.doc_check import check_documented_numbers
    from app.evaluation.probes import PROBES

    assert "documented_numbers_still_reproduce" in PROBES
    ok, detail = check_documented_numbers()
    assert ok, detail
    assert PROBES["documented_numbers_still_reproduce"]()[0] is ok
