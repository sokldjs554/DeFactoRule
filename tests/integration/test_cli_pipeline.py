"""CLI 진입점을 실제로 실행한다.

모듈 단위 테스트는 임포트가 성립하는 세계에서만 돈다. 사용자는 `python
scripts/evaluate.py` 로 실행하고, 그때 sys.path[0] 은 scripts/ 다. 진입점이
저장소 루트를 얹지 않으면 `app` 이 보이지 않는다 — 단위 테스트로는 절대
잡히지 않는 고장이다.

실제로 겪은 사고도 여기 걸린다. 30건 예측을 170건 gold 로 채점하면서
커버리지 17.6%를 못 보고 지나갔다. 그래서 미매칭 경고가 나오는지도 본다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

GOLD = [
    {"source": "t", "page": 1, "serial": "1", "pair_index": 1,
     "request": "망분리 예외 적용이 가능한지 여부", "label": "조치", "sector": "전자금융업"},
    {"source": "t", "page": 2, "serial": "2", "pair_index": 1,
     "request": "위탁 범위에 관한 질의", "label": "비조치", "sector": "전자금융업"},
    {"source": "t", "page": 3, "serial": "3", "pair_index": 1,
     "request": "겸영업무 신고 대상인지 여부", "label": "비조치", "sector": "공통"},
    {"source": "t", "page": 4, "serial": "4", "pair_index": 1,
     "request": "약관 변경 보고 시점", "label": "기타", "sector": "공통"},
]


def run(*args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"{args}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc


@pytest.fixture()
def gold_path(tmp_path: Path) -> Path:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in GOLD) + "\n",
        encoding="utf-8",
    )
    return path


def test_baseline_then_evaluate(gold_path: Path, tmp_path: Path):
    pred = tmp_path / "pred.jsonl"
    run("scripts/baseline_nonaction.py", "--gold", str(gold_path),
        "--output", str(pred), "--strategy", "majority")

    rows = [json.loads(x) for x in pred.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == len(GOLD)
    assert {r["predicted"] for r in rows} == {"비조치"}

    report = tmp_path / "report.json"
    out = run("scripts/evaluate.py", "--gold", str(gold_path), "--pred", str(pred),
              "--labels", "nonaction", "--name", "majority", "--report", str(report))

    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["coverage"] == 1.0
    assert result["n_scored"] == len(GOLD)
    # 4건 중 비조치 2건만 맞는다
    assert result["accuracy_on_scored"] == pytest.approx(0.5)
    # 다수만 찍었으므로 매크로 F1 은 정확도보다 한참 낮아야 한다
    assert result["macro_f1"] < result["accuracy_on_scored"]
    assert "매크로 F1" in out.stdout


def test_missing_predictions_are_flagged(gold_path: Path, tmp_path: Path):
    """예측이 일부만 있으면 조용히 넘어가면 안 된다."""
    pred = tmp_path / "partial.jsonl"
    pred.write_text(
        json.dumps({**GOLD[0], "predicted": "비조치", "confidence": "low"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = run("scripts/evaluate.py", "--gold", str(gold_path), "--pred", str(pred),
              "--labels", "nonaction")
    assert "예측이 없는 3건" in out.stdout
    assert "--limit" in out.stdout


def test_limit_matches_the_sample(gold_path: Path, tmp_path: Path):
    pred = tmp_path / "partial.jsonl"
    pred.write_text(
        json.dumps({**GOLD[0], "predicted": "조치", "confidence": "high"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "r.json"
    run("scripts/evaluate.py", "--gold", str(gold_path), "--pred", str(pred),
        "--labels", "nonaction", "--limit", "1", "--report", str(report))
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["gold_size"] == 1
    assert result["coverage"] == 1.0


def test_risk_coverage_runs_on_two_models(gold_path: Path, tmp_path: Path):
    majority = tmp_path / "majority.jsonl"
    keyword = tmp_path / "keyword.jsonl"
    run("scripts/baseline_nonaction.py", "--gold", str(gold_path),
        "--output", str(majority), "--strategy", "majority")
    run("scripts/baseline_nonaction.py", "--gold", str(gold_path),
        "--output", str(keyword), "--strategy", "keyword")

    out = run("scripts/risk_coverage.py", "--gold", str(gold_path), "--labels", "nonaction",
              "--pred", f"majority={majority}", "--pred", f"keyword={keyword}")
    assert "AURC" in out.stdout
    # majority 는 신뢰도 신호가 없으므로 곡선이 한 점이어야 한다
    assert "기권 신호 없음" in out.stdout
