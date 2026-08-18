"""결측 검사 자체를 검사한다.

이 검사의 첫 구현은 **오류 행만** 셌다. 그래서 156/170 짜리 파일이 "결측 0"
으로 보고되고 통과했다 — EV-01(30건 예측을 170건 gold 로 채점)과 똑같은
맹점이, 하필 그것을 막으려고 만든 검사 안에 들어 있었다.

빠진 행은 무작위가 아닐 때가 많다. sector 의 결측 39건은 2025년에 몰려
있었고, llm·prior 의 14건은 파서 수정으로 본문이 바뀐 사례였다. 그래서
'몇 건 있나' 가 아니라 'gold 를 전부 덮나' 를 물어야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.probes import check_completeness

GOLD = [
    {"source": "t", "page": i, "serial": str(i), "pair_index": 1,
     "request": "질의", "label": "비조치"}
    for i in range(1, 11)
]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def gold_path(tmp_path: Path) -> Path:
    path = tmp_path / "gold.jsonl"
    write(path, GOLD)
    return path


def test_full_coverage_passes(gold_path: Path, tmp_path: Path):
    write(tmp_path / "pred_nonaction_x.jsonl",
          [{**g, "predicted": "비조치"} for g in GOLD])
    ok, detail = check_completeness(gold_path, tmp_path)
    assert ok, detail
    assert "누락 0" in detail


def test_missing_rows_are_caught(gold_path: Path, tmp_path: Path):
    """행이 아예 없는 경우 — 첫 구현이 놓쳤던 바로 그 형태."""
    write(tmp_path / "pred_nonaction_x.jsonl",
          [{**g, "predicted": "비조치"} for g in GOLD[:6]])
    ok, detail = check_completeness(gold_path, tmp_path)
    assert not ok, "6/10 인데 통과했습니다"
    assert "누락 4" in detail


def test_error_rows_are_caught(gold_path: Path, tmp_path: Path):
    rows = [{**g, "predicted": "비조치"} for g in GOLD]
    rows[0] = {**GOLD[0], "predicted": None, "error": "APIStatusError: 400"}
    write(tmp_path / "pred_nonaction_x.jsonl", rows)
    ok, detail = check_completeness(gold_path, tmp_path)
    assert not ok
    assert "실패 1" in detail


def test_both_kinds_are_reported_together(gold_path: Path, tmp_path: Path):
    rows = [{**g, "predicted": "비조치"} for g in GOLD[:8]]
    rows[0] = {**GOLD[0], "predicted": None, "error": "boom"}
    write(tmp_path / "pred_nonaction_x.jsonl", rows)
    ok, detail = check_completeness(gold_path, tmp_path)
    assert not ok
    assert "실패 1" in detail and "누락 2" in detail


def test_no_prediction_files_is_a_failure(gold_path: Path, tmp_path: Path):
    ok, detail = check_completeness(gold_path, tmp_path)
    assert not ok
    assert "하나도 없다" in detail


def test_missing_gold_is_skipped_not_passed_silently(tmp_path: Path):
    ok, detail = check_completeness(tmp_path / "없음.jsonl", tmp_path)
    assert ok and "건너뜀" in detail
