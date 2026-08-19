"""선례 신뢰도 보정 — Router 의 전제가 사는지 죽는지가 여기 걸려 있다."""

from __future__ import annotations

import pytest

from app.agents.calibration import (
    BAND_DOUBT,
    BAND_TRUST,
    DOUBT,
    TRUST,
    band_of,
    bands_are_separable,
    calibrate,
    loo_links,
    risk_of,
    risk_table,
)
from app.agents.calibration import (
    BAND_MIDDLE as BAND_MIDDLE_KEY,
)


def _rows(n: int, label_of):
    return [{"source": "t", "page": i, "serial": str(i), "pair_index": 1,
             "request": f"요청 {i} 내부망과 외부망의 망분리 구간에 관한 질의",
             "label": label_of(i)} for i in range(n)]


def test_loo_excludes_self():
    """자기 자신을 이웃으로 삼지 않는가 — 이게 깨지면 표 전체가 거짓말이 된다.

    빼지 않으면 유사도가 1.0 이 되고 오류율이 0 이 된다. 그 표를 test 에
    적용하면 "언제나 선례를 믿어라" 가 되고, Router 는 검색을 무조건 믿는
    시스템이 된다 — 이 프로젝트가 하지 말자고 만든 바로 그것이다.
    """
    from app.evaluation.confusable import idf_table

    rows = _rows(6, lambda i: "조치" if i % 2 else "비조치")
    idf = idf_table([r["request"] for r in rows])
    links = loo_links(rows, idf)
    assert len(links) == len(rows)
    assert all(x["similarity"] < 0.9999 for x in links), (
        "유사도 1.0 이 나왔습니다 — 자기 자신을 이웃으로 잡았습니다"
    )


def test_band_boundaries():
    assert band_of(TRUST) == BAND_TRUST
    assert band_of(TRUST - 1e-9) != BAND_TRUST
    assert band_of(DOUBT) != BAND_DOUBT
    assert band_of(DOUBT - 1e-9) == BAND_DOUBT
    assert band_of(0.0) == BAND_DOUBT


def test_separability_is_detected_both_ways():
    """구간이 갈리는지 아닌지를 실제로 가르는가 — 양쪽 다 확인한다."""
    separable = {
        BAND_TRUST: {"n": 30, "wrong": 1, "risk": 0.03, "ci95": [0.0, 0.15]},
        BAND_MIDDLE_KEY: {"n": 5, "wrong": 2, "risk": 0.4, "ci95": [0.1, 0.8]},
        BAND_DOUBT: {"n": 40, "wrong": 20, "risk": 0.5, "ci95": [0.35, 0.65]},
    }
    ok, detail = bands_are_separable(separable)
    assert ok, detail

    flat = {
        BAND_TRUST: {"n": 30, "wrong": 10, "risk": 0.33, "ci95": [0.18, 0.52]},
        BAND_MIDDLE_KEY: {"n": 5, "wrong": 2, "risk": 0.4, "ci95": [0.1, 0.8]},
        BAND_DOUBT: {"n": 40, "wrong": 14, "risk": 0.35, "ci95": [0.22, 0.51]},
    }
    ok, detail = bands_are_separable(flat)
    assert not ok, "겹치는 구간을 갈렸다고 했습니다"
    assert "겹친다" in detail


def test_empty_band_falls_back_to_overall():
    """비어 있는 구간에 자신 있게 0 을 돌려주지 않는가."""
    table = {
        "overall_risk": 0.33,
        "by_band": {
            BAND_TRUST: {"n": 0, "wrong": 0, "risk": None, "ci95": [0.0, 1.0]},
            BAND_MIDDLE_KEY: {"n": 0, "wrong": 0, "risk": None, "ci95": [0.0, 1.0]},
            BAND_DOUBT: {"n": 10, "wrong": 5, "risk": 0.5, "ci95": [0.24, 0.76]},
        },
    }
    assert risk_of(table, 0.9) == 0.33, "빈 구간을 위험 0 으로 봤습니다"
    assert risk_of(table, 0.01) == 0.5


def test_risk_table_counts_add_up():
    from app.evaluation.confusable import idf_table

    rows = _rows(12, lambda i: ["비조치", "조치", "기타"][i % 3])
    idf = idf_table([r["request"] for r in rows])
    table = risk_table(loo_links(rows, idf))
    assert sum(c["n"] for c in table.values()) == len(rows)


# ── 실제 dev 위에서 — 설계의 전제 ─────────────────────────────────
def test_dev_bands_are_actually_separable():
    """실제 dev 에서 믿음/못믿음 구간이 갈리는가.

    갈리지 않으면 Router 의 R5·R8 이 근거를 잃는다. 설계서 §문제점 1이
    가리킨 바로 그 지점이고, 이 검사가 그 전제를 붙들어 둔다.
    """
    from app.core.io import load_jsonl
    from app.core.paths import EVAL, PROCESSED
    from app.evaluation.confusable import idf_table

    dev_path = EVAL / "nonaction_dev.jsonl"
    cases_path = PROCESSED / "cases_nonaction.jsonl"
    if not dev_path.exists() or not cases_path.exists():
        pytest.skip("평가셋이 없습니다")

    dev = [r for r in load_jsonl(dev_path) if r.get("label")]
    cases = load_jsonl(cases_path)
    texts = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
             for c in cases]
    report = calibrate(dev, idf_table([t for t in texts if t]))

    assert report["bands_separable"], report["separability_detail"]
    assert report["by_band"][BAND_TRUST]["risk"] < 0.2, (
        "믿음 구간의 위험이 0.2 를 넘습니다 — 그러면 믿을 구간이 아닙니다"
    )
    assert report["by_band"][BAND_DOUBT]["risk"] > 0.35, (
        "못믿음 구간의 위험이 낮습니다 — 구간을 다시 잡아야 합니다"
    )


def test_two_dimensional_table_is_too_sparse_to_use():
    """2차원 표를 안 쓰기로 한 근거가 지금도 유효한가.

    설계 초안은 (유사도 × 선례 라벨) 표를 쓰려 했다. 실제로 만들어 보니
    칸 절반이 1~2건이었다. 그 사실을 검사로 붙들어 둔다 — 나중에 누가
    "2차원으로 하면 더 정교할 텐데" 라고 할 때 답이 여기 있다.
    """
    from app.core.io import load_jsonl
    from app.core.paths import EVAL, PROCESSED
    from app.evaluation.confusable import idf_table

    dev_path = EVAL / "nonaction_dev.jsonl"
    if not dev_path.exists():
        pytest.skip("평가셋이 없습니다")
    dev = [r for r in load_jsonl(dev_path) if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    texts = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
             for c in cases]
    report = calibrate(dev, idf_table([t for t in texts if t]))
    assert report["sparse_cells"] >= len(report["joint_cell_counts"]) / 3, (
        "2차원 칸이 충분히 채워졌습니다 — 설계 판단을 다시 볼 때입니다"
    )
