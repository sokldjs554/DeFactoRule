"""API 계약 — 실제로 앱을 띄워서 검증한다.

가장 중요한 것은 **기권이 계약에 있는가**다. 이 프로젝트의 결론은 "LLM 이
규칙보다 정확한 것이 아니라 자기가 틀릴 때를 안다" 는 것이고(E2), 서비스가
'모르겠다' 를 못 돌려주면 그 이점은 경계에서 사라진다.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi 가 설치되지 않았습니다")
pytest.importorskip("httpx", reason="TestClient 에 httpx 가 필요합니다")

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import app  # noqa: E402
from app.evaluation.failure_taxonomy import MIN_CASES  # noqa: E402

client = TestClient(app)


def test_health_reports_what_is_available():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["predictions"], list)


def test_rule_engine_classifies_and_names_its_rule():
    body = client.post(
        "/classify", json={"request_text": "내부망과 외부망의 망연계 구간에 대하여"}
    ).json()
    assert body["decision"] == "조치"
    assert body["abstained"] is False
    assert body["confidence"] == "high"
    assert body["rule"].startswith("rule:")


def test_threshold_turns_a_weak_answer_into_an_abstention():
    body = client.post(
        "/classify",
        json={"request_text": "전혀 무관한 문장입니다", "min_confidence": "high"},
    ).json()
    assert body["abstained"] is True
    assert body["decision"] is None, "기권했는데 라벨이 남아 있으면 호출자가 그것을 답으로 쓴다"
    assert "문턱" in body["abstain_reason"]


def test_low_threshold_never_abstains():
    body = client.post(
        "/classify",
        json={"request_text": "전혀 무관한 문장입니다", "min_confidence": "low"},
    ).json()
    assert body["abstained"] is False
    assert body["decision"] is not None


def test_empty_request_is_rejected_by_the_schema():
    assert client.post("/classify", json={"request_text": ""}).status_code == 422


def test_unknown_label_cannot_be_requested():
    resp = client.post(
        "/classify", json={"request_text": "질의", "min_confidence": "아주높음"}
    )
    assert resp.status_code == 422


def test_base_rates_come_from_dev():
    resp = client.get("/base-rates")
    if resp.status_code == 404:
        pytest.skip("기저율 파일이 없습니다")
    body = resp.json()
    assert body["source"] == "dev", "test 에서 뽑은 기저율을 노출하면 정답 누출이다"
    assert body["n"] > 0
    small = [s for s in body["sectors"] if s["n"] < body["min_sector_n"]]
    assert all(not s["reliable"] for s in small), "표본이 적은 업권을 신뢰한다고 표시하면 안 된다"


def test_model_summary_marks_incomplete_predictions():
    resp = client.get("/evaluation/models")
    if resp.status_code == 404:
        pytest.skip("평가셋이 없습니다")
    body = resp.json()
    assert body["models"], "모델이 하나도 없습니다"
    for m in body["models"]:
        assert (m["coverage"] == 1.0) == m["complete"]
    majority = next((m for m in body["models"] if m["name"] == "majority"), None)
    if majority:
        # 다수 클래스만 찍는 분류기의 매크로 F1 은 정확도보다 한참 낮아야 한다
        assert majority["macro_f1"] < majority["accuracy"] - 0.30


def test_risk_coverage_excludes_incomplete_predictions():
    resp = client.get("/evaluation/risk-coverage")
    if resp.status_code == 404:
        pytest.skip("완전한 예측 파일이 없습니다")
    body = resp.json()
    summary = client.get("/evaluation/models").json()
    complete = {m["name"] for m in summary["models"] if m["complete"]}
    returned = {c["name"] for c in body["curves"]}
    assert returned <= complete, (
        f"결측이 있는 예측이 곡선에 들어갔습니다: {sorted(returned - complete)}"
    )
    majority = next((c for c in body["curves"] if c["name"] == "majority"), None)
    if majority:
        assert majority["flat"] is True, "신뢰도 신호가 없는 모델의 곡선은 한 점이어야 한다"


def test_risk_coverage_rejects_unknown_model():
    resp = client.get("/evaluation/risk-coverage", params={"model": "존재하지않음"})
    assert resp.status_code in (400, 404)


def test_failure_registry_is_served_with_live_probe_results():
    body = client.get("/failures").json()
    assert body["total"] >= MIN_CASES
    ran = [c for c in body["cases"] if c["probe_passed"] is not None]
    assert ran, "probe 결과가 하나도 없습니다"
    for case in ran:
        expected = case["status"] == "fixed"
        assert case["probe_passed"] is expected, (
            f"{case['id']} status={case['status']} 인데 probe={case['probe_passed']} "
            f"— {case['probe_detail']}"
        )


def test_failures_can_skip_probes():
    body = client.get("/failures", params={"run_probes": False}).json()
    assert all(c["probe_passed"] is None for c in body["cases"])
