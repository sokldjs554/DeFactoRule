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
    assert "최소 신뢰도" in body["abstain_reason"]


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


def test_ui_is_served_and_self_contained():
    """화면은 한 파일로 서빙되고 외부 자산을 부르지 않는다.

    CDN 을 걸면 네트워크가 없는 환경에서 화면이 조용히 깨진다. 심사자가
    처음 여는 순간이 그런 환경일 수 있다.

    막아야 하는 것은 **자산**이다 — script·link·img 처럼 그리기 전에 받아와야
    하는 것. 본문의 <a> 는 자산이 아니다. 누르지 않으면 나가지 않고, 네트워크가
    없어도 화면은 그대로 그려진다. 그래서 태그를 보고 판단한다.
    """
    import re

    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    html = resp.text

    assets = re.findall(
        r"<(?:script|link|img|iframe|source|embed|object)\b[^>]*?"
        r'\b(?:src|href|data)="([^"]+)"',
        html,
        re.IGNORECASE,
    )
    external = [u for u in assets if u.startswith(("http://", "https://", "//"))]
    assert not external, f"화면이 외부 자산을 부릅니다: {external}"

    assert "@import" not in html, "CSS 가 외부를 불러오면 같은 문제가 생긴다"
    assert "url(http" not in html.replace(" ", ""), "스타일이 외부 자산을 참조합니다"
    assert "cdn" not in html.lower()


def test_ui_only_calls_endpoints_that_exist():
    """화면이 부르는 경로가 실제 라우트에 있는지 본다."""
    import re

    from app.api.main import app as fastapi_app

    routes = {r.path for r in fastapi_app.routes if hasattr(r, "path")}
    called = set(re.findall(r"fetch\('([^']+)'", client.get("/").text))
    missing = {c.split("?")[0] for c in called} - routes
    assert not missing, f"화면이 없는 경로를 부릅니다: {sorted(missing)}"



def test_summary_reads_the_frozen_artifact_rather_than_recomputing():
    """상단 요약은 커밋된 최종 산출물을 그대로 읽는다."""
    import json

    from app.api.main import FINAL_FREEZE

    body = client.get("/evaluation/summary").json()
    anchor = json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))["c3_anchor_recomputed"]
    for field in ("n", "answered", "abstained", "correct", "wrong"):
        assert body["profile"][field] == anchor[field]

    p = body["profile"]
    assert p["answered"] + p["abstained"] == p["n"], "답한 것과 보류한 것의 합이 전체다"
    assert p["correct"] + p["wrong"] == p["answered"], "맞고 틀린 것의 합은 답한 것이다"


def test_summary_counts_the_data_files_it_reports():
    """규모는 데이터 파일에서 직접 센다 — 문서에 적힌 값을 옮겨 적지 않는다."""
    from app.api.main import QA_PAIRS

    body = client.get("/evaluation/summary").json()
    expected = sum(1 for line in QA_PAIRS.read_bytes().splitlines() if line.strip())
    assert body["corpus"]["qa_pairs"] == expected
    assert body["corpus"]["test_set"] == body["profile"]["n"]


def test_ui_does_not_hardcode_the_headline_numbers():
    """상단 카드의 숫자는 화면에 박아 두지 않는다.

    박아 두면 재평가한 날 화면만 옛 숫자로 남는다. 실제로 문서 쪽에서 그런 일이
    있었다(FAIL-DOC-*). 화면은 API 에서 받아 그린다.
    """
    html = client.get("/").text
    for literal in ("82.89", "45.24", "1,122", "1,095", "168건 중"):
        assert literal not in html, f"화면에 {literal} 이 박혀 있습니다"

def test_health_stays_2xx_when_an_artifact_is_missing(monkeypatch):
    """산출물이 없어도 health check 는 5xx 를 돌려주지 않는다.

    Render 의 health check 는 배포 때만 도는 것이 아니다. 살아 있는 인스턴스에도
    계속 요청이 오고, 5xx 가 60초 이어지면 인스턴스를 재시작한다. 파일이 빠진 것은
    재시작으로 낫지 않으므로, 여기서 5xx 를 내면 낫지 않는 상태를 고치려고 무한히
    재시작하다 서비스가 아예 죽는다.

    그래서 이 자리는 **살아 있는가**만 답하고, **준비됐는가**는 빌드 단계에서
    막는다(scripts/check_release.py). 이 테스트가 그 경계를 지킨다.
    """
    from pathlib import Path

    from app.api import main as api

    monkeypatch.setattr(api, "GOLD", Path("/nonexistent/gold.jsonl"))
    resp = client.get("/health")
    assert resp.status_code < 400, "5xx 를 내면 Render 가 재시작을 반복한다"

    body = resp.json()
    assert body["ready"] is False, "준비되지 않았다는 사실 자체는 알려야 한다"
    assert "gold_set" in body["missing"]


def test_release_gate_is_wired_into_the_build():
    """산출물 확인은 빌드 명령에 붙어 있어야 의미가 있다.

    스크립트만 있고 render.yaml 에 연결되지 않으면 아무것도 막지 못한다 —
    만들어 놓고 아무도 부르지 않는 코드(IN-15)와 같은 실패다.
    """
    from app.core.paths import ROOT

    assert (ROOT / "scripts" / "check_release.py").exists()
    blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "scripts/check_release.py" in blueprint, (
        "render.yaml 의 buildCommand 에 없으면 빈 배포를 막지 못한다"
    )


def test_release_gate_passes_on_this_checkout():
    """지금 트리는 배포 가능해야 한다."""
    import importlib.util

    from app.core.paths import ROOT

    spec = importlib.util.spec_from_file_location(
        "check_release", ROOT / "scripts" / "check_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
