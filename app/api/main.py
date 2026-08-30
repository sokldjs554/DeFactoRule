"""서비스 진입점.

    uvicorn app.api.main:app --reload
    python3 scripts/serve.py

엔드포인트는 이 프로젝트가 실제로 다루는 것을 그대로 노출한다 — 결론 예측과
**기권**, 기저율, 위험-커버리지 곡선, 실패 레지스트리. 대화창이 아니다.

모든 평가 응답은 커밋된 산출물에서 계산한다. 서버가 그 자리에서 모델을 돌려
숫자를 만들지 않는다. 화면에 뜬 값과 `scripts/` 로 재현한 값이 달라지면
그 자체가 버그다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from app.api.schemas import (
    BaseRatesResponse,
    ClassifyRequest,
    ClassifyResponse,
    FailureReport,
    ModelCurve,
    OperatingPoint,
    RiskCoverageResponse,
    SectorRate,
)
from app.api.service import EngineUnavailable
from app.api.service import classify as classify_service
from app.core.io import key_of, load_jsonl
from app.core.paths import DEV_BASE_RATES, EVAL, PROCESSED
from app.domain.labels import LABEL_SETS
from app.evaluation.failure_taxonomy import load_registry
from app.evaluation.metrics import macro_f1
from app.evaluation.probes import PROBES
from app.evaluation.selective import aurc, operating_points, rank_of

app = FastAPI(
    title="DeFactoRule",
    description=(
        "금융 규제당국의 회신 이력에서, 문서 어디에도 적혀 있지 않은 판단 기준을 "
        "복원한다. 이 API 는 결론 예측과 **기권**, 그리고 그 판단을 신뢰할 수 "
        "있는지 재는 지표를 함께 노출한다."
    ),
    version="0.1.0",
)

GOLD = EVAL / "nonaction_test.jsonl"
STATIC = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """문제를 시각화하는 화면.

    대화창이 아니다(명세 §15). 보여주는 것은 위험-커버리지 곡선, 같은 문턱을
    직접 움직여 보는 판정, 업권별 기저율, 그리고 실패 레지스트리의 **지금**
    상태다. 빌드 단계도 외부 자산도 없다 — 한 파일이 API 를 그대로 읽는다.
    """
    return FileResponse(STATIC / "index.html")


@app.get("/health", summary="살아 있는가, 그리고 무엇을 할 수 있는가")
def health() -> dict:
    return {
        "status": "ok",
        "gold_set": GOLD.exists(),
        "base_rates": DEV_BASE_RATES.exists(),
        "predictions": sorted(p.name for p in PROCESSED.glob("pred_nonaction_*.jsonl")),
    }


@app.post("/classify", response_model=ClassifyResponse, summary="결론 예측 (기권 포함)")
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """요청대상행위 본문을 읽고 당국이 어떤 결론을 낼지 예측한다.

    `min_confidence` 를 올리면 모델이 자신 없는 사안에서 **기권**한다.
    기권 판정은 모델이 아니라 결정론적 코드가 한다(명세 §9).
    """
    try:
        return classify_service(req)
    except EngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/base-rates", response_model=BaseRatesResponse, summary="dev 기저율")
def base_rates() -> BaseRatesResponse:
    """라벨 분포. **dev 에서만 뽑는다** — test 에서 뽑으면 정답 누출이다."""
    if not DEV_BASE_RATES.exists():
        raise HTTPException(
            status_code=404,
            detail="기저율 파일이 없습니다. scripts/base_rates.py 를 먼저 실행하세요.",
        )
    table = json.loads(DEV_BASE_RATES.read_text(encoding="utf-8"))
    return BaseRatesResponse(
        source=table["source"],
        n=table["n"],
        min_sector_n=table["min_sector_n"],
        overall=table["overall"],
        sectors=[
            SectorRate(sector=name, n=info["n"], reliable=info["reliable"], rates=info["rates"])
            for name, info in sorted(
                table["sectors"].items(), key=lambda kv: -kv[1]["n"]
            )
        ],
    )


@app.get(
    "/evaluation/risk-coverage",
    response_model=RiskCoverageResponse,
    summary="위험-커버리지 곡선과 AURC",
)
def risk_coverage(
    model: Optional[list[str]] = Query(
        None, description="비교할 모델 이름. 생략하면 결측이 없는 예측 전부"
    ),
) -> RiskCoverageResponse:
    """기권을 허용했을 때의 공정한 비교.

    커버리지가 다르면 정확도를 비교할 수 없다. 곡선 전체를 겹쳐 보고 AURC 로
    요약한다 — 위험의 평균이므로 **낮을수록 좋다.**

    결측이 있는 예측 파일은 기본적으로 제외한다. 빠진 사례가 무작위가 아니면
    남은 부분집합의 비교는 결과가 아니기 때문이다(실패 케이스 EV-08).
    """
    if not GOLD.exists():
        raise HTTPException(status_code=404, detail="평가셋이 없습니다.")
    gold = {key_of(r): r for r in load_jsonl(GOLD) if r.get("label")}

    available = {}
    skipped = []
    for path in sorted(PROCESSED.glob("pred_nonaction_*.jsonl")):
        name = path.stem.replace("pred_nonaction_", "")
        rows = load_jsonl(path)
        missing = sum(1 for r in rows if r.get("predicted") is None or "error" in r)
        if missing or len(rows) < len(gold):
            skipped.append(f"{name}(결측 {missing + max(0, len(gold) - len(rows))})")
            continue
        available[name] = {key_of(r): r for r in rows}

    wanted = list(model) if model else sorted(available)
    unknown = [m for m in wanted if m not in available]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"쓸 수 없는 모델: {unknown}. 사용 가능: {sorted(available)}. 제외됨: {skipped}",
        )
    chosen = {m: available[m] for m in wanted}
    if not chosen:
        raise HTTPException(
            status_code=404, detail=f"완전한 예측 파일이 없습니다. 제외됨: {skipped}"
        )

    labels = LABEL_SETS["nonaction"]
    keys = [k for k in gold if all(k in preds for preds in chosen.values())]
    curves = []
    for name, preds in chosen.items():
        items = [
            (gold[k]["label"], preds[k].get("predicted", ""), rank_of(preds[k]))
            for k in keys
        ]
        points = operating_points(items, labels)
        curves.append(
            ModelCurve(
                name=name,
                aurc=aurc(points),
                flat=len(points) == 1,
                points=[OperatingPoint(**{
                    f: p[f] for f in ("coverage", "n", "risk", "accuracy", "macro_f1")
                }) for p in points],
            )
        )

    note = (
        "AURC는 위험의 평균이라 낮을수록 좋습니다. "
        "점이 하나뿐이면 기권할 줄 모르는 모델이에요."
    )
    if skipped:
        note += f" 결측이 있어 제외된 예측: {', '.join(skipped)}."
    return RiskCoverageResponse(
        n=len(keys), label_set="nonaction", curves=curves, note=note
    )


@app.get("/evaluation/models", summary="모델별 매크로 F1 요약")
def models() -> dict:
    """커버리지 100%에서의 성적. 기권을 허용한 비교는 /evaluation/risk-coverage 다."""
    if not GOLD.exists():
        raise HTTPException(status_code=404, detail="평가셋이 없습니다.")
    gold = {key_of(r): r for r in load_jsonl(GOLD) if r.get("label")}
    labels = LABEL_SETS["nonaction"]

    out = []
    for path in sorted(PROCESSED.glob("pred_nonaction_*.jsonl")):
        preds = {key_of(r): r for r in load_jsonl(path)}
        scored = [
            (gold[k]["label"], preds[k]["predicted"])
            for k in gold
            if k in preds and preds[k].get("predicted") is not None
        ]
        if not scored:
            continue
        macro, per = macro_f1(scored, labels)
        out.append({
            "name": path.stem.replace("pred_nonaction_", ""),
            "n_scored": len(scored),
            "coverage": len(scored) / len(gold),
            "accuracy": sum(1 for g, p in scored if g == p) / len(scored),
            "macro_f1": macro,
            "per_label": per,
            "complete": len(scored) == len(gold),
        })
    return {
        "gold_size": len(gold),
        "models": sorted(out, key=lambda m: -m["macro_f1"]),
        "note": (
            "complete 가 false 인 모델은 결측이 있다. 빠진 사례가 무작위가 아니면 "
            "그 숫자는 다른 모델과 비교할 수 없다."
        ),
    }


@app.get("/failures", response_model=FailureReport, summary="실패 케이스 레지스트리")
def failures(
    run_probes: bool = Query(True, description="재현 검사를 실제로 돌린다"),
) -> FailureReport:
    """46건의 실패 케이스와, 그 수정이 지금도 유지되는지."""
    cases = load_registry()
    by_layer: dict[str, int] = {}
    payload = []
    for c in cases:
        by_layer[c["layer"]] = by_layer.get(c["layer"], 0) + 1
        passed = detail = None
        if run_probes and c.get("probe") in PROBES:
            try:
                passed, detail = PROBES[c["probe"]]()
            except Exception as exc:  # noqa: BLE001
                passed, detail = False, f"예외 {type(exc).__name__}: {exc}"
        payload.append({
            **{k: c[k] for k in ("id", "layer", "category", "title", "symptom", "fix", "status")},
            "metric": c.get("metric"),
            "probe": c.get("probe"),
            "probe_passed": passed,
            "probe_detail": detail,
        })
    return FailureReport(
        total=len(cases),
        by_layer=by_layer,
        open_cases=[c["id"] for c in cases if c["status"] == "open"],
        cases=payload,
    )
