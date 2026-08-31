"""DeFactoRule API와 대시보드 진입점입니다."""

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
    CorpusSize,
    FailureReport,
    FrozenProfile,
    ModelCurve,
    OperatingPoint,
    RiskCoverageResponse,
    SectorRate,
    SummaryResponse,
)
from app.api.service import EngineUnavailable
from app.api.service import classify as classify_service
from app.core.io import key_of, load_jsonl
from app.core.paths import DEV_BASE_RATES, EVAL, PROCESSED, RESULTS
from app.domain.labels import LABEL_SETS
from app.evaluation.failure_taxonomy import load_registry
from app.evaluation.metrics import macro_f1
from app.evaluation.probes import PROBES
from app.evaluation.selective import aurc, operating_points, rank_of

app = FastAPI(
    title="DeFactoRule",
    description=(
        "금융 규제 회신 사례를 바탕으로 요청대상행위의 결론을 예측하는 API입니다. "
        "결과와 함께 신뢰도와 판단 보류 여부를 반환합니다."
    ),
    version="0.1.0",
)

GOLD = EVAL / "nonaction_test.jsonl"
STATIC = Path(__file__).resolve().parent / "static"

# 화면 상단 요약이 읽는 것. 전부 커밋된 산출물이다 — 여기서 모델을 돌리지 않는다.
FINAL_FREEZE = RESULTS / "clean" / "final_clean_temporal.json"
CORPUS_FILES = ("cases_interpretation.jsonl", "cases_nonaction.jsonl")
QA_PAIRS = PROCESSED / "qa_pairs.jsonl"
TEST_CLEAN = EVAL / "nonaction_test_clean.jsonl"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """문제를 시각화하는 화면.

    대화창이 아니다(명세 §15). 보여주는 것은 위험-커버리지 곡선, 같은 문턱을
    직접 움직여 보는 판정, 업권별 기저율, 그리고 실패 레지스트리의 **지금**
    상태다. 빌드 단계도 외부 자산도 없다 — 한 파일이 API 를 그대로 읽는다.
    """
    return FileResponse(STATIC / "index.html")


@app.get("/health", summary="서비스 상태 확인")
def health() -> dict:
    """서비스가 살아 있는지, 그리고 산출물이 제자리에 있는지 알립니다.

    산출물이 없어도 **2xx 를 유지한다.** Render 의 health check 는 배포 때만
    도는 것이 아니라 살아 있는 인스턴스에도 계속 요청을 보내고, 5xx 가 60초
    이어지면 인스턴스를 재시작한다. 파일이 빠진 것은 재시작으로 낫지 않으므로
    여기서 5xx 를 돌려주면 재시작만 반복하다 서비스가 아예 죽는다.

    준비 여부는 더 앞에서 막는다 — `scripts/check_release.py` 가 빌드 단계에서
    확인하고, 없으면 빌드를 실패시켜 배포 자체가 일어나지 않게 한다.
    """
    ready = {
        "gold_set": GOLD.exists(),
        "base_rates": DEV_BASE_RATES.exists(),
        "predictions": sorted(p.name for p in PROCESSED.glob("pred_nonaction_*.jsonl")),
    }
    missing = [k for k, v in ready.items() if not v]
    return {
        "status": "ok",
        **ready,
        "ready": not missing,
        "missing": missing,
    }


@app.post("/classify", response_model=ClassifyResponse, summary="결론 예측 (판단 보류 포함)")
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """요청대상행위 본문을 읽고 결론을 예측합니다.

    신뢰도가 `min_confidence` 보다 낮으면 판단을 보류합니다.
    """
    try:
        return classify_service(req)
    except EngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/base-rates", response_model=BaseRatesResponse, summary="dev 세트 업권별 분포")
def base_rates() -> BaseRatesResponse:
    """업권별 라벨 분포를 반환합니다.

    데이터 누수를 막기 위해 dev 세트에서만 계산합니다.
    """
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
    """신뢰도 기준을 바꿨을 때 모델별 오류율이 어떻게 달라지는지 비교합니다.

    AURC는 낮을수록 좋습니다. 일부 결과가 누락된 예측 파일은 비교에서 제외합니다.
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
        "AURC는 신뢰도 기준별 오류율을 하나로 요약한 값이며 낮을수록 좋습니다. "
        "점이 하나인 모델은 판단 보류 기능을 사용하지 않습니다."
    )
    if skipped:
        note += f" 결측이 있어 제외된 예측: {', '.join(skipped)}."
    return RiskCoverageResponse(
        n=len(keys), label_set="nonaction", curves=curves, note=note
    )


def _count_lines(path: Path) -> int:
    """빈 줄을 뺀 줄 수. 파일 전체를 메모리에 올리지 않는다."""
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


@app.get(
    "/evaluation/summary",
    response_model=SummaryResponse,
    summary="최종 평가 요약",
)
def summary() -> SummaryResponse:
    """화면 상단에 표시하는 최종 평가 요약입니다.

    커밋된 산출물과 데이터 파일만 읽습니다. 모델을 새로 실행하지 않습니다.
    """
    if not FINAL_FREEZE.exists():
        raise HTTPException(status_code=404, detail="최종 평가 산출물이 없습니다.")
    anchor = json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))["c3_anchor_recomputed"]
    fields = (
        "n", "answered", "abstained", "correct", "wrong",
        "coverage", "accuracy_on_answered",
    )
    return SummaryResponse(
        profile=FrozenProfile(**{f: anchor[f] for f in fields}),
        corpus=CorpusSize(
            cases=sum(_count_lines(PROCESSED / name) for name in CORPUS_FILES),
            qa_pairs=_count_lines(QA_PAIRS),
            test_set=_count_lines(TEST_CLEAN),
        ),
        caveat=(
            "답변 정확도는 답하지 않은 사례를 빼고 계산한 값입니다. "
            "반드시 답변 비율과 함께 읽어야 합니다."
        ),
        source="experiments/results/clean/final_clean_temporal.json",
    )


@app.get("/evaluation/models", summary="모델별 매크로 F1 요약")
def models() -> dict:
    """모든 사례에 결과를 냈을 때의 성능입니다.

    판단 보류를 반영한 결과는 `/evaluation/risk-coverage` 에서 확인할 수 있습니다.
    """
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


@app.get("/failures", response_model=FailureReport, summary="실패 사례와 테스트 결과")
def failures(
    run_probes: bool = Query(True, description="재현 검사를 실제로 돌린다"),
) -> FailureReport:
    """개발 중 기록한 실패 사례와 재발 여부를 확인하는 테스트 결과입니다."""
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
