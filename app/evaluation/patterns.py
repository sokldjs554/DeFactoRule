"""재발 패턴 가드 — 개별 사례가 아니라 **패턴**을 지킨다.

레지스트리의 probe 는 과거의 그 자리를 지킨다. 하지만 같은 실수가 **다른
자리에서** 다시 나오는 것은 막지 못한다. 실제로 "걸러낸 것을 기록하지 않는다"
는 네 곳에서 같은 모양으로 나왔다.

    IN-02  API 오류에서 상태 코드만 남겼다          -> 39건이 왜 죽었는지 모름
    EV-09  결측 검사가 오류 행만 셌다                -> 156/170 이 "결측 0"
    (기준) 버린 기준을 화면에만 찍었다                -> 0개인 이유를 모름
    (규칙) 문턱에서 떨어진 후보 165개를 안 남겼다     -> 후보가 없었나 문턱이 높았나

넷 다 따로 고쳤고, 넷 다 probe 를 붙였다. 그런데 다섯 번째를 막을 장치가
없었다. 이 파일이 그 장치다.

## 무엇을 하는가

파이프라인에서 **항목을 걸러내는 단계**를 등록해 두고, 각 단계에 거부당할
입력을 실제로 넣어 본다. 그 결과물에 **버린 이유가 들어 있는지**를 본다.
"기록하는 코드가 있다" 가 아니라 "기록이 실제로 나온다" 를 보는 것이다.

새 단계를 만들면서 등록을 잊으면 이 가드는 그것을 모른다 — 그 한계는 분명히
해 둔다. 다만 등록된 단계가 기록을 멈추면 즉시 걸린다.
"""

from __future__ import annotations

from typing import Callable

Result = tuple[bool, str]


# ── 개별 단계 검사 ───────────────────────────────────────────────
def _stage_rule_induction() -> Result:
    """규칙 학습기: 문턱에서 떨어진 후보를 이유와 함께 남기는가."""
    from app.domain.labels import NON_ACTIONS
    from app.rules.induction import induce_with_audit

    rows = (
        [{"source": "t", "page": i, "serial": str(i), "pair_index": 1, "sector": "공통",
          "request": f"내부망과 외부망의 망연계 구간 질의 {i}", "label": "조치"}
         for i in range(1, 7)]
        + [{"source": "t", "page": 10 + i, "serial": str(10 + i), "pair_index": 1,
            "sector": "공통", "request": f"겸영업무 신고 대상 여부 질의 {i}",
            "label": "비조치"} for i in range(1, 9)]
    )
    _, _, discards = induce_with_audit(
        rows, NON_ACTIONS, min_support=4, min_precision=0.999, max_depth=2
    )
    if not len(discards):
        return False, "문턱을 0.999 로 올렸는데 버린 후보가 없다"
    if not all(r["rejected_for"] for r in discards.records()):
        return False, "이유 없이 버린 항목이 있다"
    return True, f"버린 후보 {len(discards)}개 · 이유 {len(discards.summary()['reasons'])}종"


def _stage_criteria_validation() -> Result:
    """기준 검증: 버린 기준을 이유와 함께 붙드는가."""
    from app.agents.criteria import validate_criterion
    from app.core.audit import Discards

    bad = {"name": "", "question": "조치 대상인가?", "quote": "없는 말", "implies": "몰라"}
    discards = Discards("criteria-validation")
    kept = discards.keep_if(bad, validate_criterion(bad, "원문 본문"))
    if kept:
        return False, "명백히 잘못된 기준이 통과했다"
    reasons = discards.records()[0]["rejected_for"]
    if len(reasons) < 2:
        return False, f"이유를 하나만 남겼다: {reasons}"
    return True, f"버린 기준 1개 · 이유 {len(reasons)}가지 전부 기록"


def _stage_field_extraction() -> Result:
    """항목 추출: 못 찾은 항목을 경고로 남기는가."""
    from app.extraction.casebook import split_fields

    _, warnings = split_fields("요청대상행위\n갑 행위", ["요청대상행위", "판단이유"])
    if not warnings:
        return False, "판단이유가 없는데 경고가 없다"
    return True, f"경고 {warnings}"


def _stage_query_splitting() -> Result:
    """질의 분할: 순번이 어긋날 때 그 사실을 남기는가."""
    from app.extraction.splitting import split_case

    rows = split_case({
        "source": "t", "doc_type": "interpretation", "serial": "1", "page": 1,
        "sector": "공통", "warnings": [],
        "fields": {"질의요지": "①갑 ②을 ③병", "회답": "①가능 ②불가", "이유": ""},
    })
    if "mark_mismatch" not in rows[0]["case_warnings"]:
        return False, "순번이 어긋났는데 경고가 없다"
    return True, f"경고 {rows[0]['case_warnings']}"


def _stage_gold_masking() -> Result:
    """gold 생성: 가린 결론 표현의 개수를 남기는가."""
    from app.evaluation.gold_nonaction import build

    rows = [
        {"source": "t", "page": i, "serial": str(i), "sector": "공통",
         "decision": "비조치" if i % 2 else "조치",
         "fields": {"요청대상행위": f"비조치를 요청드립니다 {i}"}}
        for i in range(1, 10)
    ]
    dev, test = build(rows)
    sample = (dev + test)[0]
    if "masked_leaks" not in sample:
        return False, "가린 개수를 남기지 않는다"
    if not any(r["masked_leaks"] for r in dev + test):
        return False, "누출 표현을 넣었는데 하나도 가리지 않았다"
    return True, f"gold {len(dev) + len(test)}건 · 가린 표현이 기록됨"


def _stage_api_errors() -> Result:
    """API 오류: 메시지와 본문을 남기는가.

    네트워크를 쓰지 않으므로 이 하나만 **구현 검사**다. 실제 호출로 확인할 수
    없어 오류 레코드가 담기로 한 항목이 코드에 있는지를 본다. 다른 단계보다
    약한 검사라는 것을 분명히 해 둔다.
    """
    from pathlib import Path

    from app.core.paths import ROOT

    src = Path(ROOT / "app" / "infrastructure" / "anthropic_client.py").read_text(
        encoding="utf-8"
    )
    missing = [k for k in ("error_detail", "error_body", "prompt_chars") if k not in src]
    if missing:
        return False, f"오류 레코드에 없는 항목: {missing}"
    return True, "error_detail · error_body · prompt_chars (구현 검사)"


FILTER_STAGES: dict[str, Callable[[], Result]] = {
    "rule-induction": _stage_rule_induction,
    "criteria-validation": _stage_criteria_validation,
    "field-extraction": _stage_field_extraction,
    "query-splitting": _stage_query_splitting,
    "gold-masking": _stage_gold_masking,
    "api-errors": _stage_api_errors,
}


# ── 패턴 가드 ────────────────────────────────────────────────────
def every_filter_stage_records_its_discards() -> Result:
    """등록된 모든 걸러내기 단계가 버린 것을 기록하는가."""
    failures = []
    details = []
    for name, check in sorted(FILTER_STAGES.items()):
        try:
            ok, detail = check()
        except Exception as exc:  # noqa: BLE001 — 단계가 죽는 것도 결과다
            ok, detail = False, f"예외 {type(exc).__name__}: {exc}"
        details.append(f"{name}: {detail}")
        if not ok:
            failures.append(name)
    if failures:
        return False, f"기록하지 않는 단계 {failures} — " + " / ".join(details)
    return True, f"{len(FILTER_STAGES)}개 단계 전부 기록함"


def comparisons_align_their_samples() -> Result:
    """모델 비교가 공통 표본만 쓰는가.

    표본이 다른 것을 나란히 놓는 실수는 세 번 나왔다 — 30건 예측을 170건 gold 로
    채점(EV-01), 결측 39건짜리로 3-way 비교(EV-08), 156/170 을 "결측 0" 으로
    통과(EV-09). 비교 함수가 스스로 표본을 맞추는지 본다.
    """
    from app.evaluation.comparison import aligned_pairs

    gold = {
        ("t", i, str(i), 1): {"label": "비조치" if i % 2 else "조치"}
        for i in range(1, 11)
    }
    full = {k: {"predicted": "비조치"} for k in gold}
    partial = {k: {"predicted": "비조치"} for i, k in enumerate(gold) if i < 6}

    keys, by_model, truth = aligned_pairs(gold, {"full": full, "partial": partial})
    if len(keys) != 6:
        return False, f"공통 표본을 {len(keys)}건으로 잡았다 (기대 6)"
    if not all(len(v) == len(keys) for v in by_model.values()):
        return False, "모델마다 표본 길이가 다르다"
    if len(truth) != len(keys):
        return False, "정답 길이가 표본과 다르다"
    return True, f"gold 10건 · 한쪽만 6건 -> 공통 표본 {len(keys)}건으로 정렬"
