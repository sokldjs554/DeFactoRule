"""띄어쓰기 복원 성능 바닥선.

**적용 대상 분포에서만 잰다.** 보류 표본은 법령해석 84 / 비조치 21 인데
복원을 실제로 적용하는 문서는 전부 비조치다. 섞어서 평균을 내면 0.855 가
나오지만 그건 쓰이지 않을 문체가 점수를 올려준 것이고, 실제 대상에서는
0.816 이다. 7%p 를 부풀려 보고할 뻔했다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.extraction.spacing import holdout, load_cases, prf, score_text, split_corpus

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "spacing.json"
PROCESSED = ROOT / "data" / "processed"

NONACTION_F1_FLOOR = 0.80
APPLY_THRESHOLD = -0.25


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="모델이 없습니다 (train 먼저 실행)")
def test_nonaction_f1_holds():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    ok, _ = split_corpus(load_cases(PROCESSED))
    _, test = holdout(ok)
    subset = [c for c in test if c["doc_type"] == "nonaction"]
    assert subset, "비조치 보류 표본이 없습니다"

    tp = pred = gold = 0
    for c in subset:
        a, b, g = score_text(model, c["raw"], APPLY_THRESHOLD)
        tp, pred, gold = tp + a, pred + b, gold + g
    _, _, f1 = prf(tp, pred, gold)
    assert f1 >= NONACTION_F1_FLOOR, f"비조치 F1 {f1:.3f} < {NONACTION_F1_FLOOR}"
