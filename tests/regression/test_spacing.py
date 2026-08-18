"""띄어쓰기 복원 회귀 테스트.

경계 라벨이 한 칸 밀리는 버그가 있었다. 학습과 평가가 같은 함수를 쓴 탓에
지표는 버그와 일관되어 F1 0.79 로 멀쩡해 보였고, 출력을 눈으로 보고서야
드러났다 ("후유증으 로", "것 이"). 지표만 믿으면 안 된다는 사례이므로
라벨 생성 자체를 단위 테스트로 못박는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_PATH = ROOT / "models" / "spacing.json"
PROCESSED = ROOT / "data" / "processed"

# 적용 대상(비조치의견서) 문체에서 측정한 값. 전체 평균이 아니다 —
# 복원 대상은 전부 비조치이므로 법령해석이 섞인 평균은 성능을 부풀린다.
NONACTION_F1_FLOOR = 0.80
APPLY_THRESHOLD = -0.25


def test_boundary_labels_are_not_shifted():
    """labels[i] 는 chars[i] 와 chars[i+1] '사이'의 공백이어야 한다."""
    from restore_spacing import iter_boundaries

    chars, labels = iter_boundaries("금융위원회 및 금융감독원은")
    assert chars == "금융위원회및금융감독원은"
    assert len(labels) == len(chars) - 1
    spaced = {(chars[i], chars[i + 1]) for i, v in enumerate(labels) if v}
    assert spaced == {("회", "및"), ("및", "금")}


def test_leading_space_is_ignored():
    from restore_spacing import iter_boundaries

    chars, labels = iter_boundaries("  가나 다")
    assert chars == "가나다"
    assert labels == [0, 1]


def test_roundtrip_labels_reconstruct_original():
    """라벨로 원문을 되짚어 만들 수 있어야 한다."""
    from restore_spacing import iter_boundaries

    original = "금융회사 등이 대통령령으로 정하는 경우"
    chars, labels = iter_boundaries(original)
    rebuilt = chars[0] + "".join(
        (" " if labels[i] else "") + chars[i + 1] for i in range(len(labels))
    )
    assert rebuilt == original


def test_restore_never_removes_existing_spaces():
    """복원기는 공백을 넣기만 한다. 원문의 공백은 신뢰도가 더 높다."""
    from restore_spacing import restore_line

    model = {"contexts": {}, "prior": -99.0}  # 아무것도 삽입하지 않는 모델
    line = "코로나19 펜데믹의후유증으로"
    assert restore_line(model, line, threshold=0.0) == line


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="모델이 없습니다 (train 먼저 실행)")
def test_nonaction_f1_holds():
    from restore_spacing import holdout, load_cases, prf, score_text, split_corpus

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


def test_restored_cases_are_present_and_spaced():
    """복원이 적용된 사례는 원문보다 공백이 늘어야 한다."""
    path = PROCESSED / "cases_nonaction.jsonl"
    if not path.exists():
        pytest.skip("파싱 결과가 없습니다")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    lost = [r for r in rows if "spacing_lost" in r.get("warnings", [])]
    assert lost, "spacing_lost 사례가 사라졌습니다 — 탐지 로직을 확인하세요"

    restored = [r for r in lost if "raw_restored" in r]
    assert len(restored) == len(lost), (
        f"복원 미적용 {len(lost) - len(restored)}건 — "
        "restore_spacing.py apply 를 실행하세요"
    )
    for r in restored:
        before = r["raw"].count(" ")
        after = r["raw_restored"].count(" ")
        assert after > before, f"{r['serial']}: 공백이 늘지 않았습니다"
