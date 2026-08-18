"""회답 근거 파이프라인 중 **API 없이 돌아가는 모든 경로**.

이 단계는 여섯 단계이고 그중 셋만 돈을 쓴다. 나머지 셋(consolidate · predict ·
status)과 두 개의 --dry-run 경로는 전부 결정론이라 테스트할 수 있는데,
처음에는 하나도 없었다.

테스트가 없는 채로 돌렸다가 `consolidate` 가 "채택된 기준이 하나도 없습니다"
한 줄만 뱉고 죽었고, 왜 0개인지 알 방법이 없었다. 그 뒤에야 진단을 붙였다.
이 파일은 그 진단이 실제로 나오는지까지 확인한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, "scripts/criteria.py", *args],
        cwd=ROOT, capture_output=True, text=True,
    )
    if expect_ok:
        assert proc.returncode == 0, f"{args}\n{proc.stdout}\n{proc.stderr}"
    return proc


def write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def criterion(name: str, question: str, quote: str, implies: str) -> dict:
    return {"name": name, "question": question, "quote": quote, "implies": implies}


# 같은 개념을 조금씩 다르게 쓴 질문 셋 + 확실히 다른 질문 하나
SPLIT = "내부망과 외부망이 물리적으로 분리되어 있는가?"
SPLIT_VARIANT = "내부망과 외부망이 물리적으로 분리되어 있는지 여부인가?"
AFFILIATE = "위탁 대상이 같은 금융지주회사의 계열회사인가?"


@pytest.fixture()
def raw(tmp_path: Path) -> Path:
    rows = [
        {"source": "t", "page": 1, "serial": "1", "sector": "전자금융", "decision": "조치",
         "proposed": 1, "criteria": [criterion("망분리", SPLIT, "내부망과 외부망", "조치")],
         "rejected": [], "input_tokens": 100, "output_tokens": 50},
        {"source": "t", "page": 2, "serial": "2", "sector": "전자금융", "decision": "조치",
         "proposed": 1,
         "criteria": [criterion("망분리 여부", SPLIT_VARIANT, "물리적으로 분리", "조치")],
         "rejected": [], "input_tokens": 100, "output_tokens": 50},
        {"source": "t", "page": 3, "serial": "3", "sector": "공통", "decision": "비조치",
         "proposed": 2,
         "criteria": [criterion("계열회사", AFFILIATE, "계열회사에", "비조치")],
         "rejected": [{"question": "조치 대상인가?", "quote": "x",
                       "rejected_for": ["질문이 결론을 되묻는다"]}],
         "input_tokens": 100, "output_tokens": 50},
        {"source": "t", "page": 4, "serial": "4", "sector": "공통", "decision": "비조치",
         "error": "APIStatusError: 429", "error_detail": "rate limited"},
    ]
    return write(tmp_path / "criteria_raw.jsonl", rows)


# ── consolidate ──────────────────────────────────────────────────
def test_consolidate_groups_similar_questions(raw: Path, tmp_path: Path):
    out = tmp_path / "criteria.jsonl"
    run("consolidate", "--input", str(raw), "--output", str(out), "--min-support", "1")
    merged = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]

    assert len(merged) == 2, [c["question"] for c in merged]
    top = merged[0]
    assert top["support"] == 2, "같은 개념의 두 질문이 묶이지 않았습니다"
    assert top["implies"] == "조치"
    assert top["sources"] == 2


def test_consolidate_assigns_contiguous_ids_by_support(raw: Path, tmp_path: Path):
    """predict 가 id 로 가중치를 색인하므로 0..n-1 이어야 한다."""
    out = tmp_path / "criteria.jsonl"
    run("consolidate", "--input", str(raw), "--output", str(out), "--min-support", "1")
    merged = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [c["id"] for c in merged] == list(range(len(merged)))
    assert merged == sorted(merged, key=lambda c: -c["support"])


def test_consolidate_drops_groups_below_min_support(raw: Path, tmp_path: Path):
    out = tmp_path / "criteria.jsonl"
    run("consolidate", "--input", str(raw), "--output", str(out), "--min-support", "2")
    merged = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(merged) == 1 and merged[0]["support"] == 2


def test_consolidate_diagnoses_when_nothing_was_accepted(tmp_path: Path):
    """한 줄로 죽지 않고 왜 0개인지 말해야 한다. 처음에는 그러지 못했다."""
    raw = write(tmp_path / "raw.jsonl", [
        {"source": "t", "page": 1, "serial": "1", "decision": "조치", "proposed": 2,
         "criteria": [], "rejected": [
             {"question": "조치 대상인가?", "quote": "없는 말",
              "rejected_for": ["질문이 결론을 되묻는다", "인용이 원문에 없다"]},
             {"question": "비조치인가?", "quote": "x", "rejected_for": ["질문이 결론을 되묻는다"]},
         ]},
    ])
    proc = run("consolidate", "--input", str(raw),
               "--output", str(tmp_path / "out.jsonl"), expect_ok=False)
    assert proc.returncode == 1
    assert "질문이 결론을 되묻는다: 2" in proc.stdout, proc.stdout
    assert "인용이 원문에 없다: 1" in proc.stdout
    assert "버려진 기준 예시" in proc.stdout


def test_consolidate_reports_api_failures(tmp_path: Path):
    raw = write(tmp_path / "raw.jsonl", [
        {"source": "t", "page": 1, "serial": "1", "decision": "조치",
         "error": "APIStatusError: 400", "error_detail": "credit balance is too low"},
    ])
    proc = run("consolidate", "--input", str(raw),
               "--output", str(tmp_path / "out.jsonl"), expect_ok=False)
    assert "API 실패 1건" in proc.stdout
    assert "credit balance" in proc.stdout
    assert "--resume" in proc.stdout


# ── predict ──────────────────────────────────────────────────────
GOLD_DEV = [
    {"source": "d", "page": i, "serial": str(i), "pair_index": 1,
     "request": "질의", "label": "조치" if i <= 4 else "비조치"}
    for i in range(1, 13)
]
GOLD_TEST = [
    {"source": "e", "page": i, "serial": str(i), "pair_index": 1,
     "request": "질의", "label": "조치" if i <= 2 else "비조치"}
    for i in range(1, 7)
]


@pytest.fixture()
def predict_inputs(tmp_path: Path) -> dict:
    criteria = write(tmp_path / "criteria.jsonl", [
        {"id": 0, "question": SPLIT, "name": "망분리", "support": 4,
         "sources": 4, "implies": "조치", "quotes": []},
        {"id": 1, "question": AFFILIATE, "name": "계열", "support": 3,
         "sources": 3, "implies": "비조치", "quotes": []},
    ])
    dev = write(tmp_path / "dev.jsonl", GOLD_DEV)
    # 0번 기준은 조치에만 yes, 1번은 모두에게 yes (정보 없음)
    dev_ans = write(tmp_path / "dev_answers.jsonl", [
        {**{k: r[k] for k in ("source", "page", "serial", "pair_index")},
         "answers": (["yes", "yes"] if r["label"] == "조치" else ["no", "yes"])}
        for r in GOLD_DEV
    ])
    test_ans = write(tmp_path / "test_answers.jsonl", [
        {**{k: r[k] for k in ("source", "page", "serial", "pair_index")},
         "answers": (["yes", "no"] if r["label"] == "조치" else ["no", "no"])}
        for r in GOLD_TEST
    ])
    return {"criteria": criteria, "dev": dev, "dev_answers": dev_ans,
            "test_answers": test_ans, "output": tmp_path / "pred.jsonl"}


def test_predict_produces_scorable_predictions(predict_inputs: dict):
    proc = run("predict", "--criteria", str(predict_inputs["criteria"]),
               "--dev", str(predict_inputs["dev"]),
               "--dev-answers", str(predict_inputs["dev_answers"]),
               "--test-answers", str(predict_inputs["test_answers"]),
               "--output", str(predict_inputs["output"]))
    preds = [json.loads(x) for x in
             predict_inputs["output"].read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(preds) == len(GOLD_TEST)
    by_serial = {p["serial"]: p for p in preds}
    # 0번 기준에 yes 인 사례는 조치 쪽으로 가야 한다
    assert by_serial["1"]["predicted"] == "조치", proc.stdout
    assert by_serial["3"]["predicted"] != "조치"
    for p in preds:
        assert p["confidence"] in ("high", "medium", "low")
        assert p["rule"].startswith("criteria:")


def test_predict_marks_no_evidence_as_low_confidence(predict_inputs: dict):
    """근거가 하나도 안 걸린 사례를 자신 있다고 말하면 안 된다."""
    run("predict", "--criteria", str(predict_inputs["criteria"]),
        "--dev", str(predict_inputs["dev"]),
        "--dev-answers", str(predict_inputs["dev_answers"]),
        "--test-answers", str(predict_inputs["test_answers"]),
        "--output", str(predict_inputs["output"]))
    preds = [json.loads(x) for x in
             predict_inputs["output"].read_text(encoding="utf-8").splitlines() if x.strip()]
    silent = [p for p in preds if p["rule"] == "criteria:none"]
    assert silent, "근거 없는 사례가 하나도 없어 검사가 무의미합니다"
    assert all(p["confidence"] == "low" for p in silent)


def test_predict_warns_when_dev_answers_are_incomplete(predict_inputs: dict, tmp_path: Path):
    partial = write(tmp_path / "partial.jsonl", [
        {**{k: r[k] for k in ("source", "page", "serial", "pair_index")},
         "answers": ["yes", "no"]}
        for r in GOLD_DEV[:5]
    ])
    proc = run("predict", "--criteria", str(predict_inputs["criteria"]),
               "--dev", str(predict_inputs["dev"]), "--dev-answers", str(partial),
               "--test-answers", str(predict_inputs["test_answers"]),
               "--output", str(predict_inputs["output"]))
    assert "답이 있는 것은 5건" in proc.stdout, proc.stdout


def test_predict_rejects_a_file_that_is_not_a_criteria_list(predict_inputs: dict):
    proc = run("predict", "--criteria", str(predict_inputs["dev"]),
               "--dev", str(predict_inputs["dev"]),
               "--dev-answers", str(predict_inputs["dev_answers"]),
               "--test-answers", str(predict_inputs["test_answers"]),
               "--output", str(predict_inputs["output"]), expect_ok=False)
    assert "question" in proc.stdout + proc.stderr


# ── status ───────────────────────────────────────────────────────
def test_status_lists_every_step_and_the_next_one(tmp_path: Path):
    proc = run("status", "--raw", str(tmp_path / "없음1.jsonl"),
               "--criteria", str(tmp_path / "없음2.jsonl"),
               "--dev-answers", str(tmp_path / "없음3.jsonl"),
               "--test-answers", str(tmp_path / "없음4.jsonl"),
               "--predictions", str(tmp_path / "없음5.jsonl"))
    assert proc.stdout.count("[ ]") == 5
    assert "다음: 1 extract" in proc.stdout


def test_status_counts_rows_and_failures(raw: Path, tmp_path: Path):
    proc = run("status", "--raw", str(raw),
               "--criteria", str(tmp_path / "없음.jsonl"),
               "--dev-answers", str(tmp_path / "없음2.jsonl"),
               "--test-answers", str(tmp_path / "없음3.jsonl"),
               "--predictions", str(tmp_path / "없음4.jsonl"))
    assert "[x] 1 extract" in proc.stdout
    assert "4건" in proc.stdout and "실패 1" in proc.stdout
    assert "--resume" in proc.stdout


# ── dry-run (요청을 하나도 보내지 않는다) ────────────────────────
def test_extract_dry_run_sends_nothing(tmp_path: Path):
    dev = write(tmp_path / "dev.jsonl", [
        {"source": "t", "page": 1, "serial": "1", "pair_index": 1,
         "request": "질의", "label": "조치"},
    ])
    cases = write(tmp_path / "cases.jsonl", [
        {"source": "t", "page": 1, "serial": "1", "sector": "전자금융", "decision": "조치",
         "fields": {"요청대상행위": "망연계 구간 질의",
                    "판단이유": "내부망과 외부망을 분리하여야"}},
    ])
    proc = run("extract", "--dev", str(dev), "--cases", str(cases),
               "--output", str(tmp_path / "out.jsonl"), "--dry-run")
    assert "추정 비용" in proc.stdout
    assert "요청을 보내지 않습니다" in proc.stdout
    assert "[판단이유]" in proc.stdout
    assert not (tmp_path / "out.jsonl").exists(), "dry-run 인데 파일을 썼습니다"


def test_apply_dry_run_shows_the_criteria_it_would_ask(predict_inputs: dict, tmp_path: Path):
    gold = write(tmp_path / "gold.jsonl", GOLD_TEST)
    proc = run("apply", "--gold", str(gold),
               "--criteria", str(predict_inputs["criteria"]),
               "--output", str(tmp_path / "out.jsonl"), "--dry-run")
    assert "기준 2개" in proc.stdout
    assert SPLIT in proc.stdout
    assert not (tmp_path / "out.jsonl").exists()


def test_apply_prompt_never_contains_the_answer(predict_inputs: dict, tmp_path: Path):
    """적용 단계는 회답을 보지 않는다. 프롬프트에 결론이 들어가면 순환이다.

    gold 레코드에 결론과 판단이유를 **일부러 넣어** 둔다. 프롬프트 생성기가
    요청문 말고 다른 필드를 집어 오면 여기서 걸린다. 비어 있는 레코드로
    검사하면 통과해도 아무것도 증명하지 못한다.
    """
    poisoned = [
        {**r,
         "label": "조치라는정답",
         "판단이유": "판단이유누출표지",
         "decision": "결론누출표지",
         "answer": "회답누출표지"}
        for r in GOLD_TEST
    ]
    gold = write(tmp_path / "gold.jsonl", poisoned)
    proc = run("apply", "--gold", str(gold),
               "--criteria", str(predict_inputs["criteria"]),
               "--output", str(tmp_path / "out.jsonl"), "--dry-run")

    shown = proc.stdout[proc.stdout.index("[요청대상행위]"):]
    for marker in ("조치라는정답", "판단이유누출표지", "결론누출표지", "회답누출표지"):
        assert marker not in shown, f"프롬프트에 '{marker}' 가 새어 들어갔습니다"
    assert "[판단 기준]" in shown
    assert shown.count("[") >= 2, "프롬프트 구조가 예상과 다릅니다"
