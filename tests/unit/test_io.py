"""레코드 키와 JSONL 입출력.

키가 어긋나면 예외가 나지 않고 매칭 건수가 조용히 줄어든다. 30건 예측을
170건 기준선과 비교하면서 커버리지 17.6%를 못 보고 지나간 적이 있다.
"""

from __future__ import annotations

from app.core.io import key_of, load_jsonl, write_json, write_jsonl

ROW = {"source": "2025_비조치", "page": 12, "serial": "25-001", "pair_index": 1}


def test_key_uses_all_four_fields():
    assert key_of(ROW) == ("2025_비조치", 12, "25-001", 1)


def test_same_serial_in_different_sources_is_a_different_key():
    other = dict(ROW, source="2024_비조치")
    assert key_of(ROW) != key_of(other)


def test_split_pairs_of_one_case_are_different_keys():
    assert key_of(ROW) != key_of(dict(ROW, pair_index=2))


def test_serial_may_be_absent(tmp_path):
    assert key_of(dict(ROW, serial=None)) == ("2025_비조치", 12, None, 1)


def test_jsonl_roundtrip_preserves_korean(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    rows = [{"a": "비조치", "n": 1}, {"a": "판단유보", "n": 2}]
    assert write_jsonl(path, rows) == 2
    assert load_jsonl(path) == rows
    assert "비조치" in path.read_text(encoding="utf-8"), "ensure_ascii 로 깨지면 안 된다"


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"a": 2}\n', encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_write_json_creates_parents(tmp_path):
    path = tmp_path / "deep" / "report.json"
    write_json(path, {"k": "값"})
    assert '"값"' in path.read_text(encoding="utf-8")
