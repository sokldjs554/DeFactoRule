"""**위조한 기저율 표는 전부 거부되어야 한다.**

옛 가드는 `source == "dev"` 한 줄이었다. 그것은 글자 하나를 보는 검사이므로,
글자 하나만 고치면 통과한다. 여기서 확인하는 것은 새 검사가 **글자가 아니라
지문**을 본다는 사실이다 — 각 항목마다 하나씩 망가뜨려 보고, 전부 걸리는지 센다.
"""

import json

import pytest

from app.core.paths import DEV_BASE_RATES, EVAL
from app.domain.base_rate_asset import (
    ProvenanceError,
    build,
    load_validated,
    validate,
)

CLEAN_TABLE = EVAL / "dev_base_rates_clean.json"
CLEAN_DEV = EVAL / "nonaction_dev_clean.jsonl"
CLEAN_TEST = EVAL / "nonaction_test_clean.jsonl"
LEGACY_DEV = EVAL / "nonaction_dev.jsonl"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    from app.core.io import load_jsonl

    return [r for r in load_jsonl(path) if r.get("label")]


def written(tmp_path, table):
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    return path


class TestTheRealAssetsPass:
    @pytest.mark.parametrize("path", [DEV_BASE_RATES, CLEAN_TABLE])
    def test_a_genuine_asset_validates(self, path):
        assert validate(load(path)) == []


class TestRelabelling:
    def test_legacy_asset_relabelled_as_clean_is_rejected(self):
        forged = dict(load(DEV_BASE_RATES), split="clean")
        assert validate(forged)

    def test_clean_asset_relabelled_as_dev_is_rejected(self):
        forged = dict(load(CLEAN_TABLE), source="dev")
        assert validate(forged)

    def test_a_source_split_pair_outside_the_allowed_set_is_rejected(self):
        forged = dict(load(CLEAN_TABLE), source="clean_dev", split="legacy")
        assert validate(forged)

    def test_relabelling_both_at_once_is_still_rejected(self):
        """이름표를 둘 다 legacy 로 바꿔도, 그 split 이 가리키는 파일이 아니다.

        내부 정합성만 보면 이 표는 앞뒤가 맞는다 — 지문도 분포도 자기가 적어 둔
        clean dev 와 일치한다. 그래서 **이름표를 행 집합에 묶는 규칙**이 없으면
        통과한다. 이 테스트가 그 구멍을 잡아서 `SPLIT_DEV_FILE` 이 생겼다.
        """
        forged = dict(load(CLEAN_TABLE), source="dev", split="legacy")
        problems = validate(forged)
        assert any("에서 나와야 하는데" in p for p in problems), problems


class TestFingerprints:
    def test_a_wrong_row_key_digest_is_rejected(self):
        table = load(CLEAN_TABLE)
        table["provenance"] = dict(table["provenance"], row_key_digest="0" * 64)
        assert any("행 지문" in p for p in validate(table))

    def test_a_wrong_input_sha256_is_rejected(self):
        table = load(CLEAN_TABLE)
        table["provenance"] = dict(table["provenance"], input_sha256="0" * 64)
        assert any("입력 파일 지문" in p for p in validate(table))

    def test_a_wrong_method_version_is_rejected(self):
        table = load(CLEAN_TABLE)
        table["provenance"] = dict(table["provenance"], method_version="deadbeef")
        assert any("셈법 지문" in p for p in validate(table))

    def test_a_missing_provenance_block_is_rejected(self):
        table = {k: v for k, v in load(CLEAN_TABLE).items() if k != "provenance"}
        assert any("provenance" in p for p in validate(table))

    def test_a_wrong_n_is_rejected(self):
        assert any("n 이" in p for p in validate(dict(load(CLEAN_TABLE), n=85)))


class TestWrongInputFile:
    def test_an_asset_pointing_at_the_other_dev_file_is_rejected(self):
        """`input` 만 바꿔치기하면 지문 세 개가 동시에 어긋난다."""
        table = load(CLEAN_TABLE)
        table["provenance"] = dict(table["provenance"],
                                   input="data/eval/nonaction_dev.jsonl")
        problems = validate(table)
        assert any("입력 파일 지문" in p for p in problems)
        assert any("행 지문" in p for p in problems)

    def test_an_asset_built_from_the_test_split_is_rejected(self):
        """빌더의 가드를 우회해 만들어도 검증에서 걸린다."""
        table = build(rows(CLEAN_TEST), CLEAN_TEST, "clean", "clean_dev")
        assert any("test 파일" in p for p in validate(table))

    def test_an_asset_built_from_legacy_dev_but_labelled_clean_is_rejected(self):
        table = build(rows(LEGACY_DEV), LEGACY_DEV, "clean", "clean_dev")
        problems = validate(table)
        assert problems, "legacy dev 로 만든 표가 clean 이라고 주장하는데 통과했다"


class TestTheReaderRefusesToHandBackABadTable:
    def test_load_validated_raises_instead_of_returning(self, tmp_path):
        path = written(tmp_path, dict(load(CLEAN_TABLE), split="legacy"))
        with pytest.raises(ProvenanceError) as exc:
            load_validated(path)
        assert "출처를 확인하지 못했습니다" in str(exc.value)

    def test_a_missing_file_raises_the_same_error(self, tmp_path):
        with pytest.raises(ProvenanceError):
            load_validated(tmp_path / "없는파일.json")
