"""기저율 표가 **어느 dev 에서 나왔는지**를 실제로 검사한다.

기존 방지 장치(`probes.base_rates_come_from_dev_only`)는 `source == "dev"` 만
본다. 이 저장소에는 이제 dev 가 둘이고, **clean test 168건 중 54건이 legacy
dev 안에 있었다.** 그러므로 "dev 에서 나왔다" 는 더 이상 안전을 뜻하지 않는다.

여기서 보는 것은 이름표가 아니라 **지문**이다 — 표를 만든 행 집합을 다시 세어
같은지 본다. 이름표는 손으로 고칠 수 있지만 지문은 못 고친다.
"""

import json
from pathlib import Path

import pytest

from app.core.io import key_of, load_jsonl
from app.core.paths import DEV_BASE_RATES, EVAL
from app.domain.base_rate_asset import (
    load_validated,
    method_version,
    row_key_digest,
    validate,
    verify,
)
from app.domain.base_rates import compute

CLEAN_TABLE = EVAL / "dev_base_rates_clean.json"
CLEAN_DEV = EVAL / "nonaction_dev_clean.jsonl"
CLEAN_TEST = EVAL / "nonaction_test_clean.jsonl"
LEGACY_DEV = EVAL / "nonaction_dev.jsonl"


def rows(path):
    return [r for r in load_jsonl(path) if r.get("label")]


def table(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestCleanTableIdentity:
    """clean 표는 clean dev 에서 나왔는가 — 재계산으로 확인한다."""

    def test_it_passes_full_verification_against_clean_dev(self):
        problems = verify(table(CLEAN_TABLE), rows(CLEAN_DEV), "clean", "clean_dev")
        assert problems == [], problems

    def test_its_row_fingerprint_is_clean_dev_and_not_legacy_dev(self):
        recorded = table(CLEAN_TABLE)["provenance"]["row_key_digest"]
        assert recorded == row_key_digest(rows(CLEAN_DEV))
        assert recorded != row_key_digest(rows(LEGACY_DEV))

    def test_it_declares_the_split_and_the_size(self):
        got = table(CLEAN_TABLE)
        assert got["split"] == "clean"
        assert got["source"] == "clean_dev"
        assert got["n"] == len(rows(CLEAN_DEV)) == 87

    def test_the_method_that_made_it_is_the_method_running_now(self):
        assert table(CLEAN_TABLE)["provenance"]["method_version"] == method_version()

    def test_no_clean_test_row_could_have_contributed(self):
        """표가 clean dev 만으로 정확히 재현되고, 그 dev 는 test 와 겹치지 않는다.

        둘을 합치면 test 행이 기여할 자리가 없다는 뜻이 된다.
        """
        dev, test = rows(CLEAN_DEV), rows(CLEAN_TEST)
        assert not ({key_of(r) for r in dev} & {key_of(r) for r in test})
        assert table(CLEAN_TABLE)["overall"] == compute(dev)["overall"]


class TestLegacyTableIsStillLegacy:
    def test_the_shipped_table_reproduces_from_legacy_dev_only(self):
        got, legacy = table(DEV_BASE_RATES), compute(rows(LEGACY_DEV))
        assert got["n"] == 85
        assert got["overall"] == legacy["overall"]

    def test_it_does_not_reproduce_from_clean_dev(self):
        assert table(DEV_BASE_RATES)["overall"] != compute(rows(CLEAN_DEV))["overall"]

    def test_it_now_declares_its_own_split_and_validates(self):
        """출처를 다시 찍었다. 값은 그대로이고 이름표와 지문만 붙었다."""
        got = table(DEV_BASE_RATES)
        assert (got["source"], got["split"]) == ("dev", "legacy")
        assert got["provenance"]["row_key_digest"] == row_key_digest(rows(LEGACY_DEV))
        assert validate(got) == []

    def test_the_two_assets_do_not_share_a_fingerprint(self):
        assert (table(DEV_BASE_RATES)["provenance"]["row_key_digest"]
                != table(CLEAN_TABLE)["provenance"]["row_key_digest"])


class TestVerifyCatchesTampering:
    @pytest.mark.parametrize("field,value", [("split", "legacy"),
                                             ("source", "dev"),
                                             ("n", 85)])
    def test_a_relabelled_table_is_rejected(self, field, value):
        forged = dict(table(CLEAN_TABLE))
        forged[field] = value
        assert verify(forged, rows(CLEAN_DEV), "clean", "clean_dev")

    def test_a_table_built_from_the_other_dev_is_rejected(self):
        forged = dict(table(CLEAN_TABLE))
        forged["overall"] = compute(rows(LEGACY_DEV))["overall"]
        assert verify(forged, rows(CLEAN_DEV), "clean", "clean_dev")

    def test_a_forged_fingerprint_is_rejected(self):
        forged = dict(table(CLEAN_TABLE))
        forged["provenance"] = dict(forged["provenance"], row_key_digest="0" * 64)
        assert verify(forged, rows(CLEAN_DEV), "clean", "clean_dev")


class TestClassifierWiring:
    """`--base-rates` 로 어느 표를 쓸지 고를 수 있고, 기본값은 legacy 다."""

    def test_the_default_is_still_the_legacy_asset(self):
        from app.agents import classifier

        assert classifier.BASE_RATES_PATH == DEV_BASE_RATES

    def test_the_flag_defaults_to_the_legacy_path(self):
        from app.agents.classifier import build_parser

        args = build_parser().parse_args(["--input", "a", "--output", "b"])
        assert Path(args.base_rates) == DEV_BASE_RATES
        assert args.dry_run is False

    def test_the_flag_can_choose_the_clean_asset(self):
        from app.agents.classifier import build_parser

        args = build_parser().parse_args(
            ["--input", "a", "--output", "b", "--base-rates", str(CLEAN_TABLE)])
        assert Path(args.base_rates) == CLEAN_TABLE

    @pytest.mark.parametrize("path,identity", [
        (DEV_BASE_RATES, ("dev", "legacy")),
        (CLEAN_TABLE, ("clean_dev", "clean")),
    ])
    def test_both_assets_load_through_the_validating_reader(self, path, identity):
        got, record = load_validated(path)
        assert (got["source"], got["split"]) == identity
        assert record["row_key_digest"] == got["provenance"]["row_key_digest"]
        assert record["n"] == got["n"]

    def test_the_record_carries_everything_needed_to_reproduce(self):
        _, record = load_validated(CLEAN_TABLE)
        assert set(record) == {"path", "source", "split", "n", "row_key_digest",
                               "method_version", "input", "input_sha256"}
