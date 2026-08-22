"""기저율 표가 **어느 dev 에서 나왔는지**를 실제로 검사한다.

기존 방지 장치(`probes.base_rates_come_from_dev_only`)는 `source == "dev"` 만
본다. 이 저장소에는 이제 dev 가 둘이고, **clean test 168건 중 54건이 legacy
dev 안에 있었다.** 그러므로 "dev 에서 나왔다" 는 더 이상 안전을 뜻하지 않는다.

여기서 보는 것은 이름표가 아니라 **지문**이다 — 표를 만든 행 집합을 다시 세어
같은지 본다. 이름표는 손으로 고칠 수 있지만 지문은 못 고친다.
"""

import json

import pytest

from app.core.io import key_of, load_jsonl
from app.core.paths import DEV_BASE_RATES, EVAL
from app.domain.base_rates import compute
from app.evaluation.base_rate_asset import method_version, row_key_digest, verify

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

    def test_the_old_guard_cannot_tell_the_two_apart(self):
        """왜 이 파일이 필요한지 — 옛 검사는 legacy 표를 그대로 통과시킨다."""
        assert table(DEV_BASE_RATES).get("source") == "dev"      # 옛 검사 통과
        assert "split" not in table(DEV_BASE_RATES)              # 그런데 split 이 없다


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


class TestClassifierWiringIsNotReadyYet:
    """**현재 상태를 못 박아 둔다.** 고친 것이 아니라 확인한 것이다.

    `classifier.py` 는 `BASE_RATES_PATH` 를 모듈 상수로 들고 있고 인자가 없다.
    그리고 `source != "dev"` 면 종료한다 — clean 표의 `source` 는 `clean_dev`
    이므로 지금 배선으로는 **읽히지도, 통과하지도 않는다.**
    """

    def test_the_path_is_a_module_constant_with_no_way_to_choose(self):
        from app.agents import classifier

        assert classifier.BASE_RATES_PATH == DEV_BASE_RATES

    def test_the_existing_guard_would_reject_the_clean_table(self):
        assert table(CLEAN_TABLE).get("source") != "dev"
