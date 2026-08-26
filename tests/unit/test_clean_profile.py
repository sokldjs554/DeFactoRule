"""clean 프로파일이 **세는 방식**을 고정한다.

수치 자체는 데이터가 정한다. 여기서 지키는 것은 세는 규칙이다 — 구간 경계는
도메인 한 곳에서만 오고, 규칙은 먼저 걸린 것이 이긴다. 그리고 Router 와
유도기가 **서로 다르게 발화한다는 사실**을 테스트로 박아 둔다. 그것을 모르고
`R1` 수를 읽으면 규칙이 안 걸린 이유를 영영 못 찾는다.
"""

import json

from app.agents.calibration import band_of as calibration_band
from app.core.paths import RESULTS
from app.domain.similarity import DOUBT, TRUST
from app.evaluation.clean_profile import (
    band_of,
    counts,
    fires_as_induced,
    fires_as_router,
    integrity,
    rule_transfer,
)
from app.evaluation.final_freeze import build_final_freeze
from app.rules.runtime_induction import project_runtime_asset


def row(serial="1", page="1", label="비조치", request="가나다", sector="전자금융",
        source="2024년 사례집.pdf", pair_index="1"):
    return {"source": source, "serial": serial, "page": page,
            "pair_index": pair_index, "label": label, "request": request,
            "sector": sector}


def rule(order, label, atoms, dev_support=4, dev_precision=1.0):
    return {"order": order, "label": label, "atoms": atoms,
            "description": f"r{order}", "dev_support": dev_support,
            "dev_precision": dev_precision}


class TestBands:
    def test_boundaries_come_from_the_domain_not_from_here(self):
        for score in (0.0, DOUBT - 1e-9, DOUBT, 0.4, TRUST - 1e-9, TRUST, 1.0):
            assert band_of(score) == calibration_band(score)


class TestRuleSemantics:
    """유도기와 Router 는 같은 규칙을 다르게 읽는다. 그 간극이 이 테스트다."""

    def test_the_router_is_blind_to_sector_conditions(self):
        sector_rule = rule(1, "기타", [{"kind": "sector", "value": "공통"}])
        r = row(sector="공통")
        assert fires_as_induced(sector_rule, r) is True
        assert fires_as_router(sector_rule, r) is False

    def test_both_agree_on_ngram_conditions(self):
        ngram_rule = rule(1, "비조치", [{"kind": "ngram", "value": "가나"}])
        assert fires_as_induced(ngram_rule, row(request="가나다")) is True
        assert fires_as_router(ngram_rule, row(request="가나다")) is True
        assert fires_as_induced(ngram_rule, row(request="라마바")) is False
        assert fires_as_router(ngram_rule, row(request="라마바")) is False


class TestRuleTransfer:
    def test_first_matching_rule_wins_and_later_rules_do_not_see_it(self):
        rules = [rule(1, "비조치", [{"kind": "ngram", "value": "가"}]),
                 rule(2, "기타", [{"kind": "ngram", "value": "가"}])]
        report = rule_transfer(rules, "조치", [row(request="가")])
        assert report["per_rule"][0]["test_support"] == 1
        assert report["per_rule"][1]["test_support"] == 0
        assert report["rules_that_never_fire"] == [2]

    def test_rows_no_rule_covers_fall_to_the_default_label(self):
        rules = [rule(1, "비조치", [{"kind": "ngram", "value": "없는말"}])]
        rows = [row(request="다른말", label="기타"),
                row(serial="2", request="또다른", label="비조치")]
        report = rule_transfer(rules, "기타", rows)
        assert report["fired"] == 0
        assert report["fell_to_default"] == 2
        assert report["default_correct"] == 1
        assert report["whole_model_accuracy"] == 0.5

    def test_the_sector_rule_changes_the_count_depending_on_who_reads_it(self):
        sector_rule = rule(1, "기타", [{"kind": "sector", "value": "공통"}])
        text_rule = rule(2, "비조치", [{"kind": "ngram", "value": "가"}])
        rows = [row(sector="공통", label="기타")]
        assert rule_transfer([sector_rule], "비조치", rows)["fired"] == 1
        assert rule_transfer(
            [sector_rule], "비조치", rows, matcher=fires_as_router
        )["fired"] == 0

        projected = project_runtime_asset(
            {"settings": {}, "default_label": "비조치", "rules": [sector_rule, text_rule]}
        )
        assert [r["order"] for r in projected["rules"]] == [2]
        assert projected["dropped_rules"][0]["order"] == 1
        assert projected["dropped_rules"][0]["unsupported_atom_kinds"] == ["sector"]


class TestIntegrity:
    def test_it_reports_the_legacy_dev_rows_that_ended_up_in_clean_test(self):
        shared = row(serial="100")
        report = integrity([row(serial="1")], [shared], [shared], [row(serial="2")])
        assert report["clean_test_seen_in_legacy_dev"] == 1
        assert report["dev_test_overlap"] == 0

    def test_it_notices_when_the_row_set_was_not_preserved(self):
        report = integrity([row(serial="1")], [row(serial="2")],
                           [row(serial="1")], [row(serial="3")])
        assert report["union_preserved"] is False

        asset, freeze = build_final_freeze(write=False)
        assert len(asset["rules"]) == 10
        assert [r["order"] for r in asset["dropped_rules"]] == [8]
        assert freeze["final"]["answered"] == 76
        assert freeze["final"]["abstained"] == 92
        assert freeze["final"]["correct"] == 63
        assert freeze["final"]["wrong"] == 13
        assert freeze["transitions"]["changed"] == []

        committed_asset = json.loads(
            (RESULTS / "clean" / "e6_rules_clean_runtime.json").read_text(encoding="utf-8")
        )
        committed_freeze = json.loads(
            (RESULTS / "clean" / "final_clean_temporal.json").read_text(encoding="utf-8")
        )
        assert asset == committed_asset
        assert freeze == committed_freeze


class TestCounts:
    def test_keys_are_sorted_so_reports_can_be_diffed(self):
        assert list(counts(["나", "가", "나"])) == ["가", "나"]
