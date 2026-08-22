"""감사가 **재는 방식**을 고정한다.

감사는 값을 바꾸지 않는다. 그러므로 여기서 지킬 것은 하나다 — 후보 문턱에서
구간표를 만드는 셈법이 production 보정과 **같은 셈법**이어야 한다. 다르면
"legacy 값이 clean dev 에서도 통과한다" 는 문장이 무의미해진다.
"""

from app.agents.calibration import risk_table
from app.domain.similarity import DOUBT, TRUST
from app.evaluation.protocol_audit import (
    atom_kinds,
    band_table_at,
    gap_around,
    provenance_split,
)


def link(similarity, wrong):
    return {"similarity": similarity, "wrong": wrong,
            "band": "trust" if similarity >= TRUST else (
                "middle" if similarity >= DOUBT else "doubt"),
            "neighbor_label": "비조치", "true_label": "비조치"}


LINKS = [link(0.9, False), link(0.7, True), link(0.3, False),
         link(0.05, True), link(0.01, True)]


class TestBandTable:
    def test_at_production_thresholds_it_equals_the_production_table(self):
        assert band_table_at(LINKS, DOUBT, TRUST)["by_band"] == risk_table(LINKS)

    def test_moving_the_cut_moves_the_rows(self):
        wide = band_table_at(LINKS, 0.02, 0.60)["by_band"]
        assert wide["doubt"]["n"] == 1        # 0.01 만 남는다
        assert wide["middle"]["n"] == 2       # 0.05 · 0.30


class TestGapAround:
    def test_it_names_the_nearest_value_on_each_side(self):
        gap = gap_around([0.1, 0.2, 0.5], 0.3)
        assert (gap["lower"], gap["upper"]) == (0.2, 0.5)
        assert (gap["n_below"], gap["n_above"]) == (2, 1)

    def test_an_empty_side_is_reported_as_none_not_invented(self):
        assert gap_around([0.5, 0.9], 0.1)["lower"] is None


class TestAtomKinds:
    def test_sector_is_flagged_and_ngram_and_length_are_not(self):
        rules = [
            {"order": 1, "description": "a", "atoms": [{"kind": "ngram", "value": "가"}]},
            {"order": 2, "description": "b", "atoms": [{"kind": "length", "value": "짧음"}]},
            {"order": 3, "description": "c", "atoms": [{"kind": "sector", "value": "공통"}]},
        ]
        report = atom_kinds(rules)
        assert report["n_rules_unreadable"] == 1
        assert report["unreadable_by_router"][0]["order"] == 3
        assert report["kinds"] == {"length": 1, "ngram": 1, "sector": 1}


class TestProvenanceSplit:
    def test_rows_are_split_by_whether_legacy_dev_had_seen_them(self):
        def row(serial, label):
            return {"source": "s", "page": 1, "serial": serial, "pair_index": 1,
                    "label": label, "request": "x"}

        gold = [row("1", "비조치"), row("2", "기타")]
        preds = [{"source": "s", "page": 1, "serial": "1", "pair_index": 1,
                  "predicted": "비조치", "abstained": False},
                 {"source": "s", "page": 1, "serial": "2", "pair_index": 1,
                  "predicted": "비조치", "abstained": False}]
        out = provenance_split(gold, preds, [row("1", "비조치")])
        assert out["legacy_dev_에_있던_행"]["n"] == 1
        assert out["legacy_dev_에_있던_행"]["accuracy_on_answered"] == 1.0
        assert out["처음_보는_행"]["n"] == 1
        assert out["처음_보는_행"]["accuracy_on_answered"] == 0.0
