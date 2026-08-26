"""Temporary C-5 probe. Remove after freezing the deterministic outputs."""

import json

from app.evaluation.final_freeze import build_final_freeze


def test_c5_probe_outputs() -> None:
    asset, payload = build_final_freeze(write=False)
    probe = {
        "runtime_asset": asset,
        "c3_anchor": payload["c3_anchor_recomputed"],
        "runtime_e6": payload["runtime_e6"],
        "final": payload["final"],
        "delta": payload["delta"],
        "transitions": payload["transitions"],
    }
    raise AssertionError("C5_PROBE=" + json.dumps(probe, ensure_ascii=False, separators=(",", ":")))
