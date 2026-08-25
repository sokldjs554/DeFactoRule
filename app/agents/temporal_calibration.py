"""Temporal-policy-aware risk calibration for precedent retrieval.

The existing calibration module intentionally preserves historical no-filter
experiments.  This module applies the same risk-table semantics after restricting
LOO neighbours to precedents that would have been temporally eligible at the
request point.
"""

from __future__ import annotations

from app.agents.calibration import (
    DOUBT,
    TRUST,
    bands_are_separable,
    by_neighbor_label,
    joint_cells,
    risk_table,
)
from app.domain.temporal import TemporalPolicy, precedent_is_eligible
from app.evaluation.confusable import cosine, weighted_vector


def temporal_loo_links(
    rows: list[dict],
    idf: dict[str, float],
    policy: TemporalPolicy = "serial",
) -> list[dict]:
    """Find each row's nearest *eligible past* neighbour, excluding self.

    Rows with no eligible predecessor are retained with ``has_neighbor=False`` so
    the caller can audit coverage, but they must not be treated as a successful
    precedent-following trial when estimating risk.
    """
    vecs = [weighted_vector(r["request"], idf) for r in rows]
    out: list[dict] = []

    for i, row in enumerate(rows):
        best_j, best = -1, -1.0
        for j, candidate in enumerate(rows):
            if i == j:
                continue
            if not precedent_is_eligible(candidate, row, policy):
                continue
            score = cosine(vecs[i], vecs[j])
            if score > best:
                best, best_j = score, j

        neighbor = rows[best_j] if best_j >= 0 else None
        similarity = best if neighbor else 0.0
        out.append(
            {
                "similarity": similarity,
                "neighbor_label": neighbor["label"] if neighbor else None,
                "true_label": row["label"],
                "wrong": bool(neighbor) and neighbor["label"] != row["label"],
                "has_neighbor": neighbor is not None,
                "neighbor_serial": str(neighbor.get("serial")) if neighbor else None,
                "request_serial": str(row.get("serial")),
            }
        )

    return out


def calibrate_temporal(
    dev_rows: list[dict],
    idf: dict[str, float],
    policy: TemporalPolicy = "serial",
) -> dict:
    """Calibrate precedent risk under the same temporal policy used at inference."""
    from app.agents.calibration import band_of

    links = temporal_loo_links(dev_rows, idf, policy)
    usable = [x for x in links if x["has_neighbor"]]
    for link in usable:
        link["band"] = band_of(link["similarity"])

    table = risk_table(usable)
    separable, detail = bands_are_separable(table)
    cells = joint_cells(usable)
    wrong = sum(1 for x in usable if x["wrong"])

    return {
        "n_dev": len(dev_rows),
        "n_with_eligible_precedent": len(usable),
        "n_without_eligible_precedent": len(dev_rows) - len(usable),
        "temporal_policy": policy,
        "thresholds": {"trust": TRUST, "doubt": DOUBT},
        "overall_risk": wrong / len(usable) if usable else None,
        "by_band": table,
        "by_neighbor_label": by_neighbor_label(usable),
        "joint_cell_counts": cells,
        "sparse_cells": sum(1 for v in cells.values() if v <= 2),
        "bands_separable": separable,
        "separability_detail": detail,
    }
