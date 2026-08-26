"""Production E6 induction contract.

Legacy E6 intentionally keeps its historical atom vocabulary (`ngram`, `sector`, `length`)
for reproducibility. The live Router only receives request text, so editorial `sector`
metadata is not a valid production condition. This module reuses the same inducer while
restricting candidate atoms to kinds the runtime matcher can actually evaluate.
"""

from __future__ import annotations

from app.rules.induction import (
    BEAM,
    MAX_DEPTH,
    MAX_DF_RATIO,
    MIN_DF,
    MIN_PRECISION,
    MIN_SUPPORT,
    NGRAM_LENGTHS,
    Rule,
    coverage_masks,
    induce,
    mine_atoms,
)

RUNTIME_ATOM_KINDS = frozenset({"ngram", "length"})


def runtime_atoms(rows: list[dict]):
    """Build the normal E6 vocabulary, excluding conditions unavailable at inference time."""
    candidates = [a for a in mine_atoms(rows) if a.kind in RUNTIME_ATOM_KINDS]
    # Maximalize/deduplicate only after unsupported atoms are removed. Filtering after
    # coverage dedup could accidentally let a `sector` atom hide an equivalent text atom.
    return list(coverage_masks(rows, candidates))


def induce_runtime(rows: list[dict]) -> tuple[list[Rule], str]:
    """Run the historical inducer with the production-capable atom vocabulary only."""
    return induce(rows, atoms=runtime_atoms(rows))


def serialize_runtime_rules(rules: list[Rule], default: str) -> dict:
    """Stable JSON payload for the clean production E6 asset."""
    return {
        "contract": {
            "runtime_atom_kinds": sorted(RUNTIME_ATOM_KINDS),
            "unsupported_metadata_atoms": ["sector"],
            "reason": "live AgentState receives request text; sector is editorial metadata",
        },
        "settings": {
            "ngram_lengths": list(NGRAM_LENGTHS),
            "min_df": MIN_DF,
            "max_df_ratio": MAX_DF_RATIO,
            "min_support": MIN_SUPPORT,
            "min_precision": MIN_PRECISION,
            "max_depth": MAX_DEPTH,
            "beam": BEAM,
        },
        "default_label": default,
        "rules": [
            {
                "order": r.order,
                "label": r.label,
                "description": r.describe(),
                "atoms": [{"kind": a.kind, "value": a.value} for a in r.atoms],
                "dev_support": r.dev_support,
                "dev_precision": r.dev_precision,
                "test_support": None,
                "test_precision": None,
            }
            for r in rules
        ],
    }


def validate_runtime_asset(payload: dict) -> None:
    """Fail closed if a production rule contains an atom the Router cannot evaluate."""
    unsupported = [
        (rule.get("order"), atom.get("kind"))
        for rule in payload.get("rules", [])
        for atom in rule.get("atoms", [])
        if atom.get("kind") not in RUNTIME_ATOM_KINDS
    ]
    if unsupported:
        raise ValueError(f"runtime-unsupported E6 atoms: {unsupported}")
