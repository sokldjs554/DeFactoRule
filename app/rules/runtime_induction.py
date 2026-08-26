"""Production E6 runtime capability contract.

The clean E6 asset is already a frozen learned model. Production must not silently learn a
replacement model merely because one learned condition is unavailable at inference time.
Instead, project the frozen asset onto conditions the live Router can actually evaluate.

This is deliberately structural, not performance-driven: any rule containing unsupported
editorial metadata is removed as a whole, its original order is preserved, and no replacement
rule is induced from test results.
"""

from __future__ import annotations

from copy import deepcopy

RUNTIME_ATOM_KINDS = frozenset({"ngram", "length"})


def unsupported_atom_kinds(rule: dict) -> tuple[str, ...]:
    """Return unsupported atom kinds in stable order."""
    return tuple(
        sorted(
            {
                str(atom.get("kind"))
                for atom in rule.get("atoms", [])
                if atom.get("kind") not in RUNTIME_ATOM_KINDS
            }
        )
    )


def project_runtime_asset(frozen_payload: dict) -> dict:
    """Remove only rules the live request-only matcher cannot evaluate.

    Orders are not renumbered. They identify the frozen learned rules and keeping them stable
    makes traces comparable with the clean E6 audit.
    """
    kept = []
    dropped = []
    for rule in frozen_payload.get("rules", []):
        unsupported = unsupported_atom_kinds(rule)
        if unsupported:
            dropped.append(
                {
                    "order": rule.get("order"),
                    "description": rule.get("description"),
                    "unsupported_atom_kinds": list(unsupported),
                    "reason": "condition is unavailable from the live request-only input",
                }
            )
        else:
            kept.append(deepcopy(rule))

    payload = {
        "contract": {
            "source": "frozen clean E6 asset",
            "method": "capability_projection_no_reinduction",
            "runtime_atom_kinds": sorted(RUNTIME_ATOM_KINDS),
            "unsupported_metadata_atoms": ["sector"],
            "reason": "live AgentState receives request text; sector is editorial metadata",
            "preserve_original_rule_order": True,
        },
        "settings": deepcopy(frozen_payload.get("settings", {})),
        "default_label": frozen_payload.get("default_label"),
        "rules": kept,
        "dropped_rules": dropped,
    }
    validate_runtime_asset(payload)
    return payload


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
