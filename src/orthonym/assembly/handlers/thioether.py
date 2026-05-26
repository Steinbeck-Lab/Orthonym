"""Phase 160 thioether handler — Tier B shim (gate 0.40).

Verbatim move of composer.py:970-992 dispatch logic. Wraps
``rules.sulfur.name_sulfide`` with the cyclic-thioether + fused-heterocycle
skip guards that the inline branch enforced.

IUPAC cite: P-66.5.2.4 (sulfides; functional class naming).

References:
- composer.py:970-992 (inline thioether branch; REMOVED at this commit).
- rules.sulfur.name_sulfide — chemical-logic body.
- data.fused_heterocycles.match_fused_heterocycle_core — fused-ring guard.
- 160-AUDIT-DECOMP.md § 1 row 'thioether' + § 2.21 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_thioether(features: Any) -> bool:
    """Mirrors composer.py:970 (``features.principal_group == 'thioether'``)."""
    return getattr(features, 'principal_group', None) == 'thioether'


def name_thioether(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B thioether handler with cyclic + fused-heterocycle skip guards."""
    from ..candidate_pool import get_current_pool
    from ..composer import _enrich_handler_name, _inject_stereo_if_missing
    from ...data.fused_heterocycles import match_fused_heterocycle_core
    from ...rules.sulfur import name_sulfide

    # Skip cyclic thioethers (named as heterocycles).
    ring_type = getattr(features, 'ring_type', None)
    if ring_type and ring_type.startswith('heterocyclic'):
        return None

    # Skip known fused heterocycles (phenothiazine, thianthrene, etc.).
    if match_fused_heterocycle_core(features.mol) is not None:
        return None

    matches = features.functional_groups.get('thioether', [])
    if not matches:
        return None

    # SMARTS match gives (S, C, C) — sulfur is first.
    sulfur_idx = matches[0][0]
    name = name_sulfide(features.mol, sulfur_idx)
    if not name:
        return None

    name = _enrich_handler_name(features, name, "thioether")

    pool = get_current_pool()
    cand = pool.add(name, "thioether", features)
    if cand is None:
        return None

    # composer.py:991 inline: _inject_stereo_if_missing(features, cand.name)
    final_name = _inject_stereo_if_missing(features, cand.name)
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="thioether", iupac_section_cite="P-63.2", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_thioether", "_is_thioether"]
