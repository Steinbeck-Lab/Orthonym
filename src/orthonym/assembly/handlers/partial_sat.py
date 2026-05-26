"""Phase 160 partial_sat handler — Tier B shim (gate 0.40).

Verbatim move of composer.py:1116-1131 dispatch logic. Wraps the inline
``_try_partially_saturated_carbocycle(features.mol)`` check + enrichment +
pool.add gate.

Per CONTEXT D-24 incremental migration: the underlying
``_try_partially_saturated_carbocycle`` body STAYS in composer.py during
Plan-02 and moves to this module in Plan-03 commit 03-10.

Byte-identical contract: same mutex as polycyclic — the inline gate is
nested under ``if not _complex_ring_accepted and not chain_is_parent`` so
the predicate here uses ``not _is_complex_ring_system`` for the fast-path.
The fallback for complex-ring-rejection cases lives in the inline
partial_sat block at composer.py:1112+ (unchanged in Plan-02).

IUPAC cite: P-25.3 (partially saturated carbocycles; tetrahydronaphthalene).

References:
- composer.py:1116-1131 (inline partial_sat branch; TRIMMED at this commit).
- composer.py:_try_partially_saturated_carbocycle (STAYS until 03-10).
- composer.py:_enrich_handler_name (STAYS until 03-10).
- 160-AUDIT-DECOMP.md § 1 row 'partial_sat' + § 2.27 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_partial_sat(features: Any) -> bool:
    """Predicate: is_cyclic AND not chain_is_parent AND not complex ring system.

    WR-02: the ``_is_complex_ring_system`` SMARTS check is memoized on the
    features object via ``cached_is_complex_ring_system`` so partial_sat,
    polycyclic, and ring_ester predicates share one evaluation per dispatch
    instead of three. CONTEXT D-25 predicate purity is preserved — the cache
    is per-features-instance state owned by features itself.
    """
    from ._handler_shared import cached_is_complex_ring_system

    if not getattr(features, 'is_cyclic', False):
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    if cached_is_complex_ring_system(features):
        return False
    return True


def name_partial_sat(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B partial_sat handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name, _try_partially_saturated_carbocycle,
    )

    partial_sat_name = _try_partially_saturated_carbocycle(features.mol)
    if not partial_sat_name:
        return None

    partial_sat_name = _enrich_handler_name(features, partial_sat_name, "partial_sat")

    # Phase 145.1: route through pool.add() — Tier B gate-fall-through.
    pool = get_current_pool()
    cand = pool.add(partial_sat_name, "partial_sat", features)
    if cand is None:
        # Low confidence: pool.add returned None, fall through.
        return None

    # composer.py:1130 inline: return cand.name (NO _inject_stereo wrap)
    from ..name_tree import NameTreeNode  # Phase 165 SCORE-01 Path-B coarse node
    _nm = cand.name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="partial_sat", iupac_section_cite="P-25.3", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_partial_sat", "_is_partial_sat"]
