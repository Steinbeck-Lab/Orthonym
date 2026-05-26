"""Phase 160 hydrazone handler — Tier B retained-name (gate 0.40).

Verbatim lift of the hydrazone dispatch logic from composer.py:785-794
(inline branch) + composer.py:1906-2042 (shared body of
_name_oxime_or_hydrazone with second arg ``'hydrazone'``). Per CONTEXT
D-24 incremental-migration discipline, the body stays in composer.py
until Plan-03 commit 03-10 thinning.

IUPAC cite: P-66.6 (hydrazone functional class naming).

References:
- composer.py:785-794 (inline dispatch branch; REMOVED at this commit).
- composer.py:1906-2042 (shared with oxime; second arg dispatches naming).
- 160-AUDIT-DECOMP.md § 1 row 'hydrazone' + § 2.3 predicate purity proof.
- 160-PATTERNS.md § 6 (Tier-B lift handler pattern).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_hydrazone(features: Any) -> bool:
    """Mirrors composer.py:785 (``features.principal_group == 'hydrazone'``).

    Pure read-only per CONTEXT D-25 / AP-160-26.
    """
    return getattr(features, 'principal_group', None) == 'hydrazone'


def name_hydrazone(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B hydrazone handler.

    Verbatim semantics of composer.py:785-794 (inline branch). Returns
    ``NamingResult(name, tree=None, atom_to_locant_hint=None)`` on success;
    None on gate-fail (Pool's HANDLER_POLICIES['hydrazone'] gate 0.40).
    """
    # Lazy imports per PATTERNS § Lazy Import.
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _name_oxime_or_hydrazone, _enrich_handler_name,
        _inject_stereo_if_missing,
    )

    hydrazone_name = _name_oxime_or_hydrazone(features, 'hydrazone')
    if not hydrazone_name:
        return None

    hydrazone_name = _enrich_handler_name(features, hydrazone_name, "hydrazone")

    pool = get_current_pool()
    cand = pool.add(hydrazone_name, "hydrazone", features)
    if cand is None:
        return None

    # Per composer.py:793 inline branch: wrap in _inject_stereo_if_missing
    # with atom_to_locant=None for byte-identical preservation per CONTEXT D-13.
    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="hydrazone", iupac_section_cite="P-68.3.1.2", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_hydrazone", "_is_hydrazone"]
