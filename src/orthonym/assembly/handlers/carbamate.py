"""Phase 160 carbamate handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:865-874 (inline branch) +
composer.py:2561-2671 (_name_carbamate body). Per CONTEXT D-24, body
stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: P-66.5.5.1 (carbamates; functional class naming).

References:
- composer.py:865-874 (inline dispatch branch; REMOVED at this commit).
- composer.py:2561-2671 (_name_carbamate body).
- 160-AUDIT-DECOMP.md § 1 row 'carbamate' + § 2.8 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_carbamate(features: Any) -> bool:
    """Mirrors composer.py:865 (carbamate FG present AND principal_group is None)."""
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('carbamate')) and getattr(features, 'principal_group', None) is None


def name_carbamate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B carbamate handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _name_carbamate, _enrich_handler_name, _inject_stereo_if_missing,
    )

    carb_name = _name_carbamate(features)
    if not carb_name:
        return None

    carb_name = _enrich_handler_name(features, carb_name, "carbamate")

    pool = get_current_pool()
    cand = pool.add(carb_name, "carbamate", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="carbamate", iupac_section_cite="P-66.4", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_carbamate", "_is_carbamate"]
