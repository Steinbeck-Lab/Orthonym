"""Phase 160 urea handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:876-885 (inline branch) +
composer.py:2672-2759 (_try_name_urea body). Per CONTEXT D-24, body
stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: P-66.6 (ureas; retained name with N-substitution).

References:
- composer.py:876-885 (inline dispatch branch; REMOVED at this commit).
- composer.py:2672-2759 (_try_name_urea body).
- 160-AUDIT-DECOMP.md § 1 row 'urea' + § 2.9 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_urea(features: Any) -> bool:
    """Mirrors composer.py:876 (urea FG present AND principal_group is None)."""
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('urea')) and getattr(features, 'principal_group', None) is None


def name_urea(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B urea handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _try_name_urea, _enrich_handler_name, _inject_stereo_if_missing,
    )

    urea_name = _try_name_urea(features)
    if not urea_name:
        return None

    urea_name = _enrich_handler_name(features, urea_name, "urea")

    pool = get_current_pool()
    cand = pool.add(urea_name, "urea", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    from ..name_tree import NameTreeNode  # Phase 165 SCORE-01 Path-B coarse node
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="urea", iupac_section_cite="P-66.4.1", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_urea", "_is_urea"]
