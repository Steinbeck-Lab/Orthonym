"""Phase 160 sulfoxide handler — Tier B shim (gate 0.40).

1-line wrapper around ``rules.sulfur.name_sulfoxide``. Verbatim move
of composer.py:966-979 dispatch logic (sulfoxide branch of the shared
if-else with sulfone).

IUPAC cite: P-66.5.2.4 (sulfoxides; functional class naming).

References:
- composer.py:966-979 (inline sulfoxide branch; REMOVED at this commit).
- rules.sulfur.name_sulfoxide — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'sulfoxide' + § 2.19 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_sulfoxide(features: Any) -> bool:
    """Mirrors composer.py:968 (``features.principal_group == 'sulfoxide'``)."""
    return getattr(features, 'principal_group', None) == 'sulfoxide'


def name_sulfoxide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B sulfoxide handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _enrich_handler_name, _inject_stereo_if_missing
    from ...rules.sulfur import name_sulfoxide as _name_sulfoxide

    matches = features.functional_groups.get('sulfoxide', [])
    if not matches:
        return None

    name = _name_sulfoxide(features.mol, matches[0])
    if not name:
        return None

    name = _enrich_handler_name(features, name, "sulfoxide")

    pool = get_current_pool()
    cand = pool.add(name, "sulfoxide", features)
    if cand is None:
        return None

    # composer.py:978 inline: _inject_stereo_if_missing(features, cand.name)
    # — no explicit atom_to_locant kwarg (defaults to None).
    final_name = _inject_stereo_if_missing(features, cand.name)
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="sulfoxide", iupac_section_cite="P-63.6", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_sulfoxide", "_is_sulfoxide"]
