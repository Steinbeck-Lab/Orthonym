"""Phase 160 sulfone handler — Tier B shim (gate 0.40).

1-line wrapper around ``rules.sulfur.name_sulfone``. Verbatim move
of composer.py:966-978 dispatch logic (sulfone branch parallel to
sulfoxide).

IUPAC cite: P-66.5.2.4 (sulfones; functional class naming).

References:
- composer.py:966-978 (inline sulfone branch; REMOVED at this commit).
- rules.sulfur.name_sulfone — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'sulfone' + § 2.20 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_sulfone(features: Any) -> bool:
    """Mirrors composer.py:966 (``features.principal_group == 'sulfone'``)."""
    return getattr(features, 'principal_group', None) == 'sulfone'


def name_sulfone(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B sulfone handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _enrich_handler_name, _inject_stereo_if_missing
    from ...rules.sulfur import name_sulfone as _name_sulfone

    matches = features.functional_groups.get('sulfone', [])
    if not matches:
        return None

    name = _name_sulfone(features.mol, matches[0])
    if not name:
        return None

    name = _enrich_handler_name(features, name, "sulfone")

    pool = get_current_pool()
    cand = pool.add(name, "sulfone", features)
    if cand is None:
        return None

    # composer.py:977 inline: _inject_stereo_if_missing(features, cand.name)
    final_name = _inject_stereo_if_missing(features, cand.name)
    return NamingResult(name=final_name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_sulfone", "_is_sulfone"]
