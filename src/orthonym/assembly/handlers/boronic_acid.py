"""Phase 160 boronic_acid handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:1205-1214 (inline branch) +
composer.py:2450-2497 (_name_boronic_acid body). Per CONTEXT D-24,
body stays in composer.py until Plan-03 commit 03-10.

Note: this handler's inline branch is OUT-OF-LINE in source code
(composer.py:1205 is further down than carbamic_acid/urea/etc. at L855-892).
The plan-02 priority of 1000 maintains the dispatch ordering at the
inner-dispatch layer (vs the original 1205 source-line position). Per
audit § 1: cross-predicate mutex (principal_group is a single string)
means dispatch-order vs source-order cannot create a byte-diff for
this handler.

IUPAC cite: P-66.6.4 (boronic acids; retained name).

References:
- composer.py:1205-1214 (inline dispatch branch; REMOVED at this commit).
- composer.py:2450-2497 (_name_boronic_acid body).
- 160-AUDIT-DECOMP.md § 1 row 'boronic_acid' + § 2.23 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_boronic_acid(features: Any) -> bool:
    """Mirrors composer.py:1205 (``features.principal_group == 'boronic_acid'``)."""
    return getattr(features, 'principal_group', None) == 'boronic_acid'


def name_boronic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B boronic acid handler.

    Note: inline branch at composer.py:1213 passes no explicit atom_to_locant
    to _inject_stereo_if_missing (default None). We mirror that here.
    """
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _name_boronic_acid, _enrich_handler_name, _inject_stereo_if_missing,
    )

    boronic_name = _name_boronic_acid(features)
    if not boronic_name:
        return None

    boronic_name = _enrich_handler_name(features, boronic_name, "boronic_acid")

    pool = get_current_pool()
    cand = pool.add(boronic_name, "boronic_acid", features)
    if cand is None:
        return None

    # composer.py:1213 inline branch: _inject_stereo_if_missing(features, cand.name)
    # — no explicit atom_to_locant kwarg (defaults to None).
    final_name = _inject_stereo_if_missing(features, cand.name)
    return NamingResult(name=final_name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_boronic_acid", "_is_boronic_acid"]
