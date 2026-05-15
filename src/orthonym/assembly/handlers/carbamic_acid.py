"""Phase 160 carbamic_acid handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:855-863 (inline branch) +
composer.py:2498-2560 (_name_carbamic_acid body). Per CONTEXT D-24,
body stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: P-66.5.5 (carbamic acids; retained name with N-substitution).

References:
- composer.py:855-863 (inline dispatch branch; REMOVED at this commit).
- composer.py:2498-2560 (_name_carbamic_acid body).
- 160-AUDIT-DECOMP.md § 1 row 'carbamic_acid' + § 2.7 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_carbamic_acid(features: Any) -> bool:
    """Mirrors composer.py:855 (``features.principal_group == 'carbamic_acid'``)."""
    return getattr(features, 'principal_group', None) == 'carbamic_acid'


def name_carbamic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B carbamic acid handler.

    Verbatim semantics of composer.py:855-863.
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _name_carbamic_acid, _enrich_handler_name

    carbamic_name = _name_carbamic_acid(features)
    if not carbamic_name:
        return None

    carbamic_name = _enrich_handler_name(features, carbamic_name, "carbamic_acid")

    pool = get_current_pool()
    cand = pool.add(carbamic_name, "carbamic_acid", features)
    if cand is None:
        return None

    return NamingResult(name=cand.name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_carbamic_acid", "_is_carbamic_acid"]
