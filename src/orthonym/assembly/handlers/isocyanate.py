"""Phase 160 isocyanate handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:828-836 (inline branch) +
composer.py:2180-2190 (_name_isocyanate body) + composer.py:2204-2230
(shared _name_iso_x_cyanate helper). Per CONTEXT D-24 incremental-
migration discipline, bodies stay in composer.py until Plan-03 commit
03-10 thinning.

IUPAC cite: P-66.5.4.3 (isocyanates; functional class naming).

References:
- composer.py:828-836 (inline dispatch branch; REMOVED at this commit).
- composer.py:2180-2190 (_name_isocyanate body).
- composer.py:2204-2230 (shared _name_iso_x_cyanate helper).
- 160-AUDIT-DECOMP.md § 1 row 'isocyanate' + § 2.5 predicate purity proof.
- 160-PATTERNS.md § 6 (Tier-B lift handler pattern).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_isocyanate(features: Any) -> bool:
    """Mirrors composer.py:828 (isocyanate FG present AND principal_group is None).

    Pure read-only per CONTEXT D-25 / AP-160-26.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('isocyanate')) and getattr(features, 'principal_group', None) is None


def name_isocyanate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B isocyanate handler.

    Verbatim semantics of composer.py:828-836 (inline branch).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _name_isocyanate, _enrich_handler_name

    iso_name = _name_isocyanate(features)
    if not iso_name:
        return None

    iso_name = _enrich_handler_name(features, iso_name, "isocyanate")

    pool = get_current_pool()
    cand = pool.add(iso_name, "isocyanate", features)
    if cand is None:
        return None

    return NamingResult(name=cand.name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_isocyanate", "_is_isocyanate"]
