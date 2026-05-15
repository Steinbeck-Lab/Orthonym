"""Phase 160 guanidine handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:892-901 (inline branch) +
composer.py:2760-2906 (_try_name_guanidine body). Per CONTEXT D-24,
body stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: P-66.6 (guanidines; retained name with N-substitution).

References:
- composer.py:892-901 (inline dispatch branch; REMOVED at this commit).
- composer.py:2760-2906 (_try_name_guanidine body).
- 160-AUDIT-DECOMP.md § 1 row 'guanidine' + § 2.10 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NamingResult


def _is_guanidine(features: Any) -> bool:
    """Mirrors composer.py:892 (guanidine FG present AND principal_group is None)."""
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('guanidine')) and getattr(features, 'principal_group', None) is None


def name_guanidine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B guanidine handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _try_name_guanidine, _enrich_handler_name

    guanidine_name = _try_name_guanidine(features)
    if not guanidine_name:
        return None

    guanidine_name = _enrich_handler_name(features, guanidine_name, "guanidine")

    pool = get_current_pool()
    cand = pool.add(guanidine_name, "guanidine", features)
    if cand is None:
        return None

    return NamingResult(name=cand.name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_guanidine", "_is_guanidine"]
