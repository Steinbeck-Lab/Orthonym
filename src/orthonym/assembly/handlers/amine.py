"""Phase 160 amine handler — Tier-2 mid-tier direct-return (fast-path).

Verbatim lift of composer.py:1374-1386 (inline dispatch branch). Body
``_assemble_amine_name`` at composer.py:4296-4536 (241 LOC) STAYS until
Plan-03 commit 03-10 (composer.py thinning) per CONTEXT D-24.

Predicate gates on:
1. ``principal_group in {secondary_amine, tertiary_amine}`` (primary_amine
   is handled by the general suffix path)
2. ``not is_polyfunctional`` (mutex with polyfunctional inline branch at
   composer.py:870)
3. ``not is_cyclic OR chain_is_parent`` (Tier-A mutex — same pattern as
   partial_sat handler in Plan-02 commit 02-24)

Byte-identical contract per CONTEXT D-21 (DECOMP-03): predicate gates capture
the inline cascade-order semantics so dispatch_inner reaches this handler
only when the inline amine branch at composer.py:1374 would have fired.

IUPAC cite: P-66.6.1 (amines).

References:
- composer.py:1374-1386 (inline dispatch branch; RETAINED as Tier-A-rejection
  fallback; Plan-03 commit 03-10 consolidates).
- composer.py:870-888 (polyfunctional inline branch — runs BEFORE amine).
- composer.py:4296-4536 (``_assemble_amine_name`` body; STAYS until 03-10).
- handlers/amide.py (sibling handler with same mutex pattern).
- 160-AUDIT-DECOMP.md § 1 row 'amine' + § 3 Tier-2 row.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NamingResult

logger = logging.getLogger(__name__)

_AMINE_PRINCIPAL_GROUPS = ('secondary_amine', 'tertiary_amine')


def _is_amine(features: Any) -> bool:
    """Predicate with polyfunctional + Tier-A mutex.

    Mirrors composer.py:1374 inline guard with mutex gates appended:
    - principal_group in {secondary_amine, tertiary_amine}
    - NOT is_polyfunctional
    - (NOT is_cyclic OR chain_is_parent) (Tier-A mutex)

    Pure read-only per CONTEXT D-25.
    """
    if getattr(features, 'principal_group', None) not in _AMINE_PRINCIPAL_GROUPS:
        return False
    if getattr(features, 'is_polyfunctional', False):
        return False
    is_cyclic = getattr(features, 'is_cyclic', False)
    chain_is_parent = getattr(features, 'chain_is_parent', False)
    if is_cyclic and not chain_is_parent:
        return False
    return True


def name_amine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return amine handler (fast-path).

    Verbatim semantics of composer.py:1374-1386. Returns
    ``NamingResult(name=best.name, tree=best.tree, atom_to_locant_hint=None)``
    after ``pool.add(name, 'amine', features, tree=...)`` (Phase 165 SCORE-01:
    counted coarse node, parity-safe via fragment_legacy). Returns None if
    ``_assemble_amine_name`` returns falsy (per inline guard at composer.py:1375).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _assemble_amine_name

    amine_name = _assemble_amine_name(features, style)
    if not amine_name:
        # Inline branch falls through when _assemble_amine_name returns falsy.
        # Return None so dispatch_inner caller falls through to inline cascade.
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "amine", _ha, amine_name[:60],
        )

    # Phase 165 SCORE-01: _assemble_amine_name builds the name via N-prefix
    # string concatenation (no fragment list) -> counted coarse node (D-03),
    # parity-safe via fragment_legacy. A structured upgrade requires a
    # fragments-based refactor of _assemble_amine_name (deferred; documented A1).
    from ..name_tree import NameTreeNode
    tree = NameTreeNode(
        parent_stem=amine_name, fragment_legacy=amine_name,
        class_id="amine", iupac_section_cite="P-62",
    )
    # Phase 145.1: route through pool.add() — direct_return handler.
    pool = get_current_pool()
    pool.add(amine_name, "amine", features, tree=tree)
    best = pool.best()
    return NamingResult(
        name=best.name, tree=best.tree, atom_to_locant_hint=None,
    )


__all__ = ["name_amine", "_is_amine"]
