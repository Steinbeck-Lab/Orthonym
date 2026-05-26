"""Phase 160 amide handler — Tier-2 mid-tier direct-return (fast-path).

Verbatim lift of composer.py:1343-1356 (inline dispatch branch). Body
``_assemble_amide_name`` at composer.py:4154-4295 (142 LOC) STAYS until
Plan-03 commit 03-10 (composer.py thinning) per CONTEXT D-24.

Predicate gates on:
1. ``principal_group in {primary_amide, secondary_amide, tertiary_amide}``
2. ``len(principal_group_atoms) == 1`` (single amide; multi-amide compounds
   fall through to general_acyclic suffix path per composer.py:1339-1341)
3. ``not is_polyfunctional`` (mutex with the polyfunctional inline branch at
   composer.py:870 — fires BEFORE amide in the inline cascade order and
   produces a different name like `(2S,3R)-2-(octadecanoylamino)-...`)
4. ``not is_cyclic OR chain_is_parent`` (Tier-A mutex — same pattern as
   partial_sat handler in Plan-02 commit 02-24). For cyclic-amide cases where
   Tier-A would win, dispatch_inner skips this entry and falls through to the
   inline cascade where Tier-A ring competition runs first, then the inline
   amide block at composer.py:1343 picks up the post-Tier-A-rejection fallback.

Byte-identical contract per CONTEXT D-21 (DECOMP-03): predicate gates capture
the inline cascade-order semantics so dispatch_inner reaches this handler
only when the inline amide branch at composer.py:1343 would have fired.

IUPAC cite: P-66.5.3 (amides).

References:
- composer.py:1343-1356 (inline dispatch branch; RETAINED as Tier-A-rejection
  fallback; Plan-03 commit 03-10 consolidates).
- composer.py:870-888 (polyfunctional inline branch — runs BEFORE amide in
  inline cascade order; predicate `not is_polyfunctional` mirrors this).
- composer.py:4154-4295 (``_assemble_amide_name`` body; STAYS until 03-10).
- handlers/partial_sat.py (same Tier-A mutex pattern).
- 160-AUDIT-DECOMP.md § 1 row 'amide' + § 3 Tier-2 row.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)

_AMIDE_PRINCIPAL_GROUPS = (
    'primary_amide', 'secondary_amide', 'tertiary_amide',
    # Phase 163 chalcogen amides (single-permissive [NX3] per AUDIT § 2.2;
    # no 3-way primary/secondary/tertiary split). Same _assemble_amide_name
    # pipeline; per-PG suffix lookup happens inside via _CHALCOGEN_AMIDE_SUFFIX_FORMS.
    'thioamide', 'selenoamide', 'telluroamide',
)


def _is_amide(features: Any) -> bool:
    """Predicate with full inline-cascade-order mutex.

    Gates on: principal_group is amide AND pg_count == 1 AND not is_polyfunctional
    AND (not is_cyclic OR chain_is_parent).

    Pure read-only per CONTEXT D-25.
    """
    if getattr(features, 'principal_group', None) not in _AMIDE_PRINCIPAL_GROUPS:
        return False
    pg_atoms = getattr(features, 'principal_group_atoms', None)
    pg_count = len(pg_atoms) if pg_atoms else 1
    if pg_count != 1:
        return False
    # Polyfunctional mutex: polyfunctional inline branch (composer.py:870-888)
    # fires BEFORE amide in the inline cascade order. Without this mutex,
    # dispatch_inner preempts the polyfunctional path for amide-bearing
    # polyfunctional molecules.
    if getattr(features, 'is_polyfunctional', False):
        return False
    # Tier-A mutex: only fast-path when ring competition wouldn't fire.
    is_cyclic = getattr(features, 'is_cyclic', False)
    chain_is_parent = getattr(features, 'chain_is_parent', False)
    if is_cyclic and not chain_is_parent:
        return False
    return True


def name_amide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return amide handler (fast-path).

    Verbatim semantics of composer.py:1343-1356. Returns
    ``NamingResult(name=best.name, tree=best.tree, atom_to_locant_hint=None)``
    after ``pool.add(name, 'amide', features, tree=...)`` (Phase 165 SCORE-01:
    structured tree from the chain-fragment path, else a coarse node).
    No ``_inject_stereo_if_missing``
    wrap per the inline branch behavior at composer.py:1356.
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _assemble_amide_name

    _amide_name = _assemble_amide_name(features, style)
    if not _amide_name:
        # CR-02: matches sibling name_amine pattern. pool.add(None, ...)
        # would violate the `name: str` contract per CONTEXT D-05 and
        # silently produce wrong output (the inline equivalent at
        # composer.py:1334 has implicit truthiness check via the if-block).
        return None
    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "amide", _ha, (_amide_name or "")[:60],
        )

    # Phase 165 SCORE-01: prefer the structured tree stashed by
    # _assemble_amide_name (unsaturated chain-fragment path); else a counted
    # coarse node (D-03) for the name_amide() string paths (saturated/ring
    # amides) that build no fragment list. fragment_legacy=name guarantees a
    # byte-identical round-trip. Read best().tree so the tree matches the
    # RETURNED candidate (winner-guard).
    tree = getattr(features, "_amide_tree", None)
    if tree is None:
        tree = NameTreeNode(
            parent_stem=_amide_name, fragment_legacy=_amide_name,
            class_id="amide", iupac_section_cite="P-66.1",
        )
    # Phase 145.1: route through pool.add() — direct_return handler.
    pool = get_current_pool()
    pool.add(_amide_name, "amide", features, tree=tree)
    best = pool.best()
    return NamingResult(
        name=best.name, tree=best.tree, atom_to_locant_hint=None,
    )


__all__ = ["name_amide", "_is_amide"]
