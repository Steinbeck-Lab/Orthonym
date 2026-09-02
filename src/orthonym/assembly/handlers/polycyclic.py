""" polycyclic handler — direct-return shim (LIFT body alongside).

Verbatim move of composer.py:1100-1114 dispatch logic. Wraps the inline
``_assemble_polycyclic_name(features, style)`` call + ``pool.add()`` +
``pool.best().name`` return.

Per CONTEXT incremental migration: the underlying
``_assemble_polycyclic_name`` body STAYS in composer.py during and
moves to this module in (composer.py thinning).

IUPAC cite: P-25 (polycyclic aromatics + retained names).

Byte-identical contract: this handler fires only when ``polycyclic_name``
is set AND complex_ring would NOT take the dispatch — encoded as
``not _is_complex_ring_system(features.mol)`` in the predicate to preserve
the inline gate ``not _complex_ring_accepted`` semantics. The cases where
complex_ring fires-and-rejects (low ratio in V17 mode) still fall through
to the inline fallback path further down composer.py.

References:
- composer.py:1100-1114 (inline polycyclic branch; REMOVED at this commit).
- composer.py:_assemble_polycyclic_name (STAYS until 03-10).
- composer.py:_is_complex_ring_system (mutex helper for the predicate).
- 160-AUDIT-DECOMP.md § 1 row 'polycyclic' + § 2.26 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_polycyclic(features: Any) -> bool:
    """Predicate: polycyclic_name set AND not chain_is_parent AND not a complex ring system.

    The inline composer.py gate is nested under
    ``if not _complex_ring_accepted and not chain_is_parent`` and only fires
    when complex_ring did NOT short-circuit. We encode the conservative
    half of that gate here: when ``_is_complex_ring_system`` returns True,
    the inline complex_ring block at composer.py:1056+ handles the
    molecule (or falls through to the original inline polycyclic path,
    which is now NOT present because this handler replaces it). To preserve
    byte-identical behavior we require ``not _is_complex_ring_system`` —
    cases that would have fallen through after complex_ring rejected are
    captured by the legacy inline fallback at composer.py later in the
    cascade.
    """
    from ._handler_shared import cached_is_complex_ring_system

    if getattr(features, 'polycyclic_name', None) is None:
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    # WR-02: shared memoization with partial_sat / ring_ester predicates.
    if cached_is_complex_ring_system(features):
        return False
    return True


def name_polycyclic(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return polycyclic handler — wraps _assemble_polycyclic_name."""
    from ..candidate_pool import get_current_pool
    from ..composer import _assemble_polycyclic_name

    polycyclic_name = getattr(features, 'polycyclic_name', None)
    if polycyclic_name is None:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "polycyclic", _ha, (polycyclic_name or "")[:60],
        )

    poly_assembled = _assemble_polycyclic_name(features, style)
    pool = get_current_pool()
    pool.add(poly_assembled, "polycyclic", features)
    # composer.py:1114 inline: return pool.best().name (NO _inject_stereo wrap)
    _nm = pool.best().name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="polycyclic", iupac_section_cite="P-25", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_polycyclic", "_is_polycyclic"]
