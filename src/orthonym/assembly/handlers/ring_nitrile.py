"""Phase 160 ring_nitrile handler — Tier-2 mid-tier direct-return.

Verbatim lift of composer.py:1337-1348 (inline dispatch branch) + the
composer.py:4537-4605 (``_assemble_ring_nitrile_name``, 69 LOC) body via
lazy import. Per CONTEXT the body STAYS in composer.py during Plan-03
commits 03-01..03-09; Plan-03 commit 03-10 deletes the redundant body.

Byte-identical contract per CONTEXT (DECOMP-03): handler's behavior
on every canary fixture MUST equal the inline branch's behavior bit-for-bit.

IUPAC cite: P-66.5.1 (ring-nitriles / carbonitriles on rings).

References:
- composer.py:1337-1348 (inline dispatch branch; REMOVED at this commit).
- composer.py:4537-4605 (``_assemble_ring_nitrile_name`` body; STAYS until 03-10).
- 160-AUDIT-DECOMP.md § 1 row 'ring_nitrile' + § 3 Tier-2 row.
- 160-PATTERNS.md § 6 (Tier B / mid-tier lift handler pattern).
- 160-CONTEXT.md (atomic-commit byte-identical canary lock).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_ring_nitrile(features: Any) -> bool:
    """Mirrors composer.py:1337 inline guard.

    Predicate: ``principal_group == 'nitrile'`` AND ``is_cyclic`` AND
    ``not chain_is_parent``.

    Pure read-only per CONTEXT / AP-160-26.
    """
    if getattr(features, 'principal_group', None) != 'nitrile':
        return False
    if not getattr(features, 'is_cyclic', False):
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    return True


def name_ring_nitrile(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return ring_nitrile handler.

    Verbatim semantics of composer.py:1337-1348. Returns
    ``NamingResult(name=<final string>, tree=None, atom_to_locant_hint=None)``
    on success. ``_inject_stereo_if_missing`` is applied per the inline
    branch's behavior at composer.py:1348.
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _assemble_ring_nitrile_name, _inject_stereo_if_missing

    _rn_name = _assemble_ring_nitrile_name(features, style)
    if not _rn_name:
        # CR-03: matches sibling ring_ester handler pattern. pool.add(None, ...)
        # would violate the `name: str` contract per CONTEXT;
        # _inject_stereo_if_missing at the next step would then receive
        # None or a wrong handler's name from pool.best().
        return None
    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "ring_nitrile", _ha, (_rn_name or "")[:60],
        )

    # Phase 145.1: route through pool.add() — direct_return handler.
    pool = get_current_pool()
    pool.add(_rn_name, "ring_nitrile", features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="ring_nitrile", iupac_section_cite="P-66.5", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_ring_nitrile", "_is_ring_nitrile"]
