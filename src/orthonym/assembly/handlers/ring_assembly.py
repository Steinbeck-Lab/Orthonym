""" ring_assembly handler — direct-return shim.

1-line wrapper around ``rules.ring_assemblies.name_ring_assembly``. Verbatim
move of composer.py:979-993 dispatch logic.

IUPAC cite: P-28 (ring assemblies / biaryls / multi-ring linked systems).

References:
- composer.py:979-993 (inline ring_assembly branch; REMOVED at this commit).
- rules.ring_assemblies.name_ring_assembly — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'ring_assembly' + § 2.24 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_ring_assembly(features: Any) -> bool:
    """Mirrors composer.py:979-980 inline gate.

    The inline branch checked ``assembly_info and not chain_is_parent``;
    we encode both clauses here so the predicate fully matches the
    inline guard byte-for-byte.
    """
    assembly_info = getattr(features, 'ring_assembly_info', None)
    if not assembly_info:
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    return True


def name_ring_assembly(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return ring_assembly handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ...rules.ring_assemblies import name_ring_assembly as _name_ring_assembly

    assembly_info = getattr(features, 'ring_assembly_info', None)
    if not assembly_info:
        return None

    assembly_name = _name_ring_assembly(features.mol, assembly_info, features)
    if not assembly_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "ring_assembly", _ha, assembly_name[:60],
        )

    pool = get_current_pool()
    pool.add(assembly_name, "ring_assembly", features)
    # composer.py:993 inline: _inject_stereo_if_missing(features, pool.best().name)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="ring_assembly", iupac_section_cite="P-28", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_ring_assembly", "_is_ring_assembly"]
