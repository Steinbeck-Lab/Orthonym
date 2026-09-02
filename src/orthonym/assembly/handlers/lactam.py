""" lactam handler — direct-return shim with coverage gate.

Parallel to lactone handler; verbatim lift of composer.py:830-852
(inline branch) wrapping ``rules.lactams.is_monocyclic_lactam`` +
``rules.lactams.name_monocyclic_lactam``.

IUPAC cite: P-66.6.3 (lactams / cyclic amides).

References:
- composer.py:830-852 (inline dispatch branch; REMOVED at this commit).
- rules.lactams.{is_monocyclic_lactam, name_monocyclic_lactam}.
- 160-AUDIT-DECOMP.md § 1 row 'lactam' + § 2.14 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_lactam(features: Any) -> bool:
    """Predicate: features.mol contains a monocyclic lactam (SMARTS check)."""
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    from ...rules.lactams import is_monocyclic_lactam
    return is_monocyclic_lactam(mol) is not None


def name_lactam(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return lactam handler with coverage guard."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ...rules.lactams import is_monocyclic_lactam, name_monocyclic_lactam

    lactam_info = is_monocyclic_lactam(features.mol)
    if not lactam_info:
        return None

    total_heavy = features.mol.GetNumHeavyAtoms()
    ring_size = lactam_info.get('ring_size', 0)

    # Coverage guard (same as lactone): macrocycles bypass.
    if not (ring_size > 8 or total_heavy <= ring_size + 8):
        return None

    lactam_name = name_monocyclic_lactam(features.mol)
    if not lactam_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "lactam", _ha, lactam_name[:60],
        )

    pool = get_current_pool()
    pool.add(lactam_name, "lactam", features)
    # composer.py:852 inline: _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="lactam", iupac_section_cite="P-66.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_lactam", "_is_lactam"]
