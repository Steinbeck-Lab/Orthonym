""" lactone handler — direct-return shim with coverage gate.

Verbatim lift of composer.py:826-848 (inline branch) wrapping
``rules.lactones.is_monocyclic_lactone`` + ``rules.lactones.name_monocyclic_lactone``.
Per CONTEXT, the rule bodies stay in rules.lactones unchanged.

The handler preserves the inline branch's coverage guard
(``ring_size > 8 or total_heavy <= ring_size + 8``) — without this guard,
substituted lactones in larger molecules would produce incomplete names.

IUPAC cite: P-66.6.3 (lactones / cyclic esters as heterocyclic ketones).

References:
- composer.py:826-848 (inline dispatch branch; REMOVED at this commit).
- rules.lactones.{is_monocyclic_lactone, name_monocyclic_lactone}.
- 160-AUDIT-DECOMP.md § 1 row 'lactone' + § 2.13 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_lactone(features: Any) -> bool:
    """Predicate: features.mol contains a monocyclic lactone (SMARTS check).

    Lazy import per PATTERNS § Lazy Import. The SMARTS substruct match is
    cheap; duplicate call between predicate + handler is acceptable for
     byte-identical preservation. performance benchmark
    can identify if memoization is needed.

    Pure read-only per CONTEXT /: reads features.mol via
    Chem.MolFromSmarts + GetSubstructMatches; no mutation.
    """
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    from ...rules.lactones import is_monocyclic_lactone
    return is_monocyclic_lactone(mol) is not None


def name_lactone(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return lactone handler with coverage guard.

    Verbatim semantics of composer.py:827-848. Returns None if the
    coverage guard rejects (large substituted lactone where the bare
    name would be incomplete).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ...rules.lactones import is_monocyclic_lactone, name_monocyclic_lactone

    lactone_info = is_monocyclic_lactone(features.mol)
    if not lactone_info:
        return None

    total_heavy = features.mol.GetNumHeavyAtoms()
    ring_size = lactone_info.get('ring_size', 0)

    # Coverage guard: bare lactone name only when molecule is not much
    # larger than the ring. Macrocycles (ring_size > 8) bypass this guard
    # — the ring IS the parent. Per composer.py:828-834.
    if not (ring_size > 8 or total_heavy <= ring_size + 8):
        return None

    lactone_name = name_monocyclic_lactone(features.mol)
    if not lactone_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "lactone", _ha, lactone_name[:60],
        )

    pool = get_current_pool()
    pool.add(lactone_name, "lactone", features)
    # composer.py:848 inline: _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="lactone", iupac_section_cite="P-65.7", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_lactone", "_is_lactone"]
