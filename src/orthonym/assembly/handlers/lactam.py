"""a phase lactam handler — direct-return shim with coverage gate.

Parallel to lactone handler; verbatim lift of composer.py:830-852
(inline branch) wrapping ``rules.lactams.is_monocyclic_lactam`` +
``rules.lactams.name_monocyclic_lactam``.

IUPAC cite: (lactams / cyclic amides).

References:
- composer.py:830-852 (inline dispatch branch; REMOVED at this commit).
- rules.lactams.{is_monocyclic_lactam, name_monocyclic_lactam}.
- internal notes-DECOMP.md row 'lactam' + purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_lactam(features: Any) -> bool:
    """Predicate: features.mol contains a monocyclic lactam (SMARTS check) and no senior group owns the suffix.

    A lactam is a heterocyclic pseudoketone "Cyclic anhydrides, esters and amides are named as
    pseudoketones; the resulting names are preferred IUPAC names"; "Method (1) generates preferred
    IUPAC names"), class 16 of Table 4.1, so a SEPARATE acid, ester, acid halide, amide, hydrazide, nitrile or aldehyde is the principal
    group and the ring C=O is the prefix 'oxo' "In cyclic pseudoketones, the prefix 'oxo' and/or
    acyl group prefixes are used to denote a carbonyl group"): '5-oxopyrrolidine-2-carboxylic acid', not
    '5-carboxypyrrolidin-2-one'. The same guard the lactone handler applies: "A lactone, as a
    pseudoketone, ranks lower in the seniority of classes than an acid or an ester"); dispatch then reaches
    the general suffix assembler.
    """
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    from ...rules.lactams import is_monocyclic_lactam
    info = is_monocyclic_lactam(mol)
    if info is None:
        return False
    from .lactone import has_senior_group_outside
    return not has_senior_group_outside(
        mol, set(info["ring_atoms"]) | {info["carbonyl_o_idx"]},
        (info["carbonyl_idx"], info["carbonyl_o_idx"], info["nitrogen_idx"]))


def name_lactam(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return lactam handler with coverage guard."""
    from ...rules.lactams import is_monocyclic_lactam, name_monocyclic_lactam
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

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
    # composer.py:852 inline: _inject_stereo_if_missing(features, pool.best.name, atom_to_locant=None)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="lactam", iupac_section_cite="P-66.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_lactam", "_is_lactam"]
