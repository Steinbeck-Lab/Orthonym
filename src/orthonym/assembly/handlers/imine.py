"""Wave2 T2a imine handler — N-substituted acyclic imines (P-62.3.1.1).

BB VERBATIM: 'N-methylethanimine (PIN) [not N-ethylidenemethanamine; nor
N-ethylidene(methyl)amine]'. Bare (=NH) imines already name correctly via
the generic suffix path (ethanimine, propan-2-imine); this handler covers
the N-SUBSTITUTED form the broadened imine SMARTS now perceives, citing the
lone N-substituent as an italic N- prefix via composer._assemble_imine_name
(the single-N shape of _assemble_amine_name with the imine suffix).

Predicate is deliberately narrow (FAIL-CLOSED): a single acyclic imine
whose nitrogen actually carries a substituent; polyfunctional / cyclic-
parent shapes keep their own paths.

IUPAC cite: P-62.3.1.1.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_n_substituted_imine(features: Any) -> bool:
    """principal_group == 'imine' AND the imine N carries a substituent.

    Mirrors the amine handler's mutex gates (NOT polyfunctional; NOT cyclic
    unless chain is parent). Pure read-only.
    """
    if getattr(features, 'principal_group', None) != 'imine':
        return False
    if getattr(features, 'is_polyfunctional', False):
        return False
    is_cyclic = getattr(features, 'is_cyclic', False)
    if is_cyclic and not getattr(features, 'chain_is_parent', False):
        return False
    mol = getattr(features, 'mol', None)
    pg = getattr(features, 'principal_group_atoms', None) or []
    if mol is None or len(pg) != 1:
        return False
    for idx in pg[0]:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            # Degree 2 = the double-bonded parent C + one substituent.
            return atom.GetDegree() >= 2 and not atom.IsInRing()
    return False


def name_imine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return N-substituted-imine handler (amine-handler pattern)."""
    from ..candidate_pool import get_current_pool
    from ..composer import _assemble_imine_name

    imine_name = _assemble_imine_name(features, style)
    if not imine_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "HANDLER_COVERAGE: handler=imine coverage=NA accounted=NA/%d name=%s",
            features.mol.GetNumHeavyAtoms(), imine_name[:60],
        )

    tree = NameTreeNode(
        parent_stem=imine_name, fragment_legacy=imine_name,
        class_id="imine", iupac_section_cite="P-62.3.1.1",
    )
    pool = get_current_pool()
    pool.add(imine_name, "imine", features, tree=tree)
    best = pool.best()
    return NamingResult(
        name=best.name, tree=best.tree, atom_to_locant_hint=None,
    )


__all__ = ["name_imine", "_is_n_substituted_imine"]
