""" 11-FABLEFIX pnictogen -inate ester handler — direct-return shim.

Functional-class ester of a phosphinic / arsinic / stibinic acid, R2E(=O)(OR')
(``methyl diphenylphosphinate``, ``methyl diphenylarsinate``). This is the exact
bridge shape the multiplicative pnictogen guard declines (two identical/aryl C-E
bonds, one =O, one -O-R ester), so declining that multiplicative name
(``1,1'-(methoxyphosphoryl)dibenzene``) hands the molecule here.

IUPAC cite: class 9 (esters) / (functional-class ester).

References:
- perception.functional_groups — the three ``*inate_ester`` SMARTS.
- rules.phosphorus._name_pnictogen_inate_ester — chemical-logic body (element-
  generic over P/As/Sb).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)

# principal_group key -> central-atom symbol
_INATE_ESTER_GROUPS = {
    "phosphinate_ester": "P",
    "arsinate_ester": "As",
    "stibinate_ester": "Sb",
}


def _is_pnictogen_inate_ester(features: Any) -> bool:
    """principal_group is one of the three -inate ester classes."""
    return getattr(features, "principal_group", None) in _INATE_ESTER_GROUPS


def name_pnictogen_inate_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return -inate ester handler (P/As/Sb)."""
    from ...rules.phosphorus import _name_pnictogen_inate_ester
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    fg_key = features.principal_group
    symbol = _INATE_ESTER_GROUPS.get(fg_key)
    if symbol is None:
        return None
    matches = features.functional_groups.get(fg_key, [])
    if not matches:
        return None

    name = _name_pnictogen_inate_ester(features.mol, matches[0], symbol)
    if not name:
        return None                             # complex organyl -> defer (cascade)

    pool = get_current_pool()
    pool.add(name, fg_key, features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name,
            class_id=fg_key,
            iupac_section_cite="P-65.6.3.2",
            fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_pnictogen_inate_ester", "_is_pnictogen_inate_ester"]
