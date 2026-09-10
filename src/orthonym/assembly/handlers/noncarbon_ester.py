"""W3-P07 non-carbon ester handler / /.

Dispatches the three ester classes whose acid or alcohol component is not the
ordinary carbon-on-oxygen carboxylic ester, and which therefore never match the
generic ``[CX3](=O)[OX2][#6]`` 'ester' perception:

  * pseudoester R-CO-O-Z (Z a Group-13/14/15 organyl) -> 'trimethylsilyl acetate'
  * sulfonic ester R-SO2-O-R' -> 'methyl methanesulfonate'
  * sulfinic ester R-S(=O)-O-R' -> 'methyl methanesulfinate'

The body is the single shared ``rules.esters.name_noncarbon_ester`` mechanism
(build the neutral free acid -> '-ate' + O-side organyl). Fail-closed: returns
None when the component acid/organyl cannot be named, so the surrounding
dispatch_inner loop continues to the next handler.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

_NONCARBON_ESTER_PGS = ("pseudoester", "sulfonic_ester", "sulfinic_ester")


def _is_noncarbon_ester(features: Any) -> bool:
    """Predicate: principal_group is one of the non-carbon ester classes AND a
    concrete match tuple is available (pure read-only per)."""
    if getattr(features, "principal_group", None) not in _NONCARBON_ESTER_PGS:
        return False
    return bool(getattr(features, "principal_group_atoms", None))


def name_noncarbon_ester_handler(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Name a pseudoester / sulfonic ester / sulfinic ester (W3-P07)."""
    matches = getattr(features, "principal_group_atoms", None)
    if not matches:
        return None

    from ...rules.esters import name_noncarbon_ester
    from ..candidate_pool import get_current_pool

    name = name_noncarbon_ester(features.mol, matches[0])
    if not name:
        return None

    pool = get_current_pool()
    cand = pool.add(name, "noncarbon_ester", features)
    if cand is None:
        return None
    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(
            parent_stem=cand.name, class_id="noncarbon_ester",
            iupac_section_cite="P-65.6.3", fragment_legacy=cand.name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_noncarbon_ester_handler", "_is_noncarbon_ester"]
