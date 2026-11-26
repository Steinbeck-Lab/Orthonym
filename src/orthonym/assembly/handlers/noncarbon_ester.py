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

    name = name_noncarbon_ester(features.mol, matches[0], tuple(matches[1:]))
    if not name:
        return None

    pool = get_current_pool()
    cand = pool.add(name, "noncarbon_ester", features)
    if cand is None:
        return None
    # the ester of a ring-nitrogen carboxylic acid is senior to a sulfonic or sulfinic ester class 7a, carboxylic
    # first): the name built on the sulfur acid ester is no PIN when the carbamate handler could not spell that one
    from .carbamate import demote_below_carbamate_parent
    from ...rules.polyol_mixed_anion import record_mixed_anion_polyol_non_pin
    demote_below_carbamate_parent(features, cand.name)
    record_mixed_anion_polyol_non_pin(features.mol, cand.name, ("S", "Se", "Te", "P", "As", "B", "Si"))   # '2-(nitrooxy)ethyl methanesulfonate': method (2)
    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(
            parent_stem=cand.name, class_id="noncarbon_ester",
            iupac_section_cite="P-65.6.3", fragment_legacy=cand.name,
        ),
        atom_to_locant_hint=None,
    )


def _is_ring_n_pseudoester(features: Any) -> bool:
    """Predicate: an ester of a carboxylic acid on a ring N "N(cyclic)", the Blue Book) and no group
    of a class senior to the esters: "salts > acids > anhydrides > esters"; another ester declines)."""
    if not (getattr(features, "functional_groups", None) or {}).get("ring_n_pseudoester"):
        return False
    from .nitrite_ester import principal_group_is_junior_to_esters
    mol = getattr(features, "mol", None)
    if mol is None or not principal_group_is_junior_to_esters(getattr(features, "principal_group", None)):
        return False
    from rdkit import Chem
    paired = {i for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]')        # nitro, azido, diazo: no ions
              for m in mol.GetSubstructMatches(Chem.MolFromSmarts(s)) for i in m}
    if any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in paired) for a in mol.GetAtoms()):
        return False
    # another ester (a carbamate or carbonate too: 'tert-butyl {2-[...]-2-oxoethyl}carbamate') or a salt, acid or
    # anhydride: which ester names the molecule is not decided here "salts > acids > anhydrides > esters")
    from ...validation.spelling.checks_ring_n_pseudoester import _SENIOR_OR_OTHER_ESTER
    return not any(mol.HasSubstructMatch(p) for p in _SENIOR_OR_OTHER_ESTER)


def name_ring_n_pseudoester_handler(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Offer '<ring-N-yl> <acid>ate' ('2,5-dioxopyrrolidin-1-yl acetate', as 'borinan-1-yl acetate (PIN)'); None lets
    the ring producers name the molecule as before."""
    matches = (getattr(features, "functional_groups", None) or {}).get("ring_n_pseudoester") or []
    if not matches:
        return None
    from ...rules.esters import name_noncarbon_ester
    from ..candidate_pool import get_current_pool
    name = name_noncarbon_ester(features.mol, matches[0], tuple(matches[1:]))
    if not name or ('cyclo[' in name and '^' in name):
        # a von Baeyer ring-N prefix where the ring producers keep the fusion name ('1-(acetyloxy)benzo[cd]indol-
        # 2(1H)-one', not '3-oxo-2-azatricyclo[6.3.1.0^4,12]...-2-yl acetate'): leave the molecule to them
        return None
    cand = get_current_pool().add(name, "ring_n_pseudoester", features)
    if cand is None:
        return None
    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(parent_stem=cand.name, class_id="noncarbon_ester",
                          iupac_section_cite="P-65.6.3.4.1", fragment_legacy=cand.name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_noncarbon_ester_handler", "_is_noncarbon_ester", "name_ring_n_pseudoester_handler",
           "_is_ring_n_pseudoester"]
