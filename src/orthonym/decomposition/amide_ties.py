"""The amide split a molecule is named from when two of its amides tie as parent structures.

``engine._split_parent_amide_is_junior`` declines an amide split whose parent is strictly junior to another amide
(``rules.amide_parent``:,,,, and, and says nothing of a tie.
A tie is decided by the names of the prefixes (``rules.amide_tie_names``): "the lower locant set for substituent
groups in order of citation in the name" (the Blue Book, " CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT
GROUPS",:21596) and "The preferred IUPAC name is the name that is earlier in alphanumerical order" (:22234,
" CRITERIA RELATED TO ALPHANUMERICAL ORDER OF NAMES"), so that the first split that was cleaved
('N-ethyl-2-[2-(methylamino)acetamido]acetamide') is a PIN only when its parent is the amide those criteria choose.

``resolved_amide_split_name`` is asked of the name of one amide split, after the split has been named:

* the parent of the split is the best of the amides it names the molecule on, and no other amide ties with it (or only a
  symmetric copy of it): the name stands;
* the criteria choose the parent of the split: the name stands;
* the criteria choose another amide, and the molecule has a split on that amide that names it (adequately): that name;
* the criteria choose another amide that no split names, or do not decide: the name stands, labelled below the PIN
  (``record_non_pin_label``): it is a name of the molecule, not shown to be the preferred one.

Only a molecule whose principal characteristic group is a carboxamide is judged (a name on an acid or an ester parent does
not choose among amides).
"""
from typing import Callable, Dict, List, Optional, Set

from rdkit import Chem

#: the rule and the reason of the label of a name whose parent is not shown to be the preferred one
TIE_RULE = "P-45.2.3"
TIE_DETAIL = "the parent amide is not shown to be the one that the order of citation and the alphanumerical order choose"


def split_parent_atoms(mol, bond: Dict) -> Optional[Set[int]]:
    """The atoms of the part of ``mol`` that the name of the amide split ``bond`` takes its parent amide from: the carbonyl
    carbon when the amine is cited as an N-substituent (a simple amine prefix), the whole amine side when the acyl floats
    onto the amine (``engine._split_parent_amide_is_junior``). None when the split cannot be read."""
    from .engine import _name_fragment_with_fallback
    from .fragment_assembly import _amine_to_prefix
    from .fragment_capping import cleave_and_cap
    fragments = cleave_and_cap(mol, [bond], acid_side_oh=True)
    amine = next((f for f in fragments or () if f.get("side") == "amine"), None)
    if amine is None:
        return None
    if _amine_to_prefix(_name_fragment_with_fallback(amine["smiles"]) or ""):
        return {bond["acid_atom"]}
    side, stack = {bond["amine_atom"]}, [bond["amine_atom"]]
    while stack:
        atom = stack.pop()
        for b in mol.GetAtomWithIdx(atom).GetBonds():
            other = b.GetOtherAtomIdx(atom)
            if b.GetIdx() != bond["bond_idx"] and other not in side:
                side.add(other)
                stack.append(other)
    return side


def _principal_group_is_amide(mol) -> bool:
    from ..rules.amide_parent import CARBOXAMIDE_CLASSES
    from .engine import _molecule_principal_group
    return _molecule_principal_group(mol) in CARBOXAMIDE_CLASSES


def _label(name: str) -> str:
    from ..metrics.provenance import record_non_pin_label
    record_non_pin_label(name, rule=TIE_RULE, detail=TIE_DETAIL)
    return name


def _split_of(mol, bonds: List[Dict], target_ranks: Set[int], own_bond: Dict, style: str, name_split: Callable,
              adequate: Callable) -> Optional[str]:
    """The name of the first amide split (other than ``own_bond``) whose parent amide is one of the atoms of
    ``target_ranks`` (the classes of the amide the criteria choose) that gives an adequate name."""
    from ..rules.amide_parent import _label_map, amide_parent_key, comparable_amide_units
    flat = comparable_amide_units(mol)
    labels = _label_map(mol)
    carbonyls = {m[0] for _, m in flat}
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    for bond in sorted(bonds, key=lambda b: b["bond_idx"]):
        if bond["bond_idx"] == own_bond["bond_idx"] or bond.get("type") != "amide":
            continue
        parent = split_parent_atoms(mol, bond)
        inside = {c: amide_parent_key(mol, c, labels, carbonyls) for c in carbonyls if parent and c in parent}
        if not inside:
            continue
        best = min(inside.values())
        if not any(ranks[c] in target_ranks for c, k in inside.items() if k == best):
            continue
        found = name_split(mol, bond, style)
        if found and adequate(found, mol, bond):
            return found
    return None


def resolved_amide_split_name(mol, bond: Dict, name: str, style: str, name_split: Callable, find_bonds: Callable,
                              adequate: Callable) -> Optional[str]:
    """``name`` (the name of the amide split ``bond``), the name of the split that the criteria of the module docstring choose
    instead, or ``name`` labelled below the PIN. ``name_split(mol, bond, style)`` names a split, ``find_bonds(mol)`` lists the
    cleavable bonds and ``adequate(name, mol, bond)`` is the engine's coverage gate."""
    if not name or bond.get("type") != "amide":
        return name
    try:
        from ..rules.amide_parent import comparable_amide_units, tied_amide_parents
        from ..rules.amide_tie_names import senior_amide_by_names
        if comparable_amide_units(mol) is None or not _principal_group_is_amide(mol):
            return name
        parent = split_parent_atoms(mol, bond)
        from ..rules.amide_parent import STEREO_DETAIL, STEREO_RULE, stereo_criterion_could_differ
        if parent and stereo_criterion_could_differ(mol, parent):
            from ..metrics.provenance import record_non_pin_label
            record_non_pin_label(name, rule=STEREO_RULE, detail=STEREO_DETAIL)     # (l) read literally: not shown to be the PIN
        group = tied_amide_parents(mol, parent) if parent else None
        if not group:
            return name
        winner = senior_amide_by_names(mol, [c for c, _ in group])
        if winner is not None and any(c == winner and own for c, own in group):
            return name
    except Exception:  # noqa: BLE001 -- not decidable: the split stands as it was
        return name
    if winner is not None:
        try:
            ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
            other = _split_of(mol, find_bonds(mol), {ranks[winner]}, bond, style, name_split, adequate)
        except Exception:  # noqa: BLE001 -- no split of the chosen amide: the name stands, labelled
            other = None
        if other:
            return other
    return _label(name)
