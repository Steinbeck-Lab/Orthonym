"""Anhydrides of boric acid with carboxylic acids, B(OH)3-n(O-CO-R)n.

An acidic one keeps an -OH on the boron and is named substitutively on the boron acid, (the acid is senior to
the anhydride): '(propanoyloxy)boronic acid (PIN)' (the Blue Book, not 'boric propanoic monoanhydride'),
'bis(acetyloxy)borinic acid (PIN; acid is senior to anhydride)' (:32382), as the phosphorus analogue
'(acetyloxy)phosphonic acid (PIN)' (:36999, ``phosphorus.name_acyloxy_phosphonic_acid``).

The neutral one, B(O-CO-R)3 of one unsubstituted acid, is named by "Anhydrides of mononuclear noncarbon
oxoacids" (:35982): "The names of the acids are cited in alphabetical order": 'acetic boric trianhydride (PIN)'
(:35994). The book also prints 'triacetic boric trianhydride (PIN)' at:32066; OPSIN 2.9.0 reads the form
only, and that section is the rule of this class.
"""
from typing import Optional

from rdkit import Chem


#: an acid group other than the boron's: an OH (or O-) on an atom that also bears =O or =S
_OTHER_ACID = Chem.MolFromSmarts("[!#5;$(*=[O,S])][OX2H1,OX1-]")


def _carbonyl_oxygen(c) -> Optional[int]:
    return next((b.GetOtherAtomIdx(c.GetIdx()) for b in c.GetBonds()
                 if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(c).GetSymbol() == "O"), None)


def _boron_sites(mol):
    """``(number of B-OH, [(carbonyl C, bridging O, carbonyl O)])`` of the one acyclic neutral boron of ``mol`` whose
    three neighbours are each an -OH or an -O-CO- bridge to an acyl group, else None."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return None
    borons = [a for a in mol.GetAtoms() if a.GetSymbol() == "B"]
    if len(borons) != 1:
        return None
    b = borons[0]
    if b.IsInRing() or b.GetDegree() != 3 or b.GetTotalNumHs() or b.GetNumRadicalElectrons():
        return None
    hydroxy, sites = 0, []
    for o in b.GetNeighbors():
        others = [x for x in o.GetNeighbors() if x.GetIdx() != b.GetIdx()]
        if o.GetSymbol() != "O" or o.IsInRing() or o.GetFormalCharge():
            return None
        if not others and o.GetTotalNumHs() == 1:
            hydroxy += 1
            continue
        if len(others) != 1 or others[0].GetSymbol() != "C" or o.GetTotalNumHs():
            return None
        carbonyl_o = _carbonyl_oxygen(others[0])
        # a carboxylic acyl group, R-CO- with R carbon or hydrogen: a carbamic or carbonic one has its own prefix forms
        if carbonyl_o is None or any(x.GetSymbol() not in ("C", "H") for x in others[0].GetNeighbors()
                                     if x.GetIdx() not in (o.GetIdx(), carbonyl_o)):
            return None
        sites.append((others[0].GetIdx(), o.GetIdx(), carbonyl_o))
    # each acyl group is its own side: two acyl groups joined to each other would be one acid named twice
    bridges = {o for _, o, _ in sites}
    from .anhydrides import _bfs_side_atoms
    for c, _o, _co in sites:
        side = _bfs_side_atoms(mol, c, bridges)
        if any(other in side for other, _, _ in sites if other != c):
            return None
    if not sites or mol.HasSubstructMatch(_OTHER_ACID):
        return None     # takes "the senior acid as parent" (:35990): a sulfo, phosphono or carboxy group outranks B
    return hydroxy, sites


def name_acyloxy_boron_acid(mol) -> Optional[str]:
    """The name of an acyl anhydride of boric acid (module docstring), or None."""
    found = _boron_sites(mol)
    if found is None:
        return None
    hydroxy, sites = found
    if hydroxy == 0:
        from .anhydrides import _name_acyl_acid
        acids = {_name_acyl_acid(mol, c, o, co) for c, o, co in sites}
        if len(acids) != 1:
            return None
        (stem, substituted), = acids
        if stem is None or substituted or " " in stem:
            return None
        return " ".join(sorted([stem, "boric"])) + " trianhydride"
    from ..assembly.naming_utils import enclose_if_compound
    from .lipids import _acyloxy_for_site
    prefixes = [_acyloxy_for_site(mol, ("acyl", c, o)) for c, o, _co in sites]
    if None in prefixes:
        return None
    if hydroxy == 2:
        return f"{enclose_if_compound(prefixes[0])}boronic acid"
    if prefixes[0] == prefixes[1]:
        return f"bis{enclose_if_compound(prefixes[0])}borinic acid"
    from ..assembly.naming_utils import alpha_sort_key_total
    return "".join(enclose_if_compound(p) for p in sorted(prefixes, key=alpha_sort_key_total)) + "borinic acid"
