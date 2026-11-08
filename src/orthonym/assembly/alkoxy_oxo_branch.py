"""A saturated chain branch of a ring that ends in an ester group: the 'alkoxy...oxo' prefix ('3-methoxy-3-oxopropyl').

 (the Blue Book): "When... another group is present that has priority for citation as the principal
group... an ester group is indicated by prefixes as 'acyloxy' for the group R-CO-O-, and 'alkoxy...oxo',... 'alkoxycarbonyl'
... for the group -CO-OR'"; the book prints the chain form in its PINs: '3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium bromide
(PIN)', not '[2-(ethoxycarbonyl)ethyl]...' (:31709), '3-(methoxycarbonyl)phenyl 2-methoxy-2-oxoethyl butanedioate (PIN)'
(:31886), 'potassium hydrogen 2-(2-ethoxy-2-oxoethyl)-2-hydroxybutanedioate (PIN)' (:31948), and for an S-ester whose R is an
acyl group 'acetic 1-[3-(acetylsulfanyl)-3-oxopropyl]naphthalene-2-carboxylic anhydride (PIN)',:32461).

The branch is taken on a ring only, as the 'alkoxycarbonyl' branch is (``acyl_prefix.is_alkoxycarbonyl_branch``): on a
chain the decomposition and chain-end routes own the ester.
"""
from rdkit import Chem


def _terminal_oxo(mol, atom):
    """The one terminal =O of ``atom``, else None."""
    oxo = [b.GetOtherAtom(atom) for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
           and b.GetOtherAtom(atom).GetSymbol() == 'O' and b.GetOtherAtom(atom).GetDegree() == 1]
    return oxo[0] if len(oxo) == 1 else None


def acyl_oxo_on_link(mol, link, carbonyl_idx: int):
    """The =O index of an acyl group R'-CO- on the ester link atom ``link`` (the side away from ``carbonyl_idx``), else
    None: '(acetylsulfanyl)carbonyl', the Blue Book)."""
    other = [nb for nb in link.GetNeighbors() if nb.GetIdx() != carbonyl_idx]
    if len(other) != 1 or other[0].GetSymbol() != 'C' or other[0].IsInRing():
        return None
    oxo = _terminal_oxo(mol, other[0])
    return oxo.GetIdx() if oxo is not None else None


def is_alkoxy_oxo_chain_branch(mol, sub_atoms, attach_idx) -> bool:
    """True for a branch attached at an acyclic sp3 carbon to a ring atom, made of a saturated acyclic carbon chain whose
    one heteroatom group is an ester -CO-O-R or -CO-S-R on a chain carbon other than the attachment, R carbon atoms only,
    or an acyl group R'-CO- of carbon atoms only. No charge, radical or isotope."""
    try:
        sub_set = set(sub_atoms)
        head = mol.GetAtomWithIdx(attach_idx)
        if attach_idx not in sub_set or head.GetSymbol() != 'C' or head.IsInRing() or _terminal_oxo(mol, head):
            return False
        if not any(nb.IsInRing() for nb in head.GetNeighbors() if nb.GetIdx() not in sub_set):
            return False
        if any(mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
               or mol.GetAtomWithIdx(i).GetIsotope() for i in sub_set):
            return False
        hetero = [i for i in sub_set if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
        links = [i for i in hetero if mol.GetAtomWithIdx(i).GetSymbol() in ('O', 'S')
                 and mol.GetAtomWithIdx(i).GetDegree() == 2 and not mol.GetAtomWithIdx(i).IsInRing()]
        if len(links) != 1:
            return False
        x = mol.GetAtomWithIdx(links[0])
        sides = [nb for nb in x.GetNeighbors()]
        carbonyl = [nb for nb in sides if _terminal_oxo(mol, nb) is not None and not nb.IsInRing()]
        if not carbonyl:
            return False
        # the chain side: from the attachment to the ester carbonyl without crossing the link atom
        chain, stack = set(), [attach_idx]
        while stack:
            a = stack.pop()
            if a in chain or a == x.GetIdx() or a not in sub_set:
                continue
            chain.add(a)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors())
        ester_c = [c for c in carbonyl if c.GetIdx() in chain]
        if len(ester_c) != 1:
            return False
        oxo = _terminal_oxo(mol, ester_c[0]).GetIdx()
        chain.discard(oxo)
        for a in chain:
            atom = mol.GetAtomWithIdx(a)
            if atom.GetSymbol() != 'C' or atom.IsInRing() or atom.GetIsAromatic():
                return False
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()
                   if b.GetOtherAtom(atom).GetIdx() != oxo):
                return False
        r_side = sub_set - chain - {oxo, x.GetIdx()}
        acyl_oxo = [i for i in r_side if mol.GetAtomWithIdx(i).GetSymbol() == 'O']
        if any(mol.GetAtomWithIdx(i).GetSymbol() not in ('C', 'O') for i in r_side):
            return False
        if acyl_oxo:
            # R an acyl group: its one =O on the carbon bonded to the link atom
            r_first = [nb for nb in x.GetNeighbors() if nb.GetIdx() in r_side]
            if len(acyl_oxo) != 1 or len(r_first) != 1 or _terminal_oxo(mol, r_first[0]) is None \
                    or _terminal_oxo(mol, r_first[0]).GetIdx() != acyl_oxo[0]:
                return False
        return True
    except Exception:
        return False
