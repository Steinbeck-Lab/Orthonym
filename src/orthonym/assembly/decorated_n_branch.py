"""A group on an amide nitrogen that the PIN-tier amide prefix may cite: an alkyl group decorated with simple prefixes.

 (the Blue Book): "Carbamoyl and amino groups may be substituted in the normal way"; the amide nitrogen
of a chain carboxamide prefix carries its substituents as the prefixes of 'amino' ('2-anilino-2-oxoethyl (preferred prefix)',
 :24603). A peptide named systematically (no PIN is identified for the peptide names of::2052) is built
from such prefixes: '(3S)-3-amino-4-({(2S)-1-[(carboxymethyl)amino]-3-(1H-indol-3-yl)-1-oxopropan-2-yl}amino)-4-oxobutanoic
acid', where each nitrogen carries a 'carboxyalkyl' or another chain carboxamide.

``chain_carboxamide`` admitted only saturated alkyl or ring-bearing groups there. This module admits, besides those, a
saturated acyclic carbon chain whose other atoms are carboxy (-COOH), hydroxy, sulfanyl, methylsulfanyl or amino groups or
ring-bearing groups, and a group that is itself a chain carboxamide (the next residue). Every name is still read back and
spelling-checked; this decides only which shapes the PIN-tier located namer is asked to spell.
"""
from rdkit import Chem


def _is_carboxy_carbon(atom) -> bool:
    """A -C(=O)OH carbon: one terminal =O, one -OH, one other neighbour."""
    if atom.GetSymbol() != 'C' or atom.IsInRing() or atom.GetDegree() != 3:
        return False
    oxo = [b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
           and b.GetOtherAtom(atom).GetSymbol() == 'O' and b.GetOtherAtom(atom).GetDegree() == 1]
    oh = [b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE
          and b.GetOtherAtom(atom).GetSymbol() == 'O' and b.GetOtherAtom(atom).GetDegree() == 1
          and b.GetOtherAtom(atom).GetTotalNumHs() == 1]
    return len(oxo) == 1 and len(oh) == 1


def decorated_alkyl_branch(mol, comp, start: int) -> bool:
    """True when the atoms ``comp`` (entered at ``start``, a carbon) are a saturated acyclic carbon chain bearing only
    carboxy, hydroxy, sulfanyl, methylsulfanyl, amino and ring-bearing groups."""
    comp = set(comp)
    a0 = mol.GetAtomWithIdx(start)
    if a0.GetSymbol() != 'C' or a0.IsInRing():
        return False
    for idx in comp:
        a = mol.GetAtomWithIdx(idx)
        if a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope():
            return False
        if a.IsInRing():
            continue                                    # a ring-bearing group: the ring namer names it
        sym = a.GetSymbol()
        if sym == 'C':
            if _is_carboxy_carbon(a):
                continue
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
                return False
        elif sym == 'O':
            nbs = a.GetNeighbors()
            if not (a.GetDegree() == 1 and (_is_carboxy_carbon(nbs[0]) or (
                    a.GetTotalNumHs() == 1 and not nbs[0].IsInRing()))):
                return False
        elif sym == 'S':
            heavy = [nb for nb in a.GetNeighbors()]
            if a.GetDegree() == 1 and a.GetTotalNumHs() == 1:
                continue                                # sulfanyl
            if not (a.GetDegree() == 2 and all(nb.GetSymbol() == 'C' and not nb.IsInRing() for nb in heavy)
                    and any(nb.GetDegree() == 1 for nb in heavy)):
                return False                            # methylsulfanyl only
        elif sym == 'N':
            if not (a.GetDegree() == 1 and a.GetTotalNumHs() == 2):
                return False                            # amino only
        else:
            return False
    return True


def acceptable_n_branch(mol, comp, start: int, n_idx: int) -> bool:
    """True for a group ``comp`` on the amide nitrogen ``n_idx`` (entered at ``start``) that the PIN-tier amide prefix may
    cite: ring-bearing, a decorated alkyl group (:func:`decorated_alkyl_branch`), or itself a chain carboxamide."""
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in comp) and mol.GetAtomWithIdx(start).IsInRing():
        return True
    if decorated_alkyl_branch(mol, comp, start):
        return True
    return _carboxamide_chain_branch(mol, set(comp), start)


def _carboxamide_chain_branch(mol, comp, start: int) -> bool:
    """True when ``comp`` (entered at ``start``) is a decorated alkyl chain ending in its one carboxamide -C(=O)-N<, whose N
    carries acceptable branches in turn (the next residue). Each residue is visited once: the recursion follows only the
    branch's own amide nitrogen, so the cost grows linearly with the chain."""
    a0 = mol.GetAtomWithIdx(start)
    if a0.GetSymbol() != 'C' or a0.IsInRing():
        return False
    carbons, stack = set(), [start]
    while stack:
        i = stack.pop()
        if i in carbons:
            continue
        carbons.add(i)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                     if nb.GetIdx() in comp and nb.GetSymbol() == 'C' and not nb.IsInRing())
    amides = []
    for c in carbons:
        atom = mol.GetAtomWithIdx(c)
        oxo = [b.GetOtherAtom(atom).GetIdx() for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
               and b.GetOtherAtom(atom).GetSymbol() == 'O' and b.GetOtherAtom(atom).GetDegree() == 1]
        ns = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetSymbol() == 'N' and nb.GetIdx() in comp
              and not nb.IsInRing()]
        if len(oxo) == 1 and len(ns) == 1 and c != start:
            amides.append((c, oxo[0], ns[0]))
    if len(amides) != 1:
        return False
    cc, oxo, n = amides[0]
    n_side = set()
    for nb in mol.GetAtomWithIdx(n).GetNeighbors():
        j = nb.GetIdx()
        if j == cc or j not in comp:
            continue
        side, st = set(), [j]
        while st:
            k = st.pop()
            if k in side or k == n:
                continue
            side.add(k)
            st.extend(x.GetIdx() for x in mol.GetAtomWithIdx(k).GetNeighbors() if x.GetIdx() in comp)
        if cc in side or not acceptable_n_branch(mol, side, j, n):
            return False
        n_side |= side
    rest = comp - n_side - {cc, oxo, n}
    return start in rest and decorated_alkyl_branch(mol, rest, start)
