"""A carboxamide at the end of a carbon chain, the substituent the PIN tier names by method (1) of."""
from rdkit import Chem


def is_chain_carboxamide(mol, sub_atoms, attach_idx) -> bool:
    """True for a substituent that is an acyclic carbon chain of two or more
    carbons, attached at a carbon, ending in a carboxamide -C(=O)-N< whose N
    carries only hydrogen, saturated carbon or ring-bearing groups:
    "Carbamoyl and amino groups may be substituted in the normal way",
    the Blue Book; 'C6H5-NH-CO-CH2– 2-anilino-2-oxoethyl (preferred prefix)',
    ,:24603). Apart from the ring-bearing groups on the N, the fragment holds
    no other heteroatom, no ring atom, no charge and no radical, so the chain's name
    is the method (1) 'amino' + 'oxo' form and nothing else in it is decided here;
    a ring-bearing group on the N is named by the ring namer as the substituent of
    the 'amino' prefix ('2-(cyclohexylamino)-2-oxoethyl')."""
    sub_set = set(sub_atoms)
    if _chain_carboxamide_core(mol, sub_set, attach_idx, frozenset()):
        return True
    for n in sub_set:
        a = mol.GetAtomWithIdx(n)
        if a.GetSymbol() != 'N' or a.IsInRing() or n == attach_idx:
            continue
        branches = set()
        for nb in a.GetNeighbors():
            start = nb.GetIdx()
            if start not in sub_set:
                continue
            comp, stack = {start}, [start]
            while stack:
                cur = stack.pop()
                for x in mol.GetAtomWithIdx(cur).GetNeighbors():
                    xi = x.GetIdx()
                    if xi in sub_set and xi != n and xi not in comp:
                        comp.add(xi)
                        stack.append(xi)
            if attach_idx not in comp and any(mol.GetAtomWithIdx(i).IsInRing() for i in comp):
                branches |= comp
        if branches and _chain_carboxamide_core(mol, sub_set - branches, attach_idx, frozenset(branches)):
            return True
    return False


def _chain_carboxamide_core(mol, sub_set, attach_idx, n_branches) -> bool:
    """:func:`is_chain_carboxamide` on the atoms ``sub_set`` (the fragment without the ring-bearing groups
    ``n_branches`` on the amide N)."""
    try:
        if attach_idx not in sub_set or mol.GetAtomWithIdx(attach_idx).GetSymbol() != 'C':
            return False
        ring_info = mol.GetRingInfo()
        o_atoms, n_atoms = [], []
        for idx in sub_set:
            a = mol.GetAtomWithIdx(idx)
            if (ring_info.NumAtomRings(idx) or a.GetFormalCharge()
                    or a.GetNumRadicalElectrons()):
                return False
            sym = a.GetSymbol()
            if sym == 'O':
                o_atoms.append(a)
            elif sym == 'N':
                n_atoms.append(a)
            elif sym != 'C':
                return False
        if len(o_atoms) != 1 or len(n_atoms) != 1:
            return False
        o, n = o_atoms[0], n_atoms[0]
        if o.GetDegree() != 1:
            return False
        cc = o.GetNeighbors()[0]
        if (mol.GetBondBetweenAtoms(o.GetIdx(), cc.GetIdx()).GetBondType()
                != Chem.BondType.DOUBLE or cc.GetSymbol() != 'C'
                or mol.GetBondBetweenAtoms(n.GetIdx(), cc.GetIdx()) is None):
            return False
        chain_nbrs = [x for x in cc.GetNeighbors()
                      if x.GetIdx() in sub_set and x.GetSymbol() == 'C']
        if len(chain_nbrs) != 1 or cc.GetIdx() == attach_idx:
            return False          # the amide carbon ends a chain of >= 2 carbons
        # The amide carbon is reached from the attachment through carbon atoms
        # only: an N between them is the reversed amide -NH-CO-R (an acylamino
        # group,, not a carboxamide at the chain end.
        seen, stack = {attach_idx}, [attach_idx]
        while stack:
            cur = stack.pop()
            for x in mol.GetAtomWithIdx(cur).GetNeighbors():
                xi = x.GetIdx()
                if xi in sub_set and xi not in seen and x.GetSymbol() == 'C':
                    seen.add(xi)
                    stack.append(xi)
        if cc.GetIdx() not in seen:
            return False
        for x in n.GetNeighbors():
            if x.GetIdx() == cc.GetIdx() or x.GetIdx() in n_branches:
                continue
            if x.GetSymbol() != 'C' or x.GetIdx() not in sub_set or x.GetIsAromatic():
                return False
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in x.GetBonds()):
                return False
        return True
    except Exception:
        return False


def is_ring_amino(mol, sub_atoms, attach_idx) -> bool:
    """True for an amino group attached at its nitrogen to an acyclic acyl carbon, whose other substituents are
    hydrogen, ring-bearing groups and saturated alkyl groups ('ethylamino'): the 'anilino', '(cyclohexylamino)' prefix of the amide nitrogen of a chain
    carboxamide, the Blue Book; '2-anilino-2-oxoethyl (preferred prefix)',:24603)."""
    sub_set = set(sub_atoms)
    if attach_idx not in sub_set:
        return False
    n = mol.GetAtomWithIdx(attach_idx)
    if n.GetSymbol() != 'N' or n.IsInRing() or n.GetFormalCharge() or n.GetNumRadicalElectrons():
        return False
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in n.GetBonds()):
        return False
    inner = [nb for nb in n.GetNeighbors() if nb.GetIdx() in sub_set]
    outer = [nb for nb in n.GetNeighbors() if nb.GetIdx() not in sub_set]
    # the nitrogen of an amide: its parent atom is the acyl carbon of the chain ('2-anilino-2-oxoethyl')
    acyl = len(outer) == 1 and outer[0].GetSymbol() == 'C' and not outer[0].IsInRing() and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(outer[0]).GetSymbol() == 'O'
        for b in outer[0].GetBonds())
    return acyl and bool(inner) and all(nb.IsInRing() or _saturated_alkyl_branch(mol, nb.GetIdx(), attach_idx, sub_set)
                                         for nb in inner)


def _saturated_alkyl_branch(mol, start: int, n_idx: int, sub_set) -> bool:
    """True when the branch of ``sub_set`` that starts at ``start`` (away from the nitrogen ``n_idx``) is all
    saturated acyclic carbon: the alkyl group of 'ethylamino', as:func:`is_chain_carboxamide` reads the
    N-substituents of a carboxamide)."""
    seen, stack = {start}, [start]
    while stack:
        a = mol.GetAtomWithIdx(stack.pop())
        if a.GetSymbol() != 'C' or a.IsInRing() or a.GetFormalCharge() or any(
                b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
            return False
        for nb in a.GetNeighbors():
            j = nb.GetIdx()
            if j in sub_set and j != n_idx and j not in seen:
                seen.add(j)
                stack.append(j)
    return True


def pin_amide_prefix(mol, sub_atoms, attach_idx) -> bool:
    """Whether the PIN tier names the fragment by the located namer as an amide prefix: a chain carboxamide
    (:func:`is_chain_carboxamide`) or the ring-bearing amino group on its acyl carbon (:func:`is_ring_amino`)."""
    return is_chain_carboxamide(mol, sub_atoms, attach_idx) or is_ring_amino(mol, sub_atoms, attach_idx)
