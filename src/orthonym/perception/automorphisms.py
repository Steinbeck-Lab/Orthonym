"""Graph automorphisms of an atom subset (large-polycycle plan, Tasks 2 and 11)."""
from typing import Dict, List, Set, Tuple

from rdkit import Chem


def skeleton_automorphisms(mol, atoms: Set[int], *, element_blind: bool,
                           cap: int) -> Tuple[List[Dict[int, int]], bool]:
    idx = sorted(atoms)
    pos = {a: i for i, a in enumerate(idx)}
    rw = Chem.RWMol()
    for a in idx:
        z = 6 if element_blind else mol.GetAtomWithIdx(a).GetAtomicNum()
        at = Chem.Atom(z)
        at.SetNoImplicit(True)
        rw.AddAtom(at)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            rw.AddBond(pos[i], pos[j], Chem.BondType.SINGLE)
    sk = rw.GetMol()
    sk.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(sk)
    params = Chem.SubstructMatchParameters()
    params.uniquify = False
    params.useChirality = False
    params.maxMatches = cap + 1
    matches = sk.GetSubstructMatches(sk, params)
    auts = [{idx[q]: idx[t] for q, t in enumerate(m)} for m in matches[:cap]]
    return auts, len(matches) <= cap
