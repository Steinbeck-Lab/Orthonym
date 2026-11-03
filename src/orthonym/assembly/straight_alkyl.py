"""The carbon-count alkyl name, only for the shape it names.

``naming_utils.get_alkyl_name(n)`` gives 'methyl', 'ethyl',..., 'heptyl': the prefix of an unbranched saturated
chain of n carbons bound by its end carbon (1), the Blue Book, the free valence at position 1 of
a straight chain). Several producers used it as the last resort for a branch their namer declined, so a count of
carbons named whatever the branch was: '4-heptylcyclohexane-1-carboxylic acid' for the phenylcarbamoyl branch
'-C(=O)NHC6H5' (seven carbons), '...butyl...' for a 2,5-dioxopyrrol-1-yl ring. Only the round trip withdrew those
names.:func:`straight_alkyl_name` gives the count's name for that one shape and None for every other branch, so
the caller declines as it does for any branch its namer leaves without a name.
"""
from typing import Iterable, Optional

from rdkit import Chem


def straight_alkyl_name(mol, atoms: Iterable[int]) -> Optional[str]:
    """``get_alkyl_name(len(atoms))`` when ``atoms`` are an unbranched saturated acyclic chain of uncharged,
    unlabelled carbons with one bond out of the chain, a single bond from an end carbon; else None."""
    frag = set(atoms)
    if not frag:
        return None
    outside = []
    for idx in frag:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()):
            return None
        inner = 0
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in frag:
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
                inner += 1
            elif mol.GetAtomWithIdx(other).GetAtomicNum() != 1:
                outside.append((idx, bond))
        if inner > 2:
            return None
    if len(outside) != 1 or outside[0][1].GetBondType() != Chem.BondType.SINGLE:
        return None
    end = outside[0][0]
    if len(frag) > 1 and sum(1 for b in mol.GetAtomWithIdx(end).GetBonds() if b.GetOtherAtomIdx(end) in frag) != 1:
        return None
    from .naming_utils import get_alkyl_name
    try:
        return get_alkyl_name(len(frag))
    except (ValueError, KeyError):
        return None
