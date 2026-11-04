"""The multiple bonds of the principal chain of an acyclic substituent group that is attached inside its chain.

 "THE PRINCIPAL SUBSTITUENT CHAIN" (the Blue Book): (b) "the greater number of skeletal atoms, i.e., the
longest chain" comes before (d) "the greater number of multiple bonds regardless of type", so the chain of a group whose
free valence lies inside its carbon chain runs through the free valence and is cited with its own unsaturation:
'3-hydroxy-5-phenylpent-4-yn-2-yl', not the chain counted from the free valence ('2-hydroxy-1-methyl-4-phenylbut-3-yn-1-yl').
"""
from typing import List, Optional, Tuple


def chain_multiple_bonds(mol, chain, carbon_set, attach_locant: int) -> Optional[Tuple[List[int], List[int]]]:
    """``(double bond locants, triple bond locants)`` of the multiple bonds of ``chain`` (the locant of a bond is the
    lower of its two atoms'), or None when this deriver does not name the group: a multiple bond between two acyclic
    carbon atoms of ``carbon_set`` that the chain does not hold (an ylidene or a vinyl branch: the name would not spell
    it), or a chain that starts at the free valence (``attach_locant`` 1), which the writer of the best-effort groups
    names with its own locant omissions ('(hydroxyethynyl)benzene',."""
    pos = {a: i for i, a in enumerate(chain, 1)}
    double_locs: List[int] = []
    triple_locs: List[int] = []
    for b in mol.GetBonds():
        order = b.GetBondTypeAsDouble()
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if order < 2.0 or i not in carbon_set or j not in carbon_set:
            continue
        if i not in pos or j not in pos:
            return None
        (double_locs if order == 2.0 else triple_locs).append(min(pos[i], pos[j]))
    if attach_locant == 1 and (double_locs or triple_locs):
        return None
    return sorted(double_locs), sorted(triple_locs)
