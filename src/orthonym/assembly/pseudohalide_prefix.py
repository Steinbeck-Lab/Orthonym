"""The prefix of a terminal -E-C#N group (E = O, S, Se, Te): 'cyanato', 'thiocyanato'.

 (the Blue Book): "'cyano' for -CN and 'cyanato' for -O-CN, 'thiocyanato' for -S-CN, 'selenocyanato'
for -Se-CN"; '3-(thiocyanato)propanoic acid (PIN)' (:31013), not 'carbononitridoylsulfanyl'. The general substituent
namer otherwise spells the group as a chalcogen with a cyano prefix ('(cyano)oxy'), a name that is valid but not the
preferred prefix.
"""
from typing import List, Optional

_PREFIX = {"O": "cyanato", "S": "thiocyanato", "Se": "selenocyanato", "Te": "tellurocyanato"}


def pseudohalide_prefix(mol, sub_atoms: List[int], attach_idx: int) -> Optional[str]:
    """'cyanato' when ``sub_atoms`` is exactly E-C#N attached through E, else None."""
    from rdkit import Chem
    if len(sub_atoms) != 3:
        return None
    e = mol.GetAtomWithIdx(attach_idx)
    if e.GetSymbol() not in _PREFIX or e.GetFormalCharge() or e.GetTotalNumHs():
        return None
    c = [nb for nb in e.GetNeighbors() if nb.GetIdx() in sub_atoms]
    if len(c) != 1 or c[0].GetSymbol() != "C":
        return None
    n = [nb for nb in c[0].GetNeighbors() if nb.GetIdx() in sub_atoms and nb.GetIdx() != attach_idx]
    if (len(n) != 1 or n[0].GetSymbol() != "N"
            or mol.GetBondBetweenAtoms(c[0].GetIdx(), n[0].GetIdx()).GetBondType() != Chem.BondType.TRIPLE):
        return None
    return _PREFIX[e.GetSymbol()]
