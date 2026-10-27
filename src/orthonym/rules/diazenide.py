"""The anion R-N=[N-] of a diazene: '2-phenyldiazen-1-ide'.

 "Anions derived from parent hydrides and their derivatives" (the Blue Book): the suffix 'ide' on
the name of the parent hydride describes the anion formed by the loss of a hydron, here 'diazene', HN=NH
"Diazene and related compounds",:38714), so HN=N- is 'diazen-1-ide' and the group R on its other nitrogen a prefix
at locant 2. The locant of the anionic centre is cited and so is the locant of the group,:2869): the two
nitrogen atoms of the anion are no longer equivalent. The group is the one the substitutive path names ('phenyl',
'(2-aminoethyl)'), as for the ligands of a -uide centre (``centre_ligands``). The Blue Book gives no example of the
anion of a substituted diazene, so the name is recorded as not the PIN. It is read back to the full InChIKey by the
caller's gate before it is returned.
"""
from rdkit import Chem


def emit_diazenide(mol, anion_idx: int) -> str:
    """'2-X-diazen-1-ide' for X-N=[N-] with a group X that has a prefix, else ''."""
    from ..assembly.naming_utils import enclose_if_compound
    from .group_prefixes import first_name_that_reads_back, group_prefix_candidates
    n1 = mol.GetAtomWithIdx(anion_idx)
    if (n1.GetSymbol() != 'N' or n1.GetFormalCharge() != -1 or n1.GetTotalNumHs() or n1.GetDegree() != 1
            or n1.GetIsAromatic() or n1.IsInRing()):
        return ''
    n2 = n1.GetNeighbors()[0]
    if (n2.GetSymbol() != 'N' or n2.GetFormalCharge() or n2.IsInRing() or n2.GetTotalNumHs() or n2.GetDegree() != 2
            or mol.GetBondBetweenAtoms(anion_idx, n2.GetIdx()).GetBondType() != Chem.BondType.DOUBLE):
        return ''
    attach = next(a.GetIdx() for a in n2.GetNeighbors() if a.GetIdx() != anion_idx)
    rw = Chem.RWMol(mol)
    rw.RemoveBond(n2.GetIdx(), attach)
    group = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False) if attach in c),
                 None)
    if not group or n2.GetIdx() in group:
        return ''
    prefixes = group_prefix_candidates(mol, group, attach, 'yl', [n2.GetIdx()])
    return first_name_that_reads_back(mol, (f"2-{enclose_if_compound(p)}diazen-1-ide" for p in prefixes))
