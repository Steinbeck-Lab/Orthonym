"""The anion R2C=[N-] of an imine named as the azanide of its ylidene group: '(propan-2-ylidene)azanide'.

 "Anionic centers on other characteristic groups" (the Blue Book): "Anionic centers generated
formally by the removal of hydrons from atoms of characteristic groups other than those considered in,
, and are named on the basis of the corresponding anionic parent hydrides": 'acetylazanide
(PIN)' (:41079). The anion of an imine whose carbon holds groups that the one-carbon and the ketene iminide cannot
cite (an amidine, a carbon chain, a ring) is the azanide, NH2(-), with the group of the carbon as its '-ylidene'
prefix, a substituent group of two free valences. The preferred name of the anion of an imine is its
'iminide',:41047), so the name is recorded as not the PIN; it is read back to the full InChIKey by
the caller's gate before it is returned.
"""
from rdkit import Chem


def emit_ylidene_azanide(mol, anion_idx: int) -> str:
    """'(R-ylidene)azanide' for R=[N-] with a group R that has an '-ylidene' prefix, else ''."""
    from ..assembly.naming_utils import apply_enclosing_marks
    from .group_prefixes import first_name_that_reads_back, group_prefix_candidates
    n = mol.GetAtomWithIdx(anion_idx)
    if (n.GetSymbol() != 'N' or n.GetFormalCharge() != -1 or n.GetTotalNumHs() or n.GetDegree() != 1
            or n.GetIsAromatic() or n.IsInRing()):
        return ''
    attach = n.GetNeighbors()[0]
    if attach.GetSymbol() != 'C' or mol.GetBondBetweenAtoms(anion_idx, attach.GetIdx()).GetBondType() != Chem.BondType.DOUBLE:
        return ''
    rw = Chem.RWMol(mol)
    rw.RemoveBond(anion_idx, attach.GetIdx())
    group = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                  if attach.GetIdx() in c), None)
    if not group or anion_idx in group:
        return ''
    prefixes = group_prefix_candidates(mol, group, attach.GetIdx(), 'ylidene', [anion_idx])
    return first_name_that_reads_back(mol, (f"{apply_enclosing_marks(p, -1)}azanide" for p in prefixes))
