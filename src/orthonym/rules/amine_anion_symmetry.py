"""An amine anion whose nitrogen the suffix seam cannot tell from another amine nitrogen of the molecule.

``route_charged`` ionizes the suffix of the neutral name. For ``C[N-]CCNC`` the neutral is
``N1,N2-dimethylethane-1,2-diamine`` and the seam writes a ``diaminide``, a dianion, which does not read back; the
composer's ion aspect names the same molecule on one amine as the parent suffix ('N-methyl-2-(methylamino)ethan-1-aminide',
 "Anions derived from amines and imines", the Blue Book), and that name reads back because the charged
nitrogen is equivalent to the suffix nitrogen. The azanide of the groups is a valid name of the same anion, not the PIN
,:41069), so it must not pre-empt the aminide in that case.
"""
from rdkit import Chem


def anion_equals_another_secondary_amine(mol, anion_idx: int) -> bool:
    """True when the [N-] of a secondary amine anion is, by the molecule's own symmetry, equivalent to another
    nitrogen with two carbon neighbours: the charge sits on a nitrogen the neutral name treats as its suffix."""
    anion = mol.GetAtomWithIdx(anion_idx)
    if anion.GetDegree() != 2 or any(n.GetSymbol() != 'C' for n in anion.GetNeighbors()):
        return False
    work = Chem.RWMol(mol)
    atom = work.GetAtomWithIdx(anion_idx)
    atom.SetFormalCharge(0)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(work)
    except (RuntimeError, ValueError):
        return False
    ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False))
    return any(a.GetIdx() != anion_idx and a.GetSymbol() == 'N' and ranks[a.GetIdx()] == ranks[anion_idx]
               for a in work.GetAtoms())
