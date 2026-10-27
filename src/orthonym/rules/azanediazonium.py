"""The diazonium cation on a nitrogen: 'dimethylazanediazonium' for (CH3)2N-N2(+).

 "Diazonium ions" (the Blue Book): "Cations containing an -N2+ group attached to a parent hydride are
traditionally named according to the principles of substitutive nomenclature by using the suffix 'diazonium'...
The use of the suffix 'diazonium' results in preferred IUPAC names": 'methanediazonium (PIN)' (:41613). The parent
hydride of an amino group is 'azane', NH3 'Systematic names', Table 2.1,:7913: a preselected name), so the -N2+ group of R2N-N2(+) replaces
one hydrogen atom of the azane whose other two hydrogen atoms the groups R replace: 'dimethylazanediazonium', 'acetyl
(methyl)azanediazonium'. The groups on the one atom of a mononuclear parent hydride take no locants and are cited as
the block of (:7272, ``assembly.prefix_marks.mononuclear_substituent_block``). The Blue Book gives no
example of a diazonium ion on a nitrogen, so the name is recorded as not the PIN. It is read back to the full
InChIKey by the caller's gate before it is returned (the other name that the book allows, on the parent cation
'diazenylium', reads back to a valence-two N+ that the read-back of the radical graph refuses).
"""
from rdkit import Chem


def name_azanediazonium(mol, proximal_n: int, amino_n: int) -> str:
    """'<groups>azanediazonium' for R2N-N2(+), ``amino_n`` the nitrogen the -N2+ group hangs from, else ''."""
    from ..assembly.prefix_marks import mononuclear_substituent_block
    from .group_prefixes import combinations, first_name_that_reads_back, group_prefix_candidates
    if sum(1 for a in mol.GetAtoms() if a.GetFormalCharge()) != 1:
        return ''
    nitrogen = mol.GetAtomWithIdx(amino_n)
    if nitrogen.GetSymbol() != 'N' or nitrogen.IsInRing() or nitrogen.GetFormalCharge() or nitrogen.GetIsAromatic():
        return ''
    groups = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetIdx() != proximal_n]
    if len(groups) + nitrogen.GetTotalNumHs() != 2:
        return ''
    candidates = []
    for group in groups:
        rw = Chem.RWMol(mol)
        rw.RemoveBond(amino_n, group)
        atoms = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                      if group in c), None)
        if not atoms or amino_n in atoms:
            return ''
        names = group_prefix_candidates(mol, atoms, group, 'yl', [amino_n])
        if not names:
            return ''
        candidates.append(names)

    def name_of(choice) -> str:
        block = mononuclear_substituent_block(list(choice), max_count=8) if choice else ''
        return f"{block}azanediazonium" if block is not None else ''

    return first_name_that_reads_back(mol, (name_of(choice) for choice in combinations(candidates)))
