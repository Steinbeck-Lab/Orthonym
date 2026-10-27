"""The anion R2N-[NH-] of a hydrazine: '2,2-dimethylhydrazin-1-ide'.

 "Anions derived from parent hydrides and their derivatives" (the Blue Book): the suffix 'ide' on
the name of the parent hydride describes the anion formed by the loss of a hydron. The parent hydride of the N-N
group is 'hydrazine', "a preselected name",:38509), so H2N-NH(-) is 'hydrazin-1-ide' and the groups
on its other nitrogen atom are prefixes at locant 2; the locant of the anionic centre is cited and so is the locant
of the groups,:2869). The alternative is the azanide with an 'amino' group, 'amino(R)azanide'
,:41069), which the amine anions of ``charged_router`` name; the Blue Book gives no example of the
anion of a substituted hydrazine, so the name is recorded as not the PIN. It is read back to the full InChIKey by the
caller's gate before it is returned.
"""
from rdkit import Chem


def emit_hydrazinide(mol, anion_idx: int) -> str:
    """'2-X-hydrazin-1-ide' / '2,2-X2-hydrazin-1-ide' for X2N-[NH-] with groups that have prefixes, else ''."""
    from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound, multiplied_component
    from .group_prefixes import combinations, first_name_that_reads_back, group_prefix_candidates
    n1 = mol.GetAtomWithIdx(anion_idx)
    if (n1.GetSymbol() != 'N' or n1.GetFormalCharge() != -1 or n1.GetTotalNumHs() != 1 or n1.GetDegree() != 1
            or n1.GetIsAromatic() or n1.IsInRing()):
        return ''
    n2 = n1.GetNeighbors()[0]
    if (n2.GetSymbol() != 'N' or n2.GetFormalCharge() or n2.IsInRing() or n2.GetIsAromatic()
            or mol.GetBondBetweenAtoms(anion_idx, n2.GetIdx()).GetBondType() != Chem.BondType.SINGLE):
        return ''
    groups = [a.GetIdx() for a in n2.GetNeighbors() if a.GetIdx() != anion_idx]
    if not groups or len(groups) + n2.GetTotalNumHs() != 2:
        return ''
    candidates = []
    for group in groups:
        rw = Chem.RWMol(mol)
        rw.RemoveBond(n2.GetIdx(), group)
        atoms = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                      if group in c), None)
        if not atoms or n2.GetIdx() in atoms:
            return ''
        names = group_prefix_candidates(mol, atoms, group, 'yl', [n2.GetIdx()])
        if not names:
            return ''
        candidates.append(names)

    def name_of(choice) -> str:
        names = sorted(choice, key=alpha_sort_key)
        if len(names) == 2 and names[0] == names[1]:
            body = '2,2-' + multiplied_component(2, names[0], enclose_if_compound(names[0]))
        else:
            body = '-'.join(f"2-{enclose_if_compound(p)}" for p in names)
        return f"{body}hydrazin-1-ide"

    return first_name_that_reads_back(mol, (name_of(choice) for choice in combinations(candidates)))
