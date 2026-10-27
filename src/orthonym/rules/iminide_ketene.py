"""The iminide of a ketene imine: '2-cyano-2-(trifluoromethanesulfonyl)ethen-1-iminide'.

 "Anions derived from amines and imines" (the Blue Book): an imine with one negative charge on its
nitrogen is named with the suffix 'iminide', formed by the addition of 'ide' to the suffix 'imine' ('butaniminide
(PIN)',:41061). The anion R1R2C=C=[N-] of a ketene imine is the iminide of the ethene whose carbon 1 holds the
nitrogen, with the two groups of carbon 2 as prefixes (ketenes are named on 'ethenone',,:29292:
'dibromoethenone (PIN)',:29308). The groups are named as the substituent groups of carbon 2, as in the
one-carbon iminide (``iminide_one_carbon``): a carbon chain on carbon 2 is a group, not a longer parent, so the
name is recorded as not the PIN, which would be the iminide of the longest unsaturated chain. The name is read
back to the full InChIKey by the caller's gate before it is returned.
"""
from rdkit import Chem


def _arm_candidates(mol, ketene_carbon: int, arms):
    """The candidate prefix names of each group on ``ketene_carbon``, in the order of ``arms`` (``group_prefixes``),
    or None when a group has none or a ring runs through carbon 2 (a ylidene group, not two groups)."""
    from .group_prefixes import group_prefix_candidates
    rw = Chem.RWMol(mol)
    for arm in arms:
        rw.RemoveBond(ketene_carbon, arm)
    components = Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
    found = []
    for arm in arms:
        atoms = next((list(c) for c in components if arm in c), None)
        if not atoms or ketene_carbon in atoms:
            return None
        names = group_prefix_candidates(mol, atoms, arm, 'yl', [ketene_carbon])
        if not names:
            return None
        found.append(names)
    return found


def _ring_ylidene_iminide(mol, c1: int, c2: int) -> str:
    """'(ring-ylidene)methaniminide' for ring=C=[N-]: the ring that carbon 2 belongs to is a group with two free
    valences on it, an '-ylidene' of the substitutive path (``centre_ligands``), on the one-carbon parent
    'methanimine', the Blue Book), whose carbon holds the ylidene group. '' on any decline."""
    from ..assembly.naming_utils import enclose_if_compound
    from .centre_ligands import _ligand_prefix
    rw = Chem.RWMol(mol)
    rw.RemoveBond(c1, c2)
    atoms = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False) if c2 in c), None)
    prefix = _ligand_prefix(mol, atoms, c2) if atoms and c1 not in atoms else None
    return f"{enclose_if_compound(prefix)}methaniminide" if prefix and prefix.endswith('ylidene') else ''


def emit_ketene_iminide(mol, anion_idx: int) -> str:
    """'2-X-2-Y-ethen-1-iminide' for R2C=C=[N-] with carbon 2 holding one or two groups, '(R-ylidene)methaniminide'
    for ring=C=[N-], else ''."""
    n = mol.GetAtomWithIdx(anion_idx)
    if (n.GetSymbol() != 'N' or n.GetFormalCharge() != -1 or n.GetTotalNumHs() or n.GetDegree() != 1
            or n.GetIsAromatic() or n.IsInRing()):
        return ''
    c1 = n.GetNeighbors()[0]
    if (c1.GetSymbol() != 'C' or c1.GetFormalCharge() or c1.IsInRing() or c1.GetDegree() != 2 or c1.GetTotalNumHs()
            or mol.GetBondBetweenAtoms(anion_idx, c1.GetIdx()).GetBondType() != Chem.BondType.DOUBLE):
        return ''
    c2 = next(a for a in c1.GetNeighbors() if a.GetIdx() != anion_idx)
    arms = [a.GetIdx() for a in c2.GetNeighbors() if a.GetIdx() != c1.GetIdx()]
    if (c2.GetSymbol() != 'C' or c2.GetFormalCharge() or c2.GetIsAromatic()
            or mol.GetBondBetweenAtoms(c1.GetIdx(), c2.GetIdx()).GetBondType() != Chem.BondType.DOUBLE
            or len(arms) + c2.GetTotalNumHs() != 2):
        return ''
    if c2.IsInRing():
        from .group_prefixes import first_name_that_reads_back
        return first_name_that_reads_back(mol, [_ring_ylidene_iminide(mol, c1.GetIdx(), c2.GetIdx())])
    candidates = _arm_candidates(mol, c2.GetIdx(), arms)
    if candidates is None:
        return ''
    from .group_prefixes import combinations, first_name_that_reads_back
    return first_name_that_reads_back(mol, (_ethene_iminide(choice) for choice in combinations(candidates)))


def _ethene_iminide(prefixes) -> str:
    """'2-X-2-Y-ethen-1-iminide' of the prefixes of the groups of carbon 2."""
    from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound, multiplied_component
    names = sorted(prefixes, key=alpha_sort_key)
    if len(names) == 2 and names[0] == names[1]:
        body = '2,2-' + multiplied_component(2, names[0], enclose_if_compound(names[0]))
    else:
        body = '-'.join(f"2-{enclose_if_compound(p)}" for p in names)
    return f"{body}ethen-1-iminide"
