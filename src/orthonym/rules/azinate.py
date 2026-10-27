"""The azinate anion: '{ethylidene}azinate' for CH3-CH=N(O)O-, 'methylazinate' for CH3-NH(O)O-.

 "aci-Nitro compounds" (the Blue Book): "Compounds containing the group =N(O)OH are named as
derivatives of azinic acid, H2N(O)-OH, a preselected name" (:25969): 'ethylideneazinic acid (PIN)' (:25973), as in
 "aci-Nitro compounds" (:36564, 'methylideneazinic acid (PIN)',:36570). The anion of the acid is named with
the ending 'ate' "Anions derived from acids",:40953,,:40955: the 'ic acid' ending of
the acid name is replaced by 'ate'), the groups on its nitrogen atom being prefixes: an '-ylidene' for the double bond
of an aci-nitro group (R2C=N(O)O-), an '-yl' for a single one (R-NH(O)O-, R2N(O)O-). The Blue Book gives no example
of the anion, so the name is recorded as not the PIN.

The charge-separated drawing of the anion, R2C=[N+]([O-])[O-], gives the nitrogen two oxide atoms. The perception of
the internal charges (``perception.ions``, unchanged here) takes each oxide that is bonded to a nitrogen cation for
half of a nitro or N-oxide pair, so the molecule has no ionic centre, the handlers of the ions never see it, and it was
named as a neutral one or not at all. The fix is the predicate of the dispatch row of the azinic acids
(``routing.dispatch_table._is_azinic_derivative``): the nitrogen cation balances one of its oxides, the other is the
anion of the name.
"""
from functools import lru_cache

from rdkit import Chem


def _nitrogen_with_two_oxides(mol):
    """The index of the one acyclic N+ that holds exactly two terminal O-, in a molecule of net charge -1 (the
    nitrogen cation balances one of its oxides, the other is the anion), else None."""
    if Chem.GetFormalCharge(mol) != -1:
        return None
    found = [atom.GetIdx() for atom in mol.GetAtoms()
             if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1 and not atom.IsInRing()
             and sum(1 for n in atom.GetNeighbors()
                     if n.GetSymbol() == 'O' and n.GetFormalCharge() == -1 and n.GetDegree() == 1) == 2]
    return found[0] if len(found) == 1 else None


def emit_azinate(mol) -> str:
    """The name of the azinate anion R2C=[N+]([O-])[O-] / R-[NH+]([O-])[O-], else ''. The dispatch asks the
    predicate and the handler for the same molecule, so the name, read back through OPSIN, is made once."""
    if _nitrogen_with_two_oxides(mol) is None:
        return ''
    from ..metrics.provenance import record_non_pin_fragment
    from .group_prefixes import tier_key
    name = _azinate_name(Chem.MolToSmiles(mol), tier_key())
    if name:
        record_non_pin_fragment(name)   # each time the name is handed out: the memo does not record it
    return name


@lru_cache(maxsize=128)
def _azinate_name(smiles: str, tiers: tuple) -> str:
    """``_build_azinate`` of the molecule of ``smiles`` in the tiers ``tiers`` (the names a group has depend on them;
    see ``emit_azinate``)."""
    return _build_azinate(Chem.MolFromSmiles(smiles))


def _build_azinate(mol) -> str:
    """A single N+ with two oxides and one or two groups on it, every group an acyclic or ring-ylidene / yl group
    of the substitutive path, read back to the full InChIKey; '' on any decline."""
    from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound, multiplied_component
    from .group_prefixes import combinations, first_name_that_reads_back, group_prefix_candidates
    n_idx = _nitrogen_with_two_oxides(mol)
    if n_idx is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    nitrogen = mol.GetAtomWithIdx(n_idx)
    groups = [a for a in nitrogen.GetNeighbors() if not (a.GetSymbol() == 'O' and a.GetDegree() == 1)]
    if not groups or sum(int(mol.GetBondBetweenAtoms(n_idx, a.GetIdx()).GetBondTypeAsDouble()) for a in groups) \
            + nitrogen.GetTotalNumHs() != 2:   # the two hydrogens of H2N(O)OH, a double bond holding both
        return ''
    candidates = []
    for group in groups:
        double = mol.GetBondBetweenAtoms(n_idx, group.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        rw = Chem.RWMol(mol)
        rw.RemoveBond(n_idx, group.GetIdx())
        atoms = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                      if group.GetIdx() in c), None)
        if not atoms or n_idx in atoms:
            return ''
        names = group_prefix_candidates(mol, atoms, group.GetIdx(), 'ylidene' if double else 'yl', [n_idx])
        if not names:
            return ''
        candidates.append(names)

    def name_of(choice) -> str:
        names = sorted(choice, key=alpha_sort_key)
        if len(names) == 2 and names[0] == names[1]:
            return multiplied_component(2, names[0], enclose_if_compound(names[0])) + 'azinate'
        return ''.join(enclose_if_compound(p) for p in names) + 'azinate'

    return first_name_that_reads_back(mol, (name_of(choice) for choice in combinations(candidates)), record=False)
