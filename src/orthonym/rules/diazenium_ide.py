"""The zwitterion R2N(+)=N(-) of a diazene: '1-phenyldiazen-1-ium-2-ide'.

 "Ionic centers in the same parent structure" (the Blue Book): "Zwitterionic compounds with the ionic
centers in the same parent structure may be named by combining appropriate cumulative suffixes at the end of the name
of a parent hydride in the order 'ium', 'ylium', 'ide', 'uide'" (:42417), '1,2,2,2-tetramethylhydrazin-2-ium-1-ide
(PIN)'. The parent hydride of the N=N group is 'diazene',:38714), so R-NH(+)=N(-) is
'1-R-diazen-1-ium-2-ide', the locants of the 'ium' and the 'ide' cited and the groups of the cationic nitrogen as
prefixes at locant 1.

The same structure is the isodiazene of (:39007), 'R2N-N: <-> R2N(+)=N(-)', whose preferred name is that
of the divalent radical, 'dimethylhydrazinylidene (PIN)' (:39013). The name built here is the one of the structure as
it is drawn, with its two charges, so it is recorded as not the PIN. The two charges are on neighbouring atoms, so the
perception of the species takes the molecule for a neutral one (a 1,2-dipole, and no handler of the ions ever
sees it. The name is read back to the full InChIKey and to the canonical SMILES of the input (the charges on the same
atoms).

The row of this class comes after the row of the cumulative zwitterions (``ions.emit_cumulative_ium_ide``, priority 48.35):
a group on the cationic nitrogen that is itself attached through nitrogen makes the chain a triazene or a longer one, and
 (:43106), 'the preferred name is based on the unbroken nitrogen chain', is the name of that molecule. What
that row cannot name (a ring nitrogen, a group on the chain that is no carbon) is named here.
"""
from functools import lru_cache

from rdkit import Chem


def _diazenium_pair(mol):
    """``(cation, anion)`` atom indices of the one R2N(+)=N(-) pair of a molecule of net charge 0 whose only charged
    atoms they are, with a terminal anionic nitrogen that holds no hydrogen, else None."""
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 2 or sum(a.GetFormalCharge() for a in charged) != 0:
        return None
    cation = next((a for a in charged if a.GetFormalCharge() == 1), None)
    anion = next((a for a in charged if a.GetFormalCharge() == -1), None)
    if (cation is None or anion is None or cation.GetSymbol() != 'N' or anion.GetSymbol() != 'N'
            or cation.IsInRing() or anion.IsInRing() or anion.GetDegree() != 1 or anion.GetTotalNumHs()):
        return None
    bond = mol.GetBondBetweenAtoms(cation.GetIdx(), anion.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return None
    return cation.GetIdx(), anion.GetIdx()


def emit_diazenium_ide(mol) -> str:
    """The name of R2N(+)=N(-) / R-NH(+)=N(-) as '1-R-diazen-1-ium-2-ide', else ''. The dispatch asks the predicate
    and the handler for the same molecule, so the name, read back through OPSIN, is made once."""
    if mol is None or _diazenium_pair(mol) is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    from ..metrics.provenance import record_non_pin_fragment
    from .group_prefixes import tier_key
    name = _diazenium_name(Chem.MolToSmiles(mol), tier_key())
    if name:
        record_non_pin_fragment(name)   # each time the name is handed out: the memo does not record it
    return name


@lru_cache(maxsize=128)
def _diazenium_name(smiles: str, tiers: tuple) -> str:
    """``_build_diazenium_ide`` of the molecule of ``smiles`` in the tiers ``tiers`` (the names a group has depend on them;
    see ``emit_diazenium_ide``)."""
    return _build_diazenium_ide(Chem.MolFromSmiles(smiles))


def _build_diazenium_ide(mol) -> str:
    from ..assembly.naming_utils import alpha_sort_key, enclose_if_compound, multiplied_component
    from .group_prefixes import combinations, first_name_that_reads_back, group_prefix_candidates
    cat_idx, an_idx = _diazenium_pair(mol)
    cation = mol.GetAtomWithIdx(cat_idx)
    groups = [a.GetIdx() for a in cation.GetNeighbors() if a.GetIdx() != an_idx]
    if not groups or len(groups) + cation.GetTotalNumHs() != 2:
        return ''
    candidates = []
    for group in groups:
        rw = Chem.RWMol(mol)
        rw.RemoveBond(cat_idx, group)
        atoms = next((list(c) for c in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                      if group in c), None)
        if not atoms or cat_idx in atoms:
            return ''
        names = group_prefix_candidates(mol, atoms, group, 'yl', [cat_idx])
        if not names:
            return ''
        candidates.append(names)

    def name_of(choice) -> str:
        names = sorted(choice, key=alpha_sort_key)
        if len(names) == 2 and names[0] == names[1]:
            body = '1,1-' + multiplied_component(2, names[0], enclose_if_compound(names[0]))
        else:
            body = '-'.join(f"1-{enclose_if_compound(p)}" for p in names)
        return f"{body}diazen-1-ium-2-ide"

    return first_name_that_reads_back(mol, (name_of(choice) for choice in combinations(candidates)), record=False)
