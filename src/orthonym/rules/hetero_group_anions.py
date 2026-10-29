"""The chalcogen anion of a group on a heteroatom, named on the anionic parent hydride of the chalcogen.

 "Anionic centers on other characteristic groups" (the Blue Book, sentence:41071): the anionic
centres on characteristic groups other than the acids, the hydroxy compounds and the amines
 "are named on the basis of the corresponding anionic parent hydrides" (``tert-butyltrioxidanide
(PIN)``, ``acetylazanide (PIN)``, ``hydroxyazanide``,:41075-:41083). An O-, S- or Se- on a nitrogen, phosphorus
or sulfur atom (the oxygen of a phosphinite, a sulfenate, a hydroxylamine anion) is none of the three: the engine gives
``-olate`` to an oxygen on carbon, and also on silicon or germanium (``trimethylsilanolate``, ``trimethylgermanolate``,
both ``pin_verified``), and no acid ending is at hand where the group is no oxoacid the engine names. An oxygen on a metal
is the salt and complex producers' (the one predicate ``ions._has_metal``), and ``trimethylstannanolate`` would be the
``-olate`` of tin, which no producer takes yet. The anion is the oxidanide, sulfanide or selanide of the
group: ``[methoxy(phenyl)phosphanyl]oxidanide``, ``(methylsulfanyl)oxidanide``. The Blue Book gives no example of
such an anion beside the peroxy ones, so the name is recorded as no PIN. The oxygen of an oxoacid group (a sulfate or
phosphate ester, a sulfinamate) is an acid anion that the producers of the acids name first ('methyl sulfate'); this
module is asked last (``charged_fallback``), only for an anion that no producer named, and gives it the name of its group
('[(diarylamino)sulfinyl]oxidanide') instead of none.

The group is named by the substituent-prefix namer of the substitutive path (``name_substituent_fragment``, then, at
the best-effort tier, the universal one), and the prefixes of the one atom it stands on are assembled by
(:7272). The name ships only when OPSIN reads it back to the input itself, the same atoms with the same charges.
"""
from rdkit import Chem

#: the anionic parent hydride of each chalcogen: 'oxidanide', 'sulfanide', 'selanide', 'tellanide')
_PARENT = {'O': 'oxidanide', 'S': 'sulfanide', 'Se': 'selanide', 'Te': 'tellanide'}


def _group_atoms(mol, anion_idx: int, group_idx: int):
    """The atoms of the group on the anionic atom: the component of ``group_idx`` with the anion's bond cut."""
    rw = Chem.RWMol(mol)
    rw.RemoveBond(anion_idx, group_idx)
    comps = Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
    return next((list(c) for c in comps if group_idx in c), None)


def _group_prefix(mol, atoms, group_idx: int, anion_idx: int):
    """The prefix name of the group, or None."""
    from ..assembly.substituent_naming import name_substituent_fragment
    from ..metrics.provenance import best_effort_ctx
    prefix = name_substituent_fragment(mol, atoms, group_idx, [anion_idx])
    if not prefix and best_effort_ctx.get():
        from ..assembly.universal_substituent import name_universal_substituent_prefix
        prefix = name_universal_substituent_prefix(mol, atoms, group_idx)
    return prefix or None


def hetero_group_chalcogenide(mol, style: str = 'pin') -> str:
    """'<group>oxidanide' (sulfanide, selanide) for the only ion of ``mol``: a single chalcogen anion on a heteroatom
    that is no carbon; '' on any decline."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1:
        return ''
    anion = charged[0]
    parent = _PARENT.get(anion.GetSymbol())
    heavy = [nb for nb in anion.GetNeighbors() if nb.GetSymbol() != 'H']
    if (parent is None or anion.IsInRing() or anion.GetTotalNumHs() or anion.GetNumRadicalElectrons()
            or len(heavy) != 1 or heavy[0].GetSymbol() == 'C'
            or any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms())):
        return ''
    group = heavy[0]
    from .ions import _has_metal
    if group.GetSymbol() == 'H' or _has_metal(mol):
        return ''   # a metal or a heavy element (the one predicate of ``ions``): the salt and complex producers' business
    atoms = _group_atoms(mol, anion.GetIdx(), group.GetIdx())
    prefix = _group_prefix(mol, atoms, group.GetIdx(), anion.GetIdx()) if atoms else None
    if not prefix:
        return ''
    from ..assembly.prefix_marks import mononuclear_substituent_block
    block = mononuclear_substituent_block([prefix], max_count=8)   #,:7272
    name = f"{block}{parent}" if block else ''
    from .charged_router import _charge_sites_rt_ok
    if not name or not _charge_sites_rt_ok(name, mol):
        return ''
    from ..metrics.provenance import record_non_pin_fragment
    record_non_pin_fragment(name)
    return name


def carbon_group_azanide(mol, style: str = 'pin') -> str:
    """'<group>azanide' for the only ion of ``mol``: a primary amine anion R-[NH-] whose carbon group R has no amine
    name for the suffix seam to ionise ('cyanoazanide' for ``N#C[NH-]``, the cyanamide anion, whose neutral is the amide
    'cyanamide'). (the Blue Book): the anionic centre of a characteristic group that is no amine "named on
    the basis of the corresponding anionic parent hydrides", 'acetylazanide (PIN)' (:41079). It is asked after the router
    and its amine anion emitters declined, so the amine anion that has its own name keeps it; the name is no PIN
    (``_azanide_of_substituent``). '' on any decline."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1:
        return ''
    anion = charged[0]
    heavy = [nb for nb in anion.GetNeighbors() if nb.GetSymbol() != 'H']
    if (anion.GetSymbol() != 'N' or anion.IsInRing() or anion.GetTotalNumHs() != 1 or len(heavy) != 1
            or heavy[0].GetSymbol() != 'C' or any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms())):
        return ''
    from .charged_router import _azanide_of_substituent
    return _azanide_of_substituent(mol, anion.GetIdx())
