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
    that is no carbon, or the anions of one carbon-free centre (``one_centre_chalcogenides``); '' on any decline. The first
    of ``hetero_group_chalcogenide_spellings``."""
    spellings = hetero_group_chalcogenide_spellings(mol, style)
    return spellings[0] if spellings else ''


def hetero_group_chalcogenide_spellings(mol, style: str = 'pin') -> list:
    """Every spelling of the anion that reads back to ``mol`` itself, the preferred first: for a carbon-free oxoacid group
    the retained prefix ('sulfinooxidanide' for HSO3-: 'sulfino', "Retained traditional prefixes",
    the Blue Book, ":17971 sulfino (preselected prefix)"), then the group named by the substituent namers
    ('(1-hydroxy-2-oxa-1-thiaeth-1-en-1-yl)oxidanide', the spelling these anions shipped with), then the anions of one
    carbon-free centre (``one_centre_chalcogenides``). A caller that offers spellings (``adducts._component_spellings_once``)
    offers them in this order, so a mixture whose join the first one does not repair keeps the next one."""
    out = []
    for name in (_single_anion_spellings(mol) or one_centre_chalcogenides(mol)):
        if name and name not in out:
            out.append(name)
    from ..metrics.provenance import record_non_pin_fragment
    for name in out:
        record_non_pin_fragment(name)
    return out


def _single_anion_spellings(mol) -> list:
    """The spellings of a single chalcogen anion on a heteroatom group (the module docstring), each read back;  when
    ``mol`` is not of that shape."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return []
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if len(charged) != 1 or charged[0].GetFormalCharge() != -1:
        return []
    anion = charged[0]
    parent = _PARENT.get(anion.GetSymbol())
    heavy = [nb for nb in anion.GetNeighbors() if nb.GetSymbol() != 'H']
    if (parent is None or anion.IsInRing() or anion.GetTotalNumHs() or anion.GetNumRadicalElectrons()
            or len(heavy) != 1 or heavy[0].GetSymbol() == 'C'
            or any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms())):
        return []
    group = heavy[0]
    from .ions import _has_metal
    if group.GetSymbol() == 'H' or _has_metal(mol):
        return []   # a metal or a heavy element (the one predicate of ``ions``): the salt and complex producers' business
    atoms = _group_atoms(mol, anion.GetIdx(), group.GetIdx())
    if not atoms:
        return []
    prefixes = []
    if not any(a.GetSymbol() == 'C' for a in mol.GetAtoms()):
        # a carbon-free P/S oxoacid group: its retained prefix ('sulfino', 'sulfo'), asked for carbon-free groups only, so
        # the organic anions this producer names (the late rescue of ``charged_fallback``) keep their spelling
        from .oxoacid_group_prefix import oxoacid_group_prefix
        try:
            prefixes.append(oxoacid_group_prefix(mol, atoms, group.GetIdx()))
        except Exception:   # noqa: BLE001 -- a prefix that fails is no spelling; the next one is asked
            pass
    prefixes.append(_group_prefix(mol, atoms, group.GetIdx(), anion.GetIdx()))
    from ..assembly.prefix_marks import mononuclear_substituent_block
    from .charged_router import _charge_sites_rt_ok
    out = []
    for prefix in prefixes:
        block = mononuclear_substituent_block([prefix], max_count=8) if prefix else None   #,:7272
        name = f"{block}{parent}" if block else ''
        if name and name not in out and _charge_sites_rt_ok(name, mol):
            out.append(name)
    return out


#: the parent hydride of each centre ``one_centre_chalcogenides`` names "Mononuclear parent hydrides", Table 2.1)
_CENTRE_HYDRIDE = {'N': 'azane', 'P': 'phosphane', 'As': 'arsane', 'S': 'sulfane', 'Se': 'selane', 'B': 'borane',
                   'Cl': 'chlorane', 'Br': 'bromane', 'I': 'iodane'}
#: the terminal ligands a centre may carry beside its anions: the oxo and amino groups and the halogens
_HALO = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
#: the suffix of a group with 1-4 free valences "Substituent groups": 'yl', 'diyl', 'triyl', 'tetrayl')
_FREE_VALENCE = {1: 'yl', 2: 'diyl', 3: 'triyl', 4: 'tetrayl'}
_COMPLEX_MULTIPLIER = {2: 'bis', 3: 'tris', 4: 'tetrakis'}
#: the halogen acid groups by their oxo count: "Substituent groups derived from halogen acids"
#: (the Blue Book): "OCl– chlorosyl (preselected prefix)... O2Cl– chloryl (preselected prefix)... O3Cl– perchloryl
#: (preselected prefix)... The corresponding Br, F and I groups are named in a similar manner."
_HALOGEN_ACID_GROUP = {1: '{x}osyl', 2: '{x}yl', 3: 'per{x}yl'}
_HALOGEN_ROOT = {'Cl': 'chlor', 'Br': 'brom', 'I': 'iod'}


def one_centre_chalcogenides(mol) -> list:
    """The spellings of the anions of one carbon-free centre Z named on the anionic parent hydride of the chalcogen, the centre a prefix:
    'oxo-λ4-sulfanediylbis(oxidanide)' (sulfite), 'phosphanetriyltris(oxidanide)' (phosphite),
    'aminophosphanediylbis(oxidanide)', 'perchloryloxidanide' (perchlorate); [] on any decline.

    P-72.2.2.2.4 "Anionic centers on other characteristic groups" (BlueBookV2.md:41069) names an anionic centre on a group
    "on the basis of the corresponding anionic parent hydrides"; these anions are oxoacid anions, which that rule leaves to
    the acid endings, so the names are valid but no PIN and are recorded so. They are spelled so for the adduct names of
    a mixture, where OPSIN reads the 'sulfite' / 'perchlorate' word as the neutral acid.

    The centre Z (an element of ``_CENTRE_HYDRIDE``, not in a ring) carries k terminal anions of one chalcogen (charge -1,
    no hydrogen) and otherwise only oxo, amino, halogen and hydrogen. RDKit reads a halate as its charge-separated drawing
    ('O=Cl(=O)(=O)[O-]' -> Z+3 beside four O-): a halogen centre of charge +n beside n+1 or more oxide anions is read as
    n oxo groups and the rest anions, the λ drawing that shares its standard InChIKey (a charged centre of another
    element is declined: a nitrate is no 'λ5-azanyl' group). The bonding number of Z is cited as λn
    where it is not the standard one (P-14.1.3, ``lambda_convention.STANDARD_BONDING_NUMBER``). The first spelling that
    OPSIN reads back to ``mol`` itself is offered (``charged_router._charge_sites_rt_ok``), the preselected prefix of the
    group first ('perchloryloxidanide', 'phosphoryltris(oxidanide)'), then the λ drawing of the centre."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return []
    if any(a.GetSymbol() == 'C' or a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return []
    from .ions import _has_metal
    if _has_metal(mol):
        return []
    anions = [a for a in mol.GetAtoms() if a.GetFormalCharge() == -1 and a.GetSymbol() in _PARENT
              and a.GetDegree() == 1 and not a.GetTotalNumHs()]
    centres = {a.GetNeighbors()[0].GetIdx() for a in anions}
    if not anions or len(centres) != 1 or len({a.GetSymbol() for a in anions}) != 1:
        return []
    centre = mol.GetAtomWithIdx(centres.pop())
    plus = centre.GetFormalCharge()
    if centre.GetSymbol() not in _CENTRE_HYDRIDE or centre.IsInRing() or plus < 0:
        return []
    anion_ids = {a.GetIdx() for a in anions}
    if any(a.GetFormalCharge() and a.GetIdx() != centre.GetIdx() and a.GetIdx() not in anion_ids for a in mol.GetAtoms()):
        return []
    if plus and (centre.GetSymbol() not in _HALO or anions[0].GetSymbol() != 'O' or len(anions) < plus + 1):
        return []
    if len(anions) == 1 and not plus:
        return []   # one anion on a neutral centre: the group-prefix path's shape ('chlorooxidanide'), never respelled here
    ligands = ['oxo'] * plus
    for nb in centre.GetNeighbors():
        if nb.GetIdx() in anion_ids:
            continue
        bond = mol.GetBondBetweenAtoms(centre.GetIdx(), nb.GetIdx()).GetBondType()
        if nb.GetDegree() != 1 or nb.GetFormalCharge():
            return []
        if nb.GetSymbol() == 'O' and bond == Chem.BondType.DOUBLE:
            ligands.append('oxo')
        elif nb.GetSymbol() == 'N' and bond == Chem.BondType.SINGLE and nb.GetTotalNumHs() == 2:
            ligands.append('amino')
        elif nb.GetSymbol() in _HALO and bond == Chem.BondType.SINGLE:
            ligands.append(_HALO[nb.GetSymbol()])
        else:
            return []
    free = len(anions) - plus
    bonding = int(round(sum(b.GetBondTypeAsDouble() for b in centre.GetBonds()))) + centre.GetTotalNumHs() + plus
    from .lambda_convention import LAMBDA, STANDARD_BONDING_NUMBER
    if free not in _FREE_VALENCE or (free > 1 and free not in _COMPLEX_MULTIPLIER):
        return []
    lam = f"{LAMBDA}{bonding}-" if bonding != STANDARD_BONDING_NUMBER.get(centre.GetSymbol()) else ''
    hydride = _CENTRE_HYDRIDE[centre.GetSymbol()]
    group = (hydride[:-1] if free == 1 else hydride) + _FREE_VALENCE[free]   # 'sulfanyl', 'sulfanediyl'
    parent = _PARENT[anions[0].GetSymbol()]
    from ..assembly.prefix_marks import mononuclear_substituent_block
    block = mononuclear_substituent_block(ligands, max_count=4) if ligands else ''
    if block is None:
        return []
    candidates = []
    # the preselected prefix of the group first: a halogen acid group (above) or phosphoryl, '–P(O)<',
    # Table 5.1, the Blue Book: "phosphoryl (preselected prefix)"); 'sulfinyl' / 'sulfonyl' are not
    # offered, OPSIN reads 'sulfinylbis(oxidanide)' as no anion inside an adduct name
    if centre.GetSymbol() in _HALOGEN_ROOT and free == 1 and set(ligands) == {'oxo'} and len(ligands) in _HALOGEN_ACID_GROUP:
        candidates.append(_HALOGEN_ACID_GROUP[len(ligands)].format(x=_HALOGEN_ROOT[centre.GetSymbol()]) + parent)
    if centre.GetSymbol() == 'P' and ligands == ['oxo'] and free == 3:
        candidates.append(f"phosphoryl{_COMPLEX_MULTIPLIER[free]}({parent})")
    # the λ drawing of the centre; the first cited ligand has no enclosing marks,:7272)
    prefix = f"{block}{'-' if block and lam else ''}{lam}{group}"
    if free == 1:
        candidates.append(f"({prefix}){parent}" if (block or lam) else f"{prefix}{parent}")
    else:
        candidates.append(f"{prefix}{_COMPLEX_MULTIPLIER[free]}({parent})")
    from .charged_router import _charge_sites_rt_ok
    return [name for name in candidates if _charge_sites_rt_ok(name, mol)]


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
