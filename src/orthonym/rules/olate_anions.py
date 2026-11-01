"""The anions of hydroxy-type groups that the suffix of the neutral name cannot spell.

``charged_router.olate_with_senior_neutral_group`` names a mono-anion on a hydroxy or thiol group by the
neutral name of the molecule with that group forced as the principal group. Two anions of the same family are
not reached that way, because their neutral form has no 'ol' suffix to take the ending 'ate':

* the anion of an imidic acid, ``R-C(=NH)-O(-)``: "Anions derived from acids" (the Blue Book)
  replaces "the 'ic acid'... ending of the acid name by 'ate'", and the imidic acids are acids named by the
  suffix 'imidic acid' "Carboximidic acids",:29978: 'propanimidic acid (PIN)',:17574), so the
  anion is '-imidate';
* the anion of a hydroxylamine, ``R2N-O(-)``: "Anions derived from hydroxy compounds" (:41011)
  retains 'aminoxide' for H2N-O(-) and gives '(CH3)2N-O(-) dimethylaminoxide (PIN)' (:41031), the substituents
  of the nitrogen cited as the prefixes of 'aminoxide' ('may be substituted in the same way as the
  corresponding alcohols',,:28180).

Each name is read back to the full InChIKey and to the structure's own charge sites (the key does not tell the
imidate of an amide from its amidate); a decline returns '' and the callers name the anion as before.
"""
from rdkit import Chem

#: the ending of the neutral name that the anion takes the ending 'ate' of
_IMIDIC_ENDING = 'imidic acid'


def _single_oxygen_anion(mol, sites):
    """The oxygen atom of the one anionic centre of a mono-anion (no cation, no radical, net charge -1) that is a
    terminal oxygen, or None."""
    anions = sites.get('anions') or []
    if len(anions) != 1 or sites.get('cations') or Chem.GetFormalCharge(mol) != -1:
        return None
    from ..perception.ions import get_radical_sites
    if get_radical_sites(mol):
        return None
    atom = mol.GetAtomWithIdx(anions[0]['atom_idx'])
    if atom.GetSymbol() != 'O' or atom.GetDegree() != 1 or atom.GetTotalNumHs():
        return None
    return atom


def _name_reads_back(mol, name: str) -> bool:
    """``name`` reads back to the full InChIKey of ``mol`` and to its own charge sites."""
    from .charged_router import _charge_sites_rt_ok, _full_inchikey_rt_ok
    return bool(name) and _full_inchikey_rt_ok(mol, name) and _charge_sites_rt_ok(name, mol)


def imidate_anion_name(mol, sites, neutral_smi: str, style: str) -> str:
    """'-imidate' of the anion of an acyclic imidic acid ``R-C(=NR')-O(-)`` on, or ''.

    The neutral form is named with the imidic acid forced as the principal group (a senior neutral group of the
    molecule is a prefix: the anion is the parent, class 4), and the name must end in 'imidic acid'."""
    oxygen = _single_oxygen_anion(mol, sites)
    if oxygen is None:
        return ''
    carbon = oxygen.GetNeighbors()[0]
    if carbon.GetSymbol() != 'C' or carbon.IsInRing():
        return ''
    if not any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetSymbol() == 'N'
               for b in carbon.GetBonds()):
        return ''
    from .charged_router import _is_malformed_parent, _reenter_forced_at_pin_tier
    try:
        neutral = _reenter_forced_at_pin_tier(neutral_smi, style, 'imidic_acid')
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not neutral or _is_malformed_parent(neutral) or not neutral.endswith(_IMIDIC_ENDING):
        return ''
    name = neutral[:-len(_IMIDIC_ENDING)] + 'imidate'
    return name if _name_reads_back(mol, name) else ''


def aminoxide_anion_name(mol, sites, style: str) -> str:
    """'<groups>aminoxide' of the anion ``R-N(R')-O(-)`` or ``R-NH-O(-)`` of a hydroxylamine whose nitrogen carries one
    or two carbon groups outside any ring and no acyl group: 'dimethylaminoxide (PIN)';,:28180:
    'may be substituted in the same way as the corresponding alcohols'), or ''.

    The two groups are named as substituent prefixes and cited by the rule of a mononuclear centre,
    :7272: the first group has no enclosing marks, the later ones are enclosed, a repeated group takes the
    multiplier). A nitrogen in a ring is named on its ring ('piperidin-1-olate'), a nitrogen with an acyl group is a
    hydroxamic acid anion."""
    oxygen = _single_oxygen_anion(mol, sites)
    if oxygen is None:
        return ''
    nitrogen = oxygen.GetNeighbors()[0]
    if (nitrogen.GetSymbol() != 'N' or nitrogen.GetFormalCharge() != 0
            or nitrogen.GetTotalNumHs() + nitrogen.GetDegree() != 3 or nitrogen.GetDegree() not in (2, 3)
            or nitrogen.IsInRing() or nitrogen.GetIsAromatic()):
        return ''
    from .amine_anion_label import arm_is_acyl
    arms = [n for n in nitrogen.GetNeighbors() if n.GetIdx() != oxygen.GetIdx()]
    if any(a.GetSymbol() != 'C' or arm_is_acyl(a) for a in arms):
        return ''
    n_idx, o_idx = nitrogen.GetIdx(), oxygen.GetIdx()
    rw = Chem.RWMol(mol)
    for a in arms:
        rw.RemoveBond(n_idx, a.GetIdx())
    comps = Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
    home = {atom: i for i, comp in enumerate(comps) for atom in comp}
    if len({home[a.GetIdx()] for a in arms}) != len(arms):
        return ''  # the groups close a ring through the nitrogen: its ring is the parent
    from ..assembly.prefix_marks import mononuclear_substituent_block
    from .charged_router import _stereo_composed
    from .group_prefixes import combinations, group_prefix_candidates
    candidates = []
    for a in arms:
        names = group_prefix_candidates(mol, list(comps[home[a.GetIdx()]]), a.GetIdx(), 'yl', [n_idx],
                                        ligand_first=False)
        if not names:
            return ''
        candidates.append(names)
    for prefixes in combinations(candidates):
        block = mononuclear_substituent_block(list(prefixes), max_count=8)
        if block is None:
            continue
        name = f"{block}aminoxide"
        if _name_reads_back(mol, name):
            return name
        # a group that cites no descriptor of its own double bond: the input's stereo is composed onto the name
        # (the stereo-omission reclaim), which is returned only when it reads back as well
        composed = _stereo_composed(name, mol)
        if composed and _name_reads_back(mol, composed):
            return composed
    return ''


def olate_anion_name(mol, sites, neutral_smi: str, style: str) -> str:
    """The name of a mono-anion on a hydroxy-type group: the hydroxy or thiol group named on the neutral form
    (``charged_router.olate_with_senior_neutral_group``), else the imidate, else the aminoxide; '' when none."""
    from .charged_router import olate_with_senior_neutral_group
    return (olate_with_senior_neutral_group(mol, sites, neutral_smi, style)
            or imidate_anion_name(mol, sites, neutral_smi, style)
            or aminoxide_anion_name(mol, sites, style))


def olate_on_nitrogen_must_read_back(mol, name: str) -> str:
    """``name`` of a mono-anion, or '' when the oxygen of the anion is on a nitrogen and the name does not read back.

    The suffix seam of ``route_charged`` gives the neutral name the ending of the anion, which names the anion only where the
    oxygen of the anion is the 'ol' of that name. An oxygen on a nitrogen ('R2N-O(-)',: 'aminoxide') is no
    'ol' of a neutral name that is built on the carbon groups: for ``CN(CCO)[O-]`` the neutral producer
    (``assembly.handlers.ester_family``) drops the N-hydroxy group of ``CN(O)CCO`` ('2-(methylamino)ethan-1-ol'), and the
    seam turned that into '2-(methylamino)ethan-1-olate', a name of another molecule that only a later gate stopped. The seam's
    name stands for such an anion only when it reads back to the full InChIKey; otherwise the routes after it (the
    aminoxide,:func:`aminoxide_anion_name`) name the anion. Every other anion keeps the seam's name unchecked, as before."""
    if not name or Chem.GetFormalCharge(mol) >= 0:
        return name
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() < 0 and atom.GetSymbol() == 'O'
                and any(n.GetSymbol() == 'N' for n in atom.GetNeighbors())):
            from .charged_router import _full_inchikey_rt_ok
            return name if _full_inchikey_rt_ok(mol, name) else ''
    return name
