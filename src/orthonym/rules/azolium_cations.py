"""Names of ring cations of the azolium kind: the added hydron and the indicated hydrogen of the
parent hydride sit on two ring nitrogens, the Blue Book), and the substituents
replace the hydrogens. The producers of ions._emit_ring_cumulative_suffix name a ring cation
from the numbering of its neutral ring, which cannot place the indicated hydrogen and the '-ium' on
different nitrogens when both carry a substituent, or when both are protonated."""
import re

from rdkit import Chem

from ..errors import is_failure_name
from ..perception import charge_shift
from .ions import (
    _charged_ring_locants,
    _elide_terminal_e,
    _join_prefix_stem,
    _p74_ring_substituent_groups,
    _p74_ring_substituent_prefix,
    _ring_iupac_locants,
    _ring_locant_order_key,
)


def _charge_structures(mol):
    """The structures of ``mol`` that ``azolium_hydron_centres`` reads, or None when none can be had.

    RDKit builds every bond-order arrangement of each group that carries charge or N=X bonds and its C++ call
    cannot be interrupted (a 66-atom azolium took 94 s in one call, and the Python alarm of the naming run waited
    for it), so it is asked only when ``charge_shift.rdkit_flags0_affordable(mol)``. Above that count the
    structures are the placements of the +1 onto the other three-connected nitrogens of its pi system
    (``charge_shift.charge_placements``, a matching of the pi bonds that counts its own work). An undecided
    matching, and an error in either route, give None: RDKit is never asked in the matching's place. The matching
    leaves a fragment with a radical or a bonded metal anywhere in it undecided, so above the count such a cation
    keeps only its drawn centre (``[center_idx]``); every name built from the centres is still read back."""
    if charge_shift.rdkit_flags0_affordable(mol):
        try:
            sup = Chem.ResonanceMolSupplier(mol, 0, 64)
            return [sup[i] for i in range(len(sup))]
        except Exception:
            return None
    try:
        return charge_shift.charge_placements(mol)
    except Exception:
        return None


def azolium_hydron_centres(mol, ring_system, center_idx):
    """The ring nitrogens at which the charge of the azolium ``mol`` can be drawn, ``center_idx``
    first: an aromatic ring N bearing an exocyclic substituent and no hydrogen, with the positive
    charge on it in a resonance structure of ``mol`` and none elsewhere. In 1,3-dimethylimidazolium
    both nitrogens are such centres; the cation is one species and the name cannot depend on which
    nitrogen the drawing charges."""
    structs = _charge_structures(mol)
    if structs is None:
        return [center_idx]
    found = {center_idx}
    for r in structs:
        if r is None:
            continue
        charged = [i for i in ring_system if r.GetAtomWithIdx(i).GetFormalCharge() == 1]
        if len(charged) != 1 or charged[0] in found:
            continue
        a = mol.GetAtomWithIdx(charged[0])
        if (a.GetAtomicNum() == 7 and a.GetIsAromatic() and a.GetTotalNumHs() == 0
                and a.GetDegree() == 3):
            found.add(charged[0])
    return [center_idx] + sorted(found - {center_idx})


def _bare_parent(mol, ring_system, centre):
    """The neutral parent hydride of an azolium whose ring nitrogens all carry a substituent:
    the ring with every substituent replaced by hydrogen, one aromatic ring nitrogen (the
    indicated-hydrogen position) bearing that hydrogen, ``centre`` (the position of the added
    hydron) bearing none. ``(parent_mol, ring_sorted)``, ``ring_sorted[k]`` being the atom of
    ``mol`` that is atom ``k`` of ``parent_mol``; None when it does not sanitize."""
    ring = sorted(set(ring_system))
    work = Chem.RWMol(mol)
    pyrrole_like = {i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                    and mol.GetAtomWithIdx(i).GetIsAromatic()
                    and mol.GetAtomWithIdx(i).GetDegree() == 3}
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring),
                      reverse=True):
        work.RemoveAtom(idx)
    try:
        for k, i in enumerate(ring):
            a = work.GetAtomWithIdx(k)
            a.SetFormalCharge(0)
            a.SetNoImplicit(False)
            if i == centre:
                a.SetNumExplicitHs(0)
            elif i in pyrrole_like:
                a.SetNumExplicitHs(1)
                a.SetNoImplicit(True)
        parent = work.GetMol()
        Chem.SanitizeMol(parent)
    except (RuntimeError, ValueError):
        return None
    return parent, ring


def _azolium_candidate(mol, ring_system, centre, substituted):
    """``(sort key, name)`` of the azolium ``mol`` named on the parent hydride whose added hydron is at
    ``centre`` (``_bare_parent``, the ring with every substituent replaced by hydrogen), the prefixes
    cited in the numbering of that parent; None when the parent has no name, its numbering and the
    indicated hydrogen of its stem name two nitrogens, or a substituent cannot be named. ``substituted``
    are the ring nitrogens that carry a substituent. The key is the choice the rules leave between the
    nitrogens that can be the centre: the locant of the 'ium', the set of the prefix locants (f)),
    the locants of each prefix in alphabetical order ((g))."""
    from ..assembly.naming_utils import alpha_sort_key, prefix_citation_sort_key
    built = _bare_parent(mol, ring_system, centre)
    if built is None:
        return None
    parent, ring_sorted = built
    try:
        from ..namer import Orthonym
        stem = Orthonym(style='pin', _disable_opsin_validity_gate=True
                        ).name(Chem.MolToSmiles(parent))
    except (RecursionError, ValueError, RuntimeError):
        return None
    if not stem or is_failure_name(stem):
        return None
    parent_locants = _ring_iupac_locants(parent)
    if not parent_locants or len(parent_locants) != len(ring_sorted):
        return None
    locants = {ring_sorted[k]: v for k, v in parent_locants.items() if k < len(ring_sorted)}
    if len(locants) != len(ring_sorted) or centre not in locants:
        return None
    # The indicated hydrogen the stem cites is the nitrogen it numbers: a stem that names
    # another position than the numbering used for the prefixes would join two numberings.
    m = re.match(r'^(\d+)H-', stem)
    nh = [i for i in ring_sorted if i != centre and i in substituted]
    if m and (len(nh) != 1 or str(locants[nh[0]]) != m.group(1)):
        return None
    groups = _p74_ring_substituent_groups(mol, ring_system, locants, set())
    prefix = _p74_ring_substituent_prefix(mol, ring_system, locants, set())
    if groups is None or prefix is None:
        return None
    name = (f"{_join_prefix_stem(prefix, _elide_terminal_e(stem))}"
            f"-{locants[centre]}-ium")
    lk = _ring_locant_order_key
    every = sorted((lk(x) for locs in groups.values() for x in locs))
    # tokens of one alphanumerical key ('(1R)-1-chloroethyl', '(1S)-...') in the citation order (R first), so the
    # numbering that gives the prefix cited first the lower locant is the one chosen: (g) then (j)
    by_token = tuple(tuple(sorted(lk(x) for x in groups[t]))
                     for t in sorted(groups, key=lambda t: (alpha_sort_key(t), prefix_citation_sort_key(t))))
    return (lk(locants[centre]), every, by_token, name), name


def azolium_demote_name(mol, ring_system, center_idx):
    """ (heading the Blue Book): "A cation derived formally by adding one or more
    hydrons to any position of a neutral parent hydride... is named by replacing the final
    letter 'e' of the parent hydride name, if any, by the suffix 'ium'"; "1H-imidazol-3-ium (PIN)"
    (:41396). An azolium whose ring nitrogens all carry a substituent (1,3-dimethylimidazolium,
    1-ethyl-3-methylimidazolium, 1,3-dimethylbenzimidazolium) is that cation with the
    substituents replacing the hydrogens: the indicated hydrogen of the parent hydride and the
    added hydron sit on the two nitrogens, and a substituent stands on each.

    The parent hydride is the ring with every substituent replaced by hydrogen, one nitrogen
    bearing the indicated hydrogen (``_bare_parent``); its own numbering puts that
    hydrogen at the lowest locant, (heading:3219) criterion (b) "indicated hydrogen for
    unsubstituted compounds" (:3246), and the '-ium' takes the other nitrogen, criterion (c)
    "principal characteristic groups and free valences (suffixes)" (:3256). Which nitrogen is the
    indicated-hydrogen position is left open by the structure, so each nitrogen that can carry the
    charge is tried and the cation is named once, by the choice the rules leave: the locant set
    of the prefixes (f) "detachable alphabetized prefixes, all considered together in a series of
    increasing numerical order" (:3301), then (g) "lowest locants for the substituent cited first
    as a prefix in the name" (:3307; '1-methyl-4-nitronaphthalene (PIN)',:3318). The drawing therefore
    does not change the name. Every candidate is read back by OPSIN with the hydron on the same
    atoms (``_cation_name_rt_ok``); '' (decline) when none is, or the ring is not an azolium of
    this kind."""
    center = mol.GetAtomWithIdx(center_idx)
    if (center.GetAtomicNum() != 7 or not center.GetIsAromatic() or center.GetFormalCharge() != 1
            or center.GetTotalNumHs() != 0 or center.GetDegree() != 3):
        return ''
    ring = set(ring_system)
    others = [i for i in ring if i != center_idx and mol.GetAtomWithIdx(i).GetAtomicNum() == 7
              and mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetDegree() == 3]
    # Scope: another substituted aromatic nitrogen, no ring nitrogen with hydrogen (the
    # in-place and hydron-site constructions name those) and no other charge.
    if not others:
        return ''
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 and mol.GetAtomWithIdx(i).GetTotalNumHs()
           for i in ring):
        return ''
    if any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() != center_idx):
        return ''
    from .charged_router import _cation_name_rt_ok
    candidates = []
    for centre in azolium_hydron_centres(mol, ring_system, center_idx):
        built = _azolium_candidate(mol, ring_system, centre, others + [center_idx])
        if built is not None:
            candidates.append(built)
    for _key, name in sorted(candidates):
        if _cation_name_rt_ok(name, mol):
            return name
    return ''


def azolium_hydron_site_name(mol, ring_system, cation_idx):
    """The azolium whose added hydron sits on ``cation_idx`` (``[nH+]``) and whose other ring nitrogen
    carries the substituent (``Nc1c[nH+]cn1C``): (heading the Blue Book),:41368, "A cation
    derived formally by adding one or more hydrons to any position of a neutral parent hydride", PIN
    '1H-imidazol-3-ium' (:41396), the substituents replacing hydrogens, so the substituted nitrogen is the
    indicated-hydrogen position of the parent hydride. (:42219): "This is consistent with the
    choice of lowest locants for corresponding neutral compounds (see "; with (b) the indicated
    hydrogen (:3246) and (c) the 'ium' (:3256) at 1 and 3 in both numberings, (f) (:3301) "detachable
    alphabetized prefixes, all considered together in a series of increasing numerical order" gives
    '5-amino-1-methyl-1H-imidazol-3-ium' ({1,5}), not '4-amino-3-methyl-' ({3,4}). The producers of ``ions``
    decline a ring that carries a principal group (amino, carboxy, hydroxy, cyano...) as a suffix, so
    this drawing had no name and the form choice shipped the other drawing's, with the 'ium' on the
    substituted nitrogen (decision LO9). Read back by OPSIN with the hydron on the same atom
    (``_cation_name_rt_ok``); '' (decline) for another ring, a second ring NH or substituted nitrogen, or any
    other charge."""
    cation = mol.GetAtomWithIdx(cation_idx)
    if (cation.GetAtomicNum() != 7 or not cation.GetIsAromatic() or cation.GetFormalCharge() != 1
            or cation.GetTotalNumHs() != 1 or cation.GetDegree() != 2):
        return ''
    others = [mol.GetAtomWithIdx(i) for i in ring_system if i != cation_idx
              and mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)]
    substituted = [a.GetIdx() for a in others if a.GetAtomicNum() == 7 and a.GetIsAromatic()
                   and a.GetDegree() == 3 and not a.GetTotalNumHs()]
    if (len(substituted) != 1 or any(a.GetTotalNumHs() for a in others)
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() != cation_idx)):
        return ''
    built = _azolium_candidate(mol, ring_system, cation_idx, substituted)
    from .charged_router import _cation_name_rt_ok
    return built[1] if built is not None and _cation_name_rt_ok(built[1], mol) else ''


def azolium_zwitterion_name(mol, ring_system, cation_idx, anchor):
    """The zwitterion of an azolium whose added hydron sits on ``cation_idx`` (``[nH+]``), whose
    indicated-hydrogen nitrogen carries a substituent, and whose ring carries a carboxylate
    (``anchor`` = (carboxyl carbon, ring atom it hangs on)), in one numbering:
    "Zwitterionic compounds with at least one ionic center on a characteristic group"
    (the Blue Book),:42447, cationic suffix before anionic, "ionic centers on skeletal atoms
    of the parent hydride are preferred to the locants for positions of attachment of
    characteristic groups denoted by ionic suffixes", over the parent hydride of
    (:41368, '1H-imidazol-3-ium (PIN)',:41396): ``Cn1cc(C(=O)[O-])[nH+]c1C`` is
    '1,2-dimethyl-1H-imidazol-3-ium-4-carboxylate'. The parent is the ring with every substituent
    replaced by hydrogen (``_bare_parent``), so the substituted nitrogen is its indicated-hydrogen
    atom (b),:3246) and the numbering of the parent, which ranks the 'ium' (c) and then the
    carboxylate, is the one the prefixes are cited in. A ring that is not a five-membered monocycle
    with that one substituted nitrogen, or whose stem cites an indicated hydrogen elsewhere than
    the nitrogen the numbering gives it, is declined ('')."""
    ring = set(ring_system)
    cation = mol.GetAtomWithIdx(cation_idx)
    if (len(ring) != 5 or cation.GetAtomicNum() != 7 or not cation.GetIsAromatic()
            or cation.GetFormalCharge() != 1 or cation.GetTotalNumHs() != 1 or cation.GetDegree() != 2):
        return ''
    others = [mol.GetAtomWithIdx(i) for i in ring if i != cation_idx
              and mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)]
    substituted = [a.GetIdx() for a in others if a.GetAtomicNum() == 7 and a.GetIsAromatic()
                   and a.GetDegree() == 3 and not a.GetTotalNumHs()]
    if len(substituted) != 1 or any(a.GetTotalNumHs() for a in others):
        return ''
    built = _bare_parent(mol, ring_system, cation_idx)
    if built is None:
        return ''
    parent, ring_sorted = built
    to_bare = {o: k for k, o in enumerate(ring_sorted)}
    carboxyl_c, attach = anchor
    try:
        from ..namer import Orthonym
        stem = Orthonym(style='pin', _disable_opsin_validity_gate=True
                        ).name(Chem.MolToSmiles(parent))
    except (RecursionError, ValueError, RuntimeError):
        return ''
    indicated = re.match(r'^(\d+)H-', stem or '')
    if not stem or is_failure_name(stem) or not indicated:
        return ''
    bare_subst = {to_bare[i] for i in ring if any(
        n.GetIdx() not in ring for n in mol.GetAtomWithIdx(i).GetNeighbors())}
    base, _ind = _charged_ring_locants(parent, list(range(len(ring_sorted))), to_bare[cation_idx],
                                       anion_attach_idx=to_bare[attach], subst_atoms=bare_subst)
    if not base or len(base) != len(ring_sorted):
        return ''
    locants = {ring_sorted[k]: v for k, v in base.items()}
    # the indicated hydrogen the stem cites is the substituted nitrogen of this numbering
    if str(locants[substituted[0]]) != indicated.group(1):
        return ''
    carboxyl_atoms = {carboxyl_c} | {n.GetIdx() for n in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors()
                                     if n.GetAtomicNum() == 8}
    prefix = _p74_ring_substituent_prefix(mol, ring_system, locants, carboxyl_atoms)
    if prefix is None:
        return ''
    return (f"{_join_prefix_stem(prefix, _elide_terminal_e(stem))}"
            f"-{locants[cation_idx]}-ium-{locants[attach]}-carboxylate")


def ring_has_other_nh_heteroatom(mol, ring_system, center_idx):
    """True if a ring heteroatom other than ``center_idx`` carries hydrogen (the NH of a
    benzimidazolium or a triazolium drawn with both nitrogens protonated). The neutral
    parent then cites an indicated hydrogen ('1H-'), a position of its own numbering, and
    the '-ium' is cited at the centre's locant read from the numbering of the neutral
    ring, which does not know which of two equivalent nitrogens holds that hydrogen: the
    two locants can name one nitrogen twice ('1H-1,3-benzimidazol-1-ium', the NH2+ cation)."""
    return any(i != center_idx and mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)
               and mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1
               for i in ring_system)
