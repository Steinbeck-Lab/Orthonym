"""One numbering for a ring cation whose neutral ring is a hydro derivative of a mancude
heteromonocycle ('2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium').

``rules.ions._emit_ring_cumulative_suffix`` names the neutral ring (its own numbering: hydro
locants lowest, '2-methyl-1,4,5,6-tetrahydropyrimidine') and cites the '-ium' from the numbering
of the charged ring ('-1-ium', or '-3-ium' once the read-back has moved it): two numberings in
one name. For a ring whose parent has no indicated hydrogen they disagree, and the name is
either another species or withdrawn by the check of the default tier ("suffix/free
valence [3] > [1]").

The numbering is one authority here, the one that names the neutral hydro ring
(``heterocycles._mancude_hydro_select``) with the ring atom of the '-ium' as the suffix position.
 "General rule for systematically naming cationic centers in parent hydrides"
(the Blue Book),:41368: a cation "derived formally by adding one or more hydrons to any
position of a neutral parent hydride... or whose degree of hydrogenation has been modified (see
 is named by replacing the final letter 'e' of the parent hydride name, if any, by the suffix
'ium'". The '-ium' is a suffix of the name, so "NUMBERING" (:3219) ranks its locant by
(c) "principal characteristic groups and free valences (suffixes)" (:3256), after (b) "indicated
hydrogen for unsubstituted compounds" (:3246) and before (e) "(i) low locants are given to
hydro/dehydro prefixes" (:3288-:3289) and (f) "detachable alphabetized prefixes, all considered
together" (:3301). A hydro prefix is detachable but not alphabetized and stands directly before
the parent hydride (e),:3288): '2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium (PIN)'
,:41447).

A zwitterion whose cationic ring atom and carboxylate share that ring (ectoine) is named by the
same numbering with the anionic suffix added, ``hydro_ring_zwitterion_name``.
"Zwitterionic compounds with at least one ionic center on a characteristic group"
(the Blue Book),:42447: "Zwitterionic compounds with at least one ionic center on a
characteristic group may be named by adding the appropriate ionic suffix to the name of the ionic
parent hydride. In names, cationic suffixes are cited before anionic suffixes. For assignment of
lower locants, ionic centers on skeletal atoms of the parent hydride are preferred to the locants
for positions of attachment of characteristic groups denoted by ionic suffixes"; the example is
'1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate (PIN)' (:42456). The 'ium' and the 'carboxylate'
are suffixes of one name, so (c) numbers them together, the 'ium' first, before the hydro
prefixes (e). A ring nitrogen that holds no hydron to remove (``N+(CH3)=C``) is the parent's atom plus a
hydron, then substituted above, example '1-methylpyridin-1-ium (PIN)',:41394): its
substituent is a prefix at the locant of the 'ium' (``hydron_site_parent``). The descriptors of a ring name
that carries its own numbering are cited by the ring's producer (``ring_stereo_prefix``).

The same ring cation as the cationic PREFIX of a zwitterion ('pyridin-1-ium-1-yl', is ``ring_cation_prefix``.
"""
import re

from rdkit import Chem


def hydro_ring_cation_name(mol, ring_mol, ring_system, center_idx: int) -> str:
    """The name of the ring cation of ``mol`` centred on ``center_idx``, in one numbering, or ''
    when the ring is not a partially saturated mancude heteromonocycle of this kind.

    ``ring_mol`` is ``mol`` with the centre neutralized, index-preserving (the neutral ring the
    '-ium' is added to). The numbering is that of the hydro name of ``ring_mol`` with the centre as
    the suffix atom: heteroatoms, indicated hydrogen (b), the '-ium' (c), the hydro prefixes (e),
    the substituent prefixes (f). Declines ('') for a ring that is aromatic, saturated, fused, of a
    size the hydro namer does not number, a centre that is a carbon or is itself the indicated-
    hydrogen position (the parent's own ``1H`` is then consumed by the '-ium', which the caller's
    rule treats), a second charge, a guanidine carbon (three nitrogen neighbours) or a substituent
    that cannot be named; the caller's producer then stands, as it did before this module."""
    try:
        return _hydro_ring_cation_name(mol, ring_mol, ring_system, center_idx)
    except (RuntimeError, ValueError, KeyError, IndexError, TypeError, RecursionError):
        return ''


def hydro_ring_zwitterion_name(mol, ring_system, cation_idx: int, anion_idx: int, anchor) -> str:
    """The name of the zwitterion ``mol`` whose ring cation (``cation_idx``) and carboxylate
    (``anion_idx``, its carboxyl carbon on a ring atom, ``anchor`` = (carboxyl carbon, ring atom
    it hangs on): ``rules.ions._carboxyl_ring_anchor``) share one partially saturated mancude
    heteromonocycle, in one numbering: '-<locant>-ium-<locant>-carboxylate' after the hydro
    prefixes ('2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-carboxylate'). A cation that holds no
    hydron (a quaternary ring nitrogen, ``N+(CH3)=C``) is the neutral parent plus a hydron, then
    substituted (``hydron_site_parent``): '1,2-dimethyl-3,4,5,6-tetrahydropyrimidin-1-ium-4-
    carboxylate'. '' where:func:`hydro_ring_cation_name` declines; the caller's producer then
    stands."""
    try:
        ring_mol = neutralized_cation(mol, cation_idx)
        substituents = None
        if ring_mol is None:
            ring_mol, substituents = hydron_site_parent(mol, ring_system, cation_idx), mol
            if ring_mol is None:
                return ''
        carboxyl_c, ring_attach = anchor
        return _hydro_ring_cation_name(mol, ring_mol, ring_system, cation_idx,
                                       (anion_idx, carboxyl_c, ring_attach), substituents)
    except (RuntimeError, ValueError, KeyError, IndexError, TypeError, RecursionError):
        return ''


def ring_stereo_prefix(mol, locants):
    """The front block of descriptors of the ring atoms in ``locants`` (atom index -> locant of the name's
    numbering): '(2S,4R)-', or '' for a ring that has none. "NAMING OF STEREOISOMERS"
    (the Blue Book),:44643: "They are placed at the front of the complete name when related to the
    parent structure". A quaternary ring nitrogen ('[N@+]', no hydrogen) is a stereocentre of the name and is
    cited: '(1R,2S)-1-ethyl-1-methylpyrrolidin-1-ium-2-carboxylate' is read back to the input by OPSIN. None
    where the stereogenic ring nitrogen carries a hydrogen (a specified '[N@H+]'): OPSIN reads no descriptor
    there ('(1R,2R)-1-methylazetidin-1-ium-2-carboxylate' does not parse), so the stereo-aware read-back that
    every name faces could not confirm it, and the caller declines."""
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    descriptors = collect_stereodescriptors(mol, locants)
    for i in locants:
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() == 7 and a.HasProp('_CIPCode') and a.GetTotalNumHs(includeNeighbors=True):
            return None
    return format_stereodescriptor_string(descriptors)


def ring_numbering_tie_key(mol, tokens_at):
    """The key that orders two numberings of one ring name that tie on every earlier tier of the numbering
    cascade (heteroatoms, the 'ium', the suffix, the set of substituent locants), as ``key(locants)``.
     "NUMBERING" (the Blue Book): (g) "lowest locants for the substituent cited first as a prefix
    in the name" (:3307; '1-methyl-4-nitronaphthalene (PIN)'), then (j) "When there is a choice for lower
    locants related to the presence of stereogenic centers or stereoisomers, the lower locant is assigned to
    CIP stereodescriptors Z, R, M, and r... that are preferred to E, S, P, and s" (:3346; '(2R,4S)-2,4-
    difluoropentane (PIN)',:3383): the descriptors of the ring, then those inside the prefixes at the locant
    of their attachment. ``tokens_at`` returns ``{prefix token: [ring atom,...]}`` (None: a prefix cannot
    be named, the key is then empty and the caller declines). Two numberings equal under this key give the
    same string, so the canonical-rank tie-break after it decides no name."""
    from ..assembly.naming_utils import alpha_sort_key, cip_descriptor_rank_key, cip_locant_rank_key
    from .stereochemistry import collect_stereodescriptors
    cache = []

    def key(locants):
        if not cache:
            cache.append(tokens_at())
        tokens = cache[0]
        if tokens is None:
            return ()
        placed = [(token, locants[a]) for token, atoms in tokens.items() for a in atoms]
        return (sorted((alpha_sort_key(token), loc) for token, loc in placed),
                cip_locant_rank_key(collect_stereodescriptors(mol, locants)),
                sorted((loc, cip_descriptor_rank_key(token)) for token, loc in placed))
    return key


def ring_cation_locants(mol, ring_system, centers, numbered=None, to_mol=None, subst_atoms=None):
    """``ions._charged_ring_locants`` for a ring cation whose substituents ``ions._p74_ring_substituent_prefix`` cites,
    with ``ring_numbering_tie_key`` after the criteria it ranks: a numbering that the heteroatoms, the cation and the
    locants of the prefixes leave open is decided by (g) and (j), not by the order of the atoms of the input.
    ``numbered`` is a bare copy of the ring that is numbered instead of ``mol`` and ``to_mol[k]`` the atom of ``mol``
    that is atom ``k`` of it (the numbering of ``emit_ring_poly_cation_ium``)."""
    from .ions import _charged_ring_locants, _p74_ring_substituent_groups
    tie = ring_numbering_tie_key(mol, lambda: _p74_ring_substituent_groups(
        mol, ring_system, {a: a for a in ring_system}, set()))
    if to_mol is None:
        return _charged_ring_locants(mol, ring_system, centers, tie_key=tie)
    return _charged_ring_locants(numbered, list(range(len(to_mol))), centers, subst_atoms=subst_atoms,
                                 tie_key=lambda loc: tie({to_mol[k]: v for k, v in loc.items()}))


def neutralized_cation(mol, cation_idx: int):
    """``mol`` with the cation at ``cation_idx`` neutral, index-preserving: the neutral ring that
    the '-ium' is added to: the cation is the neutral parent plus a hydron). None for
    a cation with no hydrogen to remove (a quaternary ring nitrogen) or one that does not
    sanitize. The hydron is not an indicated hydrogen of the parent: a numbering that reads the
    protonated ring atom as the 'pyrrole-type' NH of the parent cites '1H-imidazol-1-ium'."""
    from .ions import _CUMULATIVE_SUFFIX_H_DELTA
    work = Chem.RWMol(mol)
    a = work.GetAtomWithIdx(cation_idx)
    target_h = a.GetTotalNumHs() + _CUMULATIVE_SUFFIX_H_DELTA['ium']
    if target_h < 0:
        return None
    a.SetFormalCharge(0)
    a.SetNumRadicalElectrons(0)
    a.SetNoImplicit(True)
    a.SetNumExplicitHs(target_h)
    ring_mol = work.GetMol()
    try:
        Chem.SanitizeMol(ring_mol)
    except (RuntimeError, ValueError):
        return None
    return ring_mol


def hydron_site_parent(mol, ring_system, cation_idx: int):
    """The neutral parent of a ring cation that holds no hydron to remove, index-preserving: the
    parent hydride plus a hydron, which a substituent then replaces, the Blue Book:
    "a cation derived formally by adding one or more hydrons to any position of a neutral parent
    hydride"; '1-methylpyridin-1-ium (PIN)',:41394: the quaternary ring nitrogen is the parent's atom
    plus a hydron, then substituted). For ``CC1=[N+](C)CCC(C(=O)[O-])N1`` the parent is the ring with the
    ring nitrogen ``N=C`` and no substituent, 3,4,5,6-tetrahydropyrimidine, and the methyl on the
    '-ium' atom is a prefix at the locant of the '-ium'. Only the cation with one exocyclic single
    bond and no hydrogen is read so (a ring nitrogen ``N+(R)=C``); the substituent is cut from the
    parent, not deleted, so the atom indices stay those of ``mol``, and it is cited from ``mol``.
    None for any other cation, or one whose parent does not sanitize."""
    ring = set(ring_system)
    a = mol.GetAtomWithIdx(cation_idx)
    exo = [b for b in a.GetBonds() if b.GetOtherAtomIdx(cation_idx) not in ring]
    if a.GetTotalNumHs() or len(exo) != 1 or exo[0].GetBondType() != Chem.BondType.SINGLE:
        return None
    work = Chem.RWMol(mol)
    work.RemoveBond(cation_idx, exo[0].GetOtherAtomIdx(cation_idx))
    parent_atom = work.GetAtomWithIdx(cation_idx)
    parent_atom.SetFormalCharge(0)
    parent_atom.SetNoImplicit(True)
    parent_atom.SetNumExplicitHs(0)
    parent = work.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except (RuntimeError, ValueError):
        return None
    return parent


def _hydro_ring_cation_name(mol, ring_mol, ring_system, center_idx: int, anionic=None,
                            substituents=None) -> str:
    """The body of both entry points. ``anionic`` is None for a cation, or ``(anion_idx,
    carboxyl_c, ring_attach)`` for a zwitterion whose carboxylate hangs on ``ring_attach``: the
    carboxylate carbon is then a second suffix atom of the numbering, the carboxyl group is not a
    prefix, and the name ends '-<locant>-ium-<locant>-carboxylate'. ``substituents`` is the molecule
    the prefixes are read from where ``ring_mol`` lacks one (the substituent cut from a quaternary
    '-ium' atom, ``hydron_site_parent``); ``ring_mol`` itself when None."""
    from .heterocycles import _mancude_hydro_select, name_partially_saturated_monocyclic_heterocycle
    from .ions import _elide_terminal_e, _join_prefix_stem, _p74_ring_substituent_prefix
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    ring_set = set(ring_system)
    rings = [set(r) for r in ring_mol.GetRingInfo().AtomRings() if ring_set & set(r)]
    if len(rings) != 1 or rings[0] != ring_set:
        return ''                       # fused, bridged or spiro: the fusion numbering is fixed
    if ring_mol.GetAtomWithIdx(center_idx).GetAtomicNum() == 6:
        return ''
    suffix_atoms, skip, anion_atoms = {center_idx}, set(), set()
    if anionic is not None:
        anion_idx, carboxyl_c, ring_attach = anionic
        suffix_atoms.add(carboxyl_c)
        anion_atoms = {anion_idx}
        skip = {carboxyl_c} | {n.GetIdx() for n in ring_mol.GetAtomWithIdx(carboxyl_c).GetNeighbors()
                               if n.GetAtomicNum() == 8}
    if any(a.GetFormalCharge() for a in ring_mol.GetAtoms() if a.GetIdx() not in anion_atoms):
        return ''
    # A ring carbon with three nitrogen neighbours is a guanidine group. Whether its exocyclic
    # nitrogen is an 'amino' prefix of the cation or the guanidine is a parent of its own is not
    # settled by a Blue Book example "In the presence of a characteristic group
    # having seniority over guanidine...",:34266, speaks of the prefix 'carbamimidoylamino'), and
    # the caller's producers keep the cyclic guanidinium as they did before this module.
    if any(ring_mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and sum(
            1 for n in ring_mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 7) == 3
            for i in ring_set):
        return ''
    sel = _mancude_hydro_select(ring_mol, ring_set, suffix_atoms)
    if sel is None:
        return ''
    ordered, loc, ih = sel['ordered'], sel['loc'], sel['ih']
    if center_idx not in ordered or any(ordered[p] == center_idx for p in ih):
        return ''
    a2l = {ordered[p]: loc[p] for p in range(sel['n'])}
    if anionic is not None and not _cation_locant_is_lowest(ring_mol, ring_set, center_idx, a2l):
        return ''
    stem = name_partially_saturated_monocyclic_heterocycle(ring_mol, ring_system, suffix_atoms)
    if not stem or 'unknown' in stem.lower() or not _stem_is_in_numbering(stem, sel):
        return ''
    prefix = _p74_ring_substituent_prefix(ring_mol if substituents is None else substituents,
                                          ring_system, a2l, skip)
    if prefix is None:
        return ''
    # "NAMING OF STEREOISOMERS" (the Blue Book),:44643: "They are placed at the front
    # of the complete name when related to the parent structure": the ring's descriptors, in the
    # locants of the numbering this name carries.
    stereo = format_stereodescriptor_string(collect_stereodescriptors(mol, a2l))
    name = f"{stereo}{_join_prefix_stem(prefix, _elide_terminal_e(stem))}-{a2l[center_idx]}-ium"
    if anionic is not None:
        name += f"-{a2l[ring_attach]}-carboxylate"
    return name


def _cation_locant_is_lowest(ring_mol, ring_set, center_idx: int, a2l) -> bool:
    """ (:42447): "For assignment of lower locants, ionic centers on skeletal atoms of the
    parent hydride are preferred to the locants for positions of attachment of characteristic
    groups denoted by ionic suffixes." The numbering of a zwitterion is selected with both suffix
    atoms as one set (``_mancude_hydro_select``), which ranks the set; the 'ium' comes first, so
    the numbering stands only where it gives the cation the locant it has when the cation is the
    only suffix."""
    from .heterocycles import _mancude_hydro_select
    alone = _mancude_hydro_select(ring_mol, ring_set, {center_idx})
    if alone is None:
        return False
    first = {alone['ordered'][p]: alone['loc'][p] for p in range(alone['n'])}
    return first.get(center_idx) == a2l.get(center_idx)


def _stem_is_in_numbering(stem: str, sel) -> bool:
    """True when the hydro prefixes and the indicated hydrogen that ``stem`` cites are those of the
    selected numbering ``sel`` (``heterocycles._mancude_hydro_select``): the stem and the locant
    of the '-ium' are two readings of one numbering or the name is not built. The stem is written
    by the namer of the neutral hydro ring (its retained parent 'pyrimidine' or 'imidazole',
    , the numbering by the selection that namer's hydro locants follow."""
    loc, ih = sel['loc'], sel['ih']
    hydro = ','.join(str(loc[p]) for p in sorted(sel['sp3_pos'] - ih, key=lambda q: loc[q]))
    lead = f"{hydro}-{sel['prefix']}"
    if not stem.startswith(lead):
        return False
    rest = stem[len(lead):]
    if ih:
        cited = ','.join(f"{loc[p]}H" for p in sorted(ih, key=lambda q: loc[q]))
        return rest.startswith(f"-{cited}-")
    return not re.match(r'-?\d+H', rest)


#: a parent ring and nothing else: no hydro prefix, no fusion or bridge bracket, no lambda
_PLAIN_RING_STEM = re.compile(r'^(?:\d+H-)?(?:\d+(?:,\d+)*-)?[a-z]+$')


def ring_cation_prefix(mol, cation_idx: int, attach_idx: int) -> str:
    """The prefix of a ring nitrogen cation that is bonded to the rest of the molecule at the cationic centre
    itself, as the cationic part of a zwitterion: 'pyridin-1-ium-1-yl', '1H-pyrrol-1-ium-1-ylidene',
    '1,2-thiazol-2-ium-2-yl'. '' where it is not built (no simple monocycle, a hydro or fused parent, an
    exocyclic double bond or a charge elsewhere on the ring side, a substituent with no name).

    ** "Anionic and cationic centers on different parent structures"** (the Blue Book): zwitterions
    "may be named by prefixing the name of the cationic center or the parts of the structure containing the
    cationic centers to the name of the anionic parent structure". ** "CATIONIC PREFIX NAMES"** (:42290),
    method (1), "Method (1) leads to preferred IUPAC names": "all prefix names are formed by adding the suffixes
    'yl', 'ylidene', etc. to the cation name", 'pyridin-1-ium-1-yl (preferred prefix)', 'the suffix 'yl' being
    senior to 'ylidene''. The cation is the parent hydride plus a hydron,:41366): the ring is
    numbered once (the heteroatoms, the 'ium' and its free valence at one atom, the prefixes of the ring in
     order) by the numbering of ``ions._charged_ring_locants``. ``attach_idx`` is the anionic atom the free
    valence goes to; the bond to it decides 'yl' (single) or 'ylidene' (double)."""
    from ..perception.rings import get_ring_systems
    from .ions import (
        _charged_ring_locants,
        _elide_terminal_e,
        _join_prefix_stem,
        _order_ring_cycle,
        _p74_bare_ring_stem,
        _p74_ring_substituent_groups,
        _p74_ring_substituent_prefix,
    )
    cat = mol.GetAtomWithIdx(cation_idx)
    bond = mol.GetBondBetweenAtoms(cation_idx, attach_idx)
    if (bond is None or cat.GetSymbol() != 'N' or cat.GetFormalCharge() != 1 or cat.GetTotalNumHs()
            or cat.GetDegree() != 3 or not cat.IsInRing()
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or bond.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)):
        return ''
    ring_system = next((rs for rs in get_ring_systems(mol, include_spiro=True) if cation_idx in rs), None)
    if ring_system is None or attach_idx in ring_system or _order_ring_cycle(mol, ring_system) is None:
        return ''
    ring = set(ring_system)
    # the anion side: the free valence's atom and everything beyond it, which is no prefix of the ring
    skip, stack = {attach_idx}, [attach_idx]
    while stack:
        for n in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            if n.GetIdx() != cation_idx and n.GetIdx() not in skip:
                skip.add(n.GetIdx())
                stack.append(n.GetIdx())
    if skip & ring:
        return ''
    # a bond that leaves the ring (other than the free valence) is a single bond to a substituent: an exocyclic
    # double bond is a '-ylidene' that the ring-substituent namer would spell as a '-yl'
    if any(b.GetBondType() != Chem.BondType.SINGLE for i in ring for b in mol.GetAtomWithIdx(i).GetBonds()
           if b.GetOtherAtomIdx(i) not in ring and b.GetOtherAtomIdx(i) != attach_idx):
        return ''
    tie = ring_numbering_tie_key(mol, lambda: _p74_ring_substituent_groups(
        mol, ring_system, {a: a for a in ring_system}, skip))
    locants, indicated_h = _charged_ring_locants(mol, ring_system, cation_idx, tie_key=tie)
    if not locants or indicated_h is not None:
        return ''
    prefix = _p74_ring_substituent_prefix(mol, ring_system, locants, skip)
    stem = _p74_bare_ring_stem(mol, ring_system, cation_idx)
    if prefix is None or not stem or not _PLAIN_RING_STEM.match(stem):
        return ''
    where = locants[cation_idx]
    free_valence = 'ylidene' if bond.GetBondType() == Chem.BondType.DOUBLE else 'yl'
    return f"{_join_prefix_stem(prefix, _elide_terminal_e(stem))}-{where}-ium-{where}-{free_valence}"
