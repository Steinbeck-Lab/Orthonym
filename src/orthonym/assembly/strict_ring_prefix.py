"""Strict decorated ring-yl prefixes for the PIN tier's promotion re-run (PB lane R).

The default (PIN) tier names a molecule once with the strict producers (the first run). Only when
that run finds no name does ``Orthonym._name_with_pin_promotion`` name it again with the
best-effort ring-substituent producers admitted (``rules.pin_vocabulary.promote_at_pin_tier``),
and a name built that way is labelled ``pin_unverified``: "a name in PIN form that only a breadth
producer built". ``pin_tier_ring_prefix`` puts one producer in front of that promotion: the
existing recursive decoration composer (``substituent_enumerator._recursive_fragment_substituent_name``)
restricted by a policy to the classes whose name is the PIN by construction. A name every piece of
which comes from this producer or from the strict namer is not flagged (``metrics.provenance.
record_breadth_promotion`` is called only by ``promote_at_pin_tier`` when a breadth piece is kept),
so it keeps ``pin_verified``.

Where it runs. Only inside the re-run (``assembly.memo.pin_promotion_var`` set, no breadth context):
a molecule the first run names returns before the re-run, so its name is untouched. In the first run,
and in every best-effort / complete / aromatic-general / strict-twin run, ``pin_tier_ring_prefix``
makes the same ``promote_at_pin_tier`` call as before (it returns None at once outside the re-run).

What it admits (the Blue Book, " NUMBERING":3219, items (b) indicated hydrogen, (c) free valence
(``cyclohex-3-en-1-yl (preferred prefix)``:3268), (e) hydro/ene, (f) prefixes, (g) the prefix cited first):

* ``stem`` a monocycle named by the PIN stem tables (``_monocycle_core_tail``);
* ``book`` a monocycle named by ``rules.monocycle_forms``:8482,:15800);
* ``terminal`` an all-carbon monocycle with ene/yne locants (a ring with heteroatoms would be an
  'a'-replacement spelling and is refused);
* ``poly`` a fused core named by ``rules.fused_forms`` / the retained-name catalogues when a fusion
  name exists:23708), or a von Baeyer core when none does and no bridged fused name is
  preferred to it:23831; no ring multiple bond: (c):3256 before (e):3289,
  ``bicyclo[2.2.2]oct-5-en-2-yl (preferred prefix)``:17293);
* ``azine`` a six-membered azine with ring ketones / N-H / a saturated free-valence atom: added
  hydrogen:24695, ``pyridin-1(2H)-yl (preferred prefix)``) and hydro prefixes at the
  saturated positions:24766).

What it declines (the breadth path then runs as before and the name stays ``pin_unverified``):
a carbohydrate ring:52681), a ring assembly:15542), a fragment that holds an
instance of the molecule's principal characteristic group, a ring skeleton repeated outside the
fragment when the molecule has no principal group (the parent is then chosen by /, not
here), a fragment whose ring system is the parent ring (no functional parent: ``parent_ring_outside``;
an amine: ``amine_parent_ok``,:26247), a ring nitrogen on a urea / carbamate carbon
(``acylated_ring_nitrogen``: ``piperidine-1-carboxamide (PIN)``:32685), a sugar ring of unspecified
configuration, an exocyclic =O/=S on the ring unless a group of a class senior to ketones sits outside the
fragment, Table 4.1:18182-:18189: cyclic esters and amides are class 16), a charged core,
an attachment that is not a single bond, a ring heteroatom of nonstandard bonding number (S(=O)(=O),
P(=O): the lambda convention,, is not written here), an aromatic ring atom with an exocyclic
double bond (the stem tables carry no hydro prefixes: ``2-oxo-1H-imidazol-3-yl`` for ``2-oxo-2,3-dihydro-
1H-imidazol-3-yl``,:24862), a name carrying non-PIN vocabulary.
"""
from __future__ import annotations

import logging
import re
from typing import FrozenSet, List, Optional, Set

from rdkit import Chem

logger = logging.getLogger(__name__)

#: core kinds the composer reports that this producer admits (see the module docstring)
_KINDS: FrozenSet[str] = frozenset({"stem", "book", "terminal", "poly", "azine"})

_NAMESPACE = "strict_ring_prefix"
#: heteroatoms of an exocyclic double bond that make a ring atom a would-be 'one' position
_OXO_ATOMS = ("O", "S", "Se", "Te", "N")


# ---------------------------------------------------------------------------
# entry point (called from substituent_enumerator.name_substituent and from
# substituent_naming._name_substituent_fragment_uncached)
# ---------------------------------------------------------------------------

def pin_tier_ring_prefix(mol, frag_atoms, attach_idx, breadth):
    """The promoted prefix of a ring-bearing fragment the strict cascade declined, or None.

    ``breadth`` is the zero-argument breadth producer the caller has always offered to
    ``promote_at_pin_tier``. Outside the promotion re-run this is exactly that one call (plus a
    second call that records this module's producer as a promotion site, so that
    ``promotion_could_change`` previews it: both return None at once)."""
    from ..rules.pin_vocabulary import at_pin_tier, promote_at_pin_tier
    from .memo import pin_promotion_var
    if not (pin_promotion_var.get() and at_pin_tier()):
        if _KINDS:
            promote_at_pin_tier(lambda: strict_ring_substituent_name(mol, frag_atoms, attach_idx))
        return promote_at_pin_tier(breadth)
    name = _cached_strict_name(mol, frag_atoms, attach_idx)
    if name is not None:
        return name
    return promote_at_pin_tier(breadth)


def general_tier_ring_prefix(mol, frag_atoms, attach_idx, wider):
    """The ring-yl prefix a general tier (``complete``, ``best-effort``) gives a ring the strict cascade declined.

    The default tier names such a ring in its promotion re-run with the strict producer above, and the name is a
    certified PIN ('adamantan-2-yl', the retained 'adamantane'). A general tier called the wider producer
    (``wider``, the call with the composition switch) and labelled the name, which is the same string, as a prefix
    only a general tier builds (``record_general_ring_prefix``): the name came out ``pin_unverified`` where the
    default tier gives ``pin_verified``. The strict producer is asked first; the wider producer, and the label, are for
    a ring it declines."""
    name = _cached_strict_name(mol, frag_atoms, attach_idx)
    if name is None:
        name = wider()
        if name is not None:
            from ..metrics.provenance import record_general_ring_prefix
            record_general_ring_prefix()
    return name


def _cached_strict_name(mol, frag_atoms, attach_idx):
    if not _KINDS:
        return None
    from .memo import cache_or_compute
    from .substituent_enumerator import _substituent_memo_key
    try:
        key = _substituent_memo_key(mol, frag_atoms, attach_idx, False, False)
    except Exception:  # noqa: BLE001 -- a key error is a cache skip
        key = None

    def _compute():
        try:
            return strict_ring_substituent_name(mol, frag_atoms, attach_idx)
        except Exception:  # noqa: BLE001 -- a producer bug must never crash naming
            logger.debug("strict ring prefix failed", exc_info=True)
            return None
    if key is None:
        return _compute()
    return cache_or_compute(_NAMESPACE, key, _compute)


def strict_ring_substituent_name(mol, frag_atoms, attach_idx) -> Optional[str]:
    """The decorated ring-yl name of a ring-rooted fragment when it is the PIN by construction, else None."""
    from . import substituent_enumerator as se
    fset = set(frag_atoms)
    ring_info = mol.GetRingInfo()
    if attach_idx not in fset or ring_info.NumAtomRings(attach_idx) == 0:
        return None
    if not single_attachment(mol, fset, attach_idx):
        return None
    core = se._ring_system_core_atoms(mol, attach_idx, fset, ring_info)
    if not core or attach_idx not in core:
        return None
    if charged_core(mol, core) or carbohydrate_core(mol, core) or unspecified_carbohydrate(mol):
        return None
    from ..rules.ring_assembly_screen import joins_identical_ring_systems
    if joins_identical_ring_systems(mol):
        return None
    pg, pg_matches = principal_group(mol)
    if principal_group_in_fragment(pg_matches, fset, attach_idx):
        return None
    if pg is None:
        classes = functional_group_classes(mol)
        if (classes is None or (skeleton_twin(mol, fset, attach_idx) and not parent_ring_outside(mol, fset))
                or (classes & _PARENT_NAMING and classes <= _SUBSTITUENT_ONLY | _PARENT_NAMING)
                or (classes <= _SUBSTITUENT_ONLY and not parent_ring_outside(mol, fset))):
            return None
    if (not amine_parent_ok(mol, fset, attach_idx, pg, pg_matches) or tied_amine_parent(mol, pg, pg_matches)
            or acylated_ring_nitrogen(mol, fset, attach_idx)):
        return None
    policy = StrictRingPolicy(fset)
    name = se._recursive_fragment_substituent_name(
        mol, fset, attach_idx, allow_mancude=True, policy=policy)
    if name is None:
        return None
    from ..rules.pin_vocabulary import non_pin_vocabulary
    if non_pin_vocabulary(name, label_forms=False) is not None:
        return None
    return name


# ---------------------------------------------------------------------------
# guards (pure functions of the molecule and the fragment)
# ---------------------------------------------------------------------------

def single_attachment(mol, frag_set: Set[int], attach_idx: int) -> bool:
    """True when the attach atom has exactly one neighbour outside the fragment, by a single bond
    (a ``-yl`` prefix; a double bond would be an ``-ylidene``, a second bond a ``-diyl``)."""
    outside = [b for b in mol.GetAtomWithIdx(attach_idx).GetBonds()
               if b.GetOtherAtomIdx(attach_idx) not in frag_set]
    return len(outside) == 1 and outside[0].GetBondTypeAsDouble() == 1.0


def charged_core(mol, core) -> bool:
    """True when a core atom is charged or a radical ium/ide names are not built here)."""
    return any(mol.GetAtomWithIdx(a).GetFormalCharge() != 0
               or mol.GetAtomWithIdx(a).GetNumRadicalElectrons() for a in core)


def _sugar_rings(mol, within=None):
    """The rings that match the pyranose / furanose pattern: a saturated 5- or 6-ring with one ring
    oxygen and at least two ring carbons that carry a single-bonded O or N (inside ``within`` if given)."""
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if (within is not None and not ring_set <= within) or len(ring_set) not in (5, 6):
            continue
        atoms = [mol.GetAtomWithIdx(a) for a in ring]
        if sum(1 for a in atoms if a.GetSymbol() == 'O') != 1:
            continue
        if any(a.GetIsAromatic() or a.GetHybridization() != Chem.HybridizationType.SP3 for a in atoms):
            continue
        carriers = 0
        for a in atoms:
            if a.GetSymbol() != 'C':
                continue
            if any(n.GetIdx() not in ring_set and n.GetSymbol() in ('O', 'N')
                   and mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1.0
                   for n in a.GetNeighbors()):
                carriers += 1
        if carriers >= 2:
            yield ring


def carbohydrate_core(mol, core) -> bool:
    """True when the core holds a pyranose / furanose pattern. "Carbohydrates"
    (the Blue Book): carbohydrate names are the PINs, the systematic oxane name is not (:53767)."""
    return next(_sugar_rings(mol, set(core)), None) is not None


def unspecified_carbohydrate(mol) -> bool:
    """True when the molecule has a pyranose / furanose ring with a stereocentre left unspecified. The
    Blue Book names carbohydrates by their configuration ff.) and no rule read for this
    producer makes the systematic oxane name of a sugar of unknown configuration the PIN, so no prefix of
    such a molecule is promoted (the molecule stays with the first run)."""
    centres = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    return any(centres.get(a) == '?' for ring in _sugar_rings(mol) for a in ring)


def principal_group(mol):
    """``(class, instances)`` of the molecule's principal characteristic group, ``(None, )`` if none."""
    from ..perception.functional_groups import detect_functional_groups
    from ..rules.seniority import get_principal_group
    try:
        pg, matches = get_principal_group(mol, detect_functional_groups(mol))
    except Exception:  # noqa: BLE001
        return None, ()
    return pg, tuple(matches or ())


def principal_group_in_fragment(pg_matches, frag_set: Set[int], attach_idx: int) -> bool:
    """True when an instance of the principal group has an atom in the fragment other than the attach
    atom: the fragment could be the parent, not a substituent. An instance that touches the
    fragment only at the attach atom (the acetyl of ``1-(...cyclohexenyl)ethan-1-one``) does not count."""
    return any((set(m) & frag_set) - {attach_idx} for m in pg_matches)


def _ring_systems(mol) -> List[Set[int]]:
    systems: List[Set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for other in [s for s in systems if s & merged]:
            systems.remove(other)
            merged |= other
        systems.append(merged)
    return systems


def _skeleton(mol, atoms) -> str:
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), canonical=True, isomericSmiles=False)


def skeleton_twin(mol, frag_set: Set[int], attach_idx: int) -> bool:
    """True when the fragment's ring system has the skeleton of a ring system outside the fragment.

    Called when the molecule has no principal group: the parent is then the senior ring or ring system
    ,:19420 ff.) and, for equal skeletons,; the substituent namer does not decide that
    (``4-chloro-7-[(3-chloro-4-nitroquinolin-7-yl)sulfanyl]-3-nitroquinoline`` is not the PIN, BB
    ``3-chloro-7-[(4-chloro-3-nitroquinolin-7-yl)sulfanyl]-4-nitroquinoline``:22088)."""
    systems = _ring_systems(mol)
    mine = [s for s in systems if attach_idx in s and s <= frag_set]
    if not mine:
        return False
    key = _skeleton(mol, mine[0])
    return any(s.isdisjoint(frag_set) and _skeleton(mol, s) == key for s in systems)


def _outside_neighbour(mol, frag_set: Set[int], attach_idx: int):
    return next((b.GetOtherAtom(mol.GetAtomWithIdx(attach_idx))
                 for b in mol.GetAtomWithIdx(attach_idx).GetBonds()
                 if b.GetOtherAtomIdx(attach_idx) not in frag_set), None)


#: functional-group classes that are only ever cited as prefixes (or as the ring/chain they sit on): a
#: molecule with no principal group whose groups are all of these has no functional parent, so its parent
#: is the senior ring system (urea, guanidine, carbamate, hydrazine derivatives... name the parent
#: themselves and are not listed)
_SUBSTITUENT_ONLY: FrozenSet[str] = frozenset({
    "alkene", "alkyne", "ether", "aromatic_ether", "vinyl_ether", "thioether", "selenoether",
    "telluroether", "chloro", "fluoro", "bromo", "iodo", "nitro", "nitroso", "diazo", "azido",
    "disulfide", "peroxide", "sulfone", "sulfoxide", "selenone", "selenoxide", "isocyanide", "cyanate",
    "thiocyanate", "isocyanate", "isothiocyanate"})

#: classes that name the PARENT of a molecule without a principal group, the Blue Book:
#: ``{7-[(anthracen-2-yl)diazenyl]naphthalen-2-yl}(phenyl)diazene (PIN)``:38829; phenylhydrazine (PIN)
#::38535; O-substituted hydroxylamines): a ring-yl prefix of such a molecule is not built here unless a
#: class that names a parent of its own (urea, thiourea, guanidine, carbamate...) is present
_PARENT_NAMING: FrozenSet[str] = frozenset({"azo", "diazenyl", "hydrazine_fg", "aminooxy"})


def functional_group_classes(mol) -> Optional[FrozenSet[str]]:
    """The functional-group classes the molecule has, None when perception fails."""
    from ..perception.functional_groups import detect_functional_groups
    try:
        return frozenset(name for name, matches in detect_functional_groups(mol).items() if matches)
    except Exception:  # noqa: BLE001
        return None


def parent_ring_outside(mol, frag_set: Set[int]) -> bool:
    """For a molecule with no functional parent: True when a ring system outside the fragment is strictly
    senior, "... (a) to (g)", the Blue Book ff.) to every ring system the
    fragment holds. The parent is then the senior ring system of the molecule: ring before
    chain), so a fragment that is senior or tied cannot be a prefix of it; the ring-system score is the
    one the namer's own parent choice uses (``rules.ring_selection.ring_system_score``)."""
    from ..rules.ring_selection import ring_system_score
    systems = _ring_systems(mol)
    inside = [ring_system_score(mol, s) for s in systems if s & frag_set]
    outside = [ring_system_score(mol, s) for s in systems if not s & frag_set]
    return bool(inside) and bool(outside) and min(outside) < min(inside)


def amine_parent_ok(mol, frag_set: Set[int], attach_idx: int, pg, pg_matches) -> bool:
    """When the fragment hangs on the nitrogen of the principal amine, True only when another ring system
    bonded to that nitrogen is strictly senior to the fragment's. "Selection of the principal
    chain or senior ring system in secondary and tertiary amines" (the Blue Book): the amine is
    named on "a principal chain and a senior ring system (see ", the ring before the chain, and
     ranks the rings (``N-(furan-2-yl)-1H-pyrrol-2-amine``). A fragment that is the senior ring
    is the parent, not a prefix; the name that cites it as a prefix names the junior ring as the parent
    (``N-(benzimidazol-5-yl)-N,1-dimethylpiperidin-4-amine`` for a benzimidazole beside a piperidine)."""
    nitrogen = _outside_neighbour(mol, frag_set, attach_idx)
    if (nitrogen is None or nitrogen.GetSymbol() != 'N' or not pg or 'amine' not in pg
            or not any(nitrogen.GetIdx() in m for m in pg_matches)):
        return True
    from ..rules.ring_selection import ring_system_score
    systems = _ring_systems(mol)
    mine = next((s for s in systems if attach_idx in s), None)
    if mine is None:
        return True
    bonded = {n.GetIdx() for n in nitrogen.GetNeighbors()}
    others = [ring_system_score(mol, s) for s in systems if s is not mine and s & bonded]
    return bool(others) and min(others) < ring_system_score(mol, mine)


def tied_amine_parent(mol, pg, pg_matches) -> bool:
    """True when two ring systems bonded to the principal amine nitrogen tie on the ring-system score:
    the parent is then chosen by "the senior parent structure that has the lower locant or
    set of locants for substituents cited as prefixes", the Blue Book), which the re-run's ring
    selection does not compare ('...-N-(4-methoxypyridin-3-yl)pyridin-3-amine' with locants {N,N,6} where
    the other pyridine gives {N,N,4}); such a molecule keeps the first run's result."""
    if not pg or 'amine' not in pg:
        return False
    from ..rules.ring_selection import ring_system_score
    systems = _ring_systems(mol)
    for nitrogen in {a for m in pg_matches for a in m if mol.GetAtomWithIdx(a).GetSymbol() == 'N'}:
        bonded = {n.GetIdx() for n in mol.GetAtomWithIdx(nitrogen).GetNeighbors()}
        scores = sorted(ring_system_score(mol, s) for s in systems if s & bonded)
        if len(scores) > 1 and scores[0] == scores[1]:
            return True
    return False


_INDICATED_OR_ADDED_H = re.compile(r"\d+[a-z]?H(?:,\d+[a-z]?H)*-|\(\d+[a-z]?H(?:,\d+[a-z]?H)*\)")


def n_yl_without_added_hydrogen(mol, attach_idx: int, core_tail: str) -> bool:
    """True when the free valence sits on a ring nitrogen and the tail is a hydro form that cites neither
    indicated hydrogen nor added hydrogen: a ring nitrogen with no hydrogen in the mancude parent takes
    'added indicated hydrogen', the Blue Book "When no hydrogen atoms are present... it is
    necessary to use 'added indicated hydrogen'";,:24687 "preferred over the use of nondetachable
    hydro prefixes... for preferred IUPAC names"): '3,4-dihydroquinolin-1(2H)-yl (PIN)' (:7467),
    'hexahydropyrrolo[1,2-a]pyrazin-2(1H)-yl', not '1,2,3,4-tetrahydroquinolin-1-yl' or
    'octahydropyrrolo[1,2-a]pyrazin-2-yl'. ('2,3-dihydro-1H-isoindol-2-yl (preferred prefix)',:17380, cites
    its indicated hydrogen.)"""
    return (mol.GetAtomWithIdx(attach_idx).GetSymbol() == 'N' and 'hydro' in core_tail
            and _INDICATED_OR_ADDED_H.search(core_tail) is None)


def acylated_ring_nitrogen(mol, frag_set: Set[int], attach_idx: int) -> bool:
    """True when the attach atom is a ring nitrogen bonded to a carbonyl-type carbon that carries a
    further heteroatom (a urea, carbamate or thiocarbamate carbon). The ring is then the parent of a
    carboxamide or carboxylate (``piperidine-1-carboxamide (PIN)``, the Blue Book), not a prefix
    of ``formamide``."""
    atom = mol.GetAtomWithIdx(attach_idx)
    carbon = _outside_neighbour(mol, frag_set, attach_idx)
    if atom.GetSymbol() != 'N' or carbon is None or carbon.GetSymbol() != 'C':
        return False
    others = [(b.GetOtherAtom(carbon), b.GetBondTypeAsDouble()) for b in carbon.GetBonds()
              if b.GetOtherAtomIdx(carbon.GetIdx()) != attach_idx]
    return (any(order >= 2.0 and n.GetSymbol() in ('O', 'S', 'N') for n, order in others)
            and any(order == 1.0 and n.GetSymbol() in ('O', 'S', 'N') for n, order in others))


_STANDARD_VALENCE = {'N': 3, 'O': 2, 'S': 2, 'Se': 2, 'Te': 2, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,
                     'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4, 'B': 3, 'C': 4}


def hypervalent_core(mol, core) -> bool:
    """True when a ring atom has a bonding number other than its standard one (a ring sulfone, a
    phosphine oxide): the name needs the lambda convention, the Blue Book)."""
    for a in core:
        atom = mol.GetAtomWithIdx(a)
        standard = _STANDARD_VALENCE.get(atom.GetSymbol())
        if standard is None or atom.GetTotalValence() != standard:
            return True
    return False


def aromatic_with_exocyclic_double_bond(mol, core) -> bool:
    """True when an aromatic ring atom carries an exocyclic double bond (a ring ketone or thione of
    an azole): its name needs hydro prefixes and indicated hydrogen, which the stem tables do not carry."""
    core = set(core)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if b.GetBondTypeAsDouble() >= 2.0 and (i in core) != (j in core):
            ring_atom = mol.GetAtomWithIdx(i if i in core else j)
            if ring_atom.GetIsAromatic():
                return True
    return False


def ring_has_multiple_bond(mol, core) -> bool:
    return any((b.GetBondTypeAsDouble() > 1.0 or b.GetIsAromatic())
               and b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core for b in mol.GetBonds())


def ortho_fused_pair(mol, core) -> bool:
    """True when two rings of the core, each of five or more members, share exactly one bond: a bridged
    fused name exists and is preferred to the von Baeyer name, the Blue Book;
    ``hexadecahydro-1H-8,12-methanobenzo[13]annulene (PIN)`` against ``tricyclo[12.3.1.0^5,10]octadecane``,:23875)."""
    info = mol.GetRingInfo()
    atom_rings, bond_rings = info.AtomRings(), info.BondRings()
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            a, b = set(atom_rings[i]), set(atom_rings[j])
            if (a <= set(core) and b <= set(core) and len(a) >= 5 and len(b) >= 5
                    and len(set(bond_rings[i]) & set(bond_rings[j])) == 1):
                return True
    return False


def cage_core_ok(mol, core) -> bool:
    """A von Baeyer core is admitted only with no ring multiple bond and no bridged fused name."""
    return not ring_has_multiple_bond(mol, core) and not ortho_fused_pair(mol, core)


def oxo_class_ok(mol, frag_set: Set[int], core) -> bool:
    """ (the Blue Book;:18182 class 9 and:18184 class 11 name cyclic esters and amides
    as heterocycles 'see 16',:18189 class 16 ketones): a ring-yl with an exocyclic =O/=S/=N on a ring atom
    is a prefix ``oxo`` only when a group of a class senior to class 16 sits outside the fragment (an acyclic
    acid, carboxylate, ester, amide, nitrile, aldehyde). Otherwise the ring ketone competes for the parent
    and decides, which this producer does not."""
    ring_info = mol.GetRingInfo()
    if not any(b.GetBondTypeAsDouble() >= 2.0
               and (b.GetBeginAtomIdx() in core) != (b.GetEndAtomIdx() in core)
               and mol.GetAtomWithIdx(b.GetEndAtomIdx() if b.GetBeginAtomIdx() in core
                                      else b.GetBeginAtomIdx()).GetSymbol() in _OXO_ATOMS
               for b in mol.GetBonds() if b.GetBeginAtomIdx() in core or b.GetEndAtomIdx() in core):
        return True
    from ..perception.functional_groups import detect_functional_groups
    from ..rules.seniority import compare_seniority
    for cls, matches in detect_functional_groups(mol).items():
        cls_rank = 'carboxylic_acid' if cls == 'carboxylate' else cls
        try:
            if compare_seniority(cls_rank, 'ketone') >= 0:
                continue
        except Exception:  # noqa: BLE001
            continue
        for m in matches:
            if set(m) & frag_set:
                continue
            carbonyls = [a for a in m if mol.GetAtomWithIdx(a).GetSymbol() == 'C' and any(
                n.GetIdx() in m and n.GetSymbol() in ('O', 'S', 'N')
                and mol.GetBondBetweenAtoms(a, n.GetIdx()).GetBondTypeAsDouble() >= 2.0
                for n in mol.GetAtomWithIdx(a).GetNeighbors())]
            if carbonyls and all(ring_info.NumAtomRings(a) > 0 for a in carbonyls):
                continue                      # a lactone / lactam / cyclic imide: class 16
            return True
    return False


def poly_branch(mol, core, attach_idx, carriers, core_tail) -> Optional[str]:
    """Which polycyclic core numberer produced ``core_tail`` ('pah', 'fhet', 'fform', 'cage'), or None.
    The composer reports only the winning tail; the numberers are tried in its order."""
    from ..rules import ring_substituents as rs
    for label, numberer in (("pah", rs._pah_core_numbering),
                            ("fhet", rs._fused_heterocycle_core_numbering),
                            ("fform", rs._fused_form_core_numbering),
                            ("cage", rs._cage_core_numbering)):
        try:
            got = numberer(mol, tuple(sorted(core)), attach_idx, list(carriers))
        except Exception:  # noqa: BLE001
            got = None
        if got is not None and got[1] == core_tail:
            return label
    return None


# ---------------------------------------------------------------------------
# the policy handed to the composer
# ---------------------------------------------------------------------------

class StrictRingPolicy:
    """Admission and numbering rules of ``_recursive_fragment_substituent_name`` for this producer."""

    def __init__(self, frag_set: Set[int]):
        self.frag_set = frag_set

    def admit_core(self, kind, mol, core, attach_idx, carriers, core_tail) -> bool:
        if kind not in _KINDS:
            return False
        hetero = any(mol.GetAtomWithIdx(a).GetSymbol() != 'C' for a in core)
        if kind == 'terminal' and hetero:
            return False
        if hypervalent_core(mol, core) or n_yl_without_added_hydrogen(mol, attach_idx, core_tail):
            return False
        if kind == 'stem' and aromatic_with_exocyclic_double_bond(mol, core):
            return False
        if kind == 'poly':
            from ..rules.fused_forms import is_fusable_system
            branch = poly_branch(mol, core, attach_idx, carriers, core_tail)
            fusable = is_fusable_system(mol, sorted(core))
            if branch == 'cage':
                # a von Baeyer core only where no fusion name exists,:23708) and no
                # bridged fused name is preferred,:23831); no ring multiple bond
                if fusable or not cage_core_ok(mol, core):
                    return False
            elif branch not in ('pah', 'fhet', 'fform') or not fusable:
                return False
            # the retained-name catalogues carry no hydro prefixes: a ring ketone of a mancude core
            # is named by the fusion form (indicated hydrogen, hydro prefixes), not by them
            elif branch != 'fform' and aromatic_with_exocyclic_double_bond(mol, core):
                return False
        return oxo_class_ok(mol, self.frag_set, core)

    def renumber(self, mol, core, attach_idx, pos, named):
        """``pos`` resolved by (g) (the Blue Book, "lowest locants for the substituent cited
        first as a prefix in the name") among the numberings that tie on (a)-(f), or None when the tie
        cannot be resolved by it (the descriptors of a stereocentre would take their locants from the
        input atom order)."""
        from .ring_prefix_numbering import g_winners, stereo_atoms
        winners = g_winners(mol, core, attach_idx, pos, named)
        if len(winners) <= 1:
            return winners[0]
        stereo = stereo_atoms(mol, core)
        if len({tuple(numbering[a] for a in stereo) for numbering in winners}) > 1:
            return None
        return winners[0]
