"""
Ring assembly detection and naming per IUPAC.

Ring assemblies are molecules composed of two or more identical ring systems
joined by single bonds. Examples: biphenyl, bipyridine, bithiophene.

IUPAC: Ring assemblies use multiplicative prefixes (bi-, ter-, quater-)
before the parent ring name. Benzene assemblies use "phenyl" as base name.

Locant convention: First ring unprimed, second primed ('), third double-primed ('').
Connection locants are separated by commas, multi-connection pairs by colons.

Example outputs:
  - biphenyl SMILES -> "1,1'-biphenyl"
  - 2,2'-bipyridine SMILES -> "2,2'-bipyridine"
  - 4-chlorobiphenyl SMILES -> "4-chloro-1,1'-biphenyl"
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem import rdchem
from ..perception.molcache import bonds_of


# a phase.B: detect indicated-H prefix in a ring stem name like
# "1H-indole" so name_ring_assembly can replicate the descriptor across
# all primed rings ("1H,1'H-2,2'-biindole") instead of leaving it embedded
# as a substring of a single ring name ("2,2'-bi1H-indole").
# Source: 155-internal notes; AUTONOM-followups.md Follow-up 12 placement
# subset; IUPAC.
_INDICATED_H_RE = re.compile(r"^(\d+)H-(.*)$")

# a phase: detect embedded indicated-H descriptors inside
# a ring stem name like "phenanthridin-6(5H)-one" or "acridin-9(10H)-one".
# The placement-replication logic in `_INDICATED_H_RE` only covers the
# leading "<n>H-" prefix form; embedded "(<n>H)" descriptors require a
# different per-ring threading strategy (deferred to a phase Follow-up
# 12's deeper assembly-builder rewrite). Until that rewrite lands, ring
# assemblies whose component stem carries an embedded descriptor fail
# closed (return None) rather than emit the buggy mid-string `bi...`
# token (e.g., the pre-fix `2,2'-biphenanthridin-6(5H)-one` form).
# Source: internal notes; IUPAC.
_INDICATED_H_EMBEDDED_RE = re.compile(r"\(\d+H\)")

# a phase-03 type alias: cascade-step-6 supplier returns int|tuple
# locants. Tuples are reserved for fusion-atom locants like (8, 'a');
# ring assemblies use plain ints because primes are name-format-layer only
# (per internal notes Pattern S-3 and internal notes /).
_Locant = Union[int, Tuple[int, str]]


# Assembly multiplier prefixes (IUPAC
# Source: IUPAC Blue Book Table, verified against OPSIN multipliers.xml
ASSEMBLY_MULTIPLIERS = {
    2: "bi",
    3: "ter",
    4: "quater",
    5: "quinque",
    6: "sexi",
    7: "septi",
    8: "octi",
    9: "novi",
    10: "deci",
    # / IUPAC multiplying affixes: ring assemblies of more than six (and
    # beyond the old deci cap) identical cyclic systems. OPSIN-verified that the
    # explicit-locant assembly with 'undeci' (11) round-trips. Counts above 12
    # keep the.get(count)->None decline (no OPSIN-verifiable affix -> fail closed).
    11: "undeci",
    12: "dodeci",
}


# IUPAC: the multiplied parent-hydride name of a ring assembly is
# enclosed in parentheses "to avoid confusion with von Baeyer names". This is
# required for cycloalkane / cycloalkene monocycles (cyclopropane, cyclohexene,
# cyclopentylidene) and von Baeyer polycyclics (bicyclo[2.2.1]heptane,
# spiro[...]). Mancude rings (phenyl, pyridine, furan, naphthalene, indole) are
# NOT enclosed: 1,1'-biphenyl, 2,2'-bipyridine, 2,3'-bifuran, 1,2'-binaphthalene.
# Worked examples; all PINs OPSIN-RT-verified (/).
_VON_BAEYER_NAME_RE = re.compile(
    r"^cyclo[a-z]+(?:ane|ene|yne|adiene|atriene|ylidene)$"
)


# IUPAC "Indicated hydrogen" (the Blue Book): in a ring assembly the
# indicated hydrogen is cited at the position REQUIRED IN EACH COMPONENT RING
# (per-ring, e.g. "1H,3'H-4,4'-biazepine"), never front-replicated from the
# first ring. An indicated-H atom is a mancude-ring skeletal atom saturated at
# a position that would be sp2 in the mancude parent. The elements below are
# those whose mancude position can hold an indicated hydrogen (they are part of
# the double-bond framework in the parent): C, B, N, Si, P, Ge, As, Sn, Sb, Bi.
# Divalent O/S/Se/Te are always saturated ring linkers and never bear an
# indicated hydrogen, so they are excluded — the same bonding-number >= 3 split
# ``heterocycles._monocycle_indicated_h_prefix`` applies.
_INDICATED_H_ATOM_ELEMENTS = frozenset({5, 6, 7, 14, 15, 32, 33, 50, 51, 83})


def _ring_indicated_h_atoms(mol, ring) -> List[int]:
    """Ring atoms holding a mancude saturated (indicated-hydrogen) position.

    IUPAC (the Blue Book) /: the indicated hydrogen of a component
    ring is required at the ring atom that is saturated where the mancude parent
    would carry a double bond (the ``2H`` of 2H-pyran's CH2, the ``1H`` of
    1H-pyrrole's N-H). Kekulize a COPY so aromatic and explicit-double rings are
    treated alike (hazard: an aromatic CH must NOT be mistaken for an
    indicated-H position — ``GetTotalNumHs>=1`` would over-match every
    aromatic ring CH; the ring-double-bond test below does not). An indicated-H
    atom has NO ring double bond in the kekulized structure and is an element
    that would be sp2 in the mancude parent (``_INDICATED_H_ATOM_ELEMENTS``).

    One refinement handles substitution AT the indicated-H position, which is
    exactly what a ring–ring bond in an assembly is:
      - A genuinely sp3 (non-aromatic) skeletal position KEEPS its indicated
        hydrogen even when a substituent / ring bond takes one of its hydrogens
        — ``2H,2'H-2,2'-bipyran`` attaches AT the 2H carbon and still cites it.
      - An AROMATIC pyrrole-type atom whose hydrogen has been displaced (no H
        left) leaves the ring fully mancude with NO saturated position, so no
        indicated hydrogen is cited — ``1,1'-bipyrrole``, not ``1H,1'H-…``
        .

    Returns the list of such atom indices (unsorted; caller maps them to the
    assembly numbering). Fails soft to ```` if the ring cannot be kekulized.
    """
    ring_set = set(ring)
    try:
        km = Chem.Mol(mol)
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:
        return []
    out: List[int] = []
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)  # original: aromaticity + H count
        if atom.GetAtomicNum() not in _INDICATED_H_ATOM_ELEMENTS:
            continue
        katom = km.GetAtomWithIdx(idx)
        has_ring_double = any(
            bond.GetOtherAtomIdx(idx) in ring_set
            and bond.GetBondType() == rdchem.BondType.DOUBLE
            for bond in katom.GetBonds()
        )
        if has_ring_double:
            continue
        # A ring atom bearing an EXOCYCLIC double bond (the ``=C`` junction of a
        # ylidene ring assembly, or an exocyclic =O/=N) is sp2, not a
        # saturated position, so it holds no indicated hydrogen. Excluding it is
        # what leaves ``2'H,3H-2,3'-bifuranylidene`` citing only the CH2 positions
        # (3 and 2') rather than also the two ylidene carbons.
        has_exo_double = any(
            bond.GetOtherAtomIdx(idx) not in ring_set
            and bond.GetBondType() == rdchem.BondType.DOUBLE
            for bond in katom.GetBonds()
        )
        if has_exo_double:
            continue
        # An indicated-H bearer stripped of its hydrogen holds NO indicated
        # hydrogen. Two sub-cases, both keyed on 0 remaining H:
        # - AROMATIC: N-substituted pyrrole / the ring bond of 1,1'-bipyrrole
        # (the ring stays fully mancude -> no saturated position);
        # - NON-aromatic: the pyridine-type ring nitrogen at a ring-assembly
        # junction (2H-1,2'-bipyridine cites only the 2H CH2, NOT a 1H on the
        # substituted N). A pyridine N is sp2 with no H in the mancude parent,
        # so once substituted it carries a substituent, not an indicated H.
        # A genuinely-sp3 indicated position (2H-pyran's C2) keeps 1 H after the
        # ring bond and is still cited (2H,2'H-2,2'-bipyran), so 0-H is the clean
        # discriminator regardless of aromaticity.
        if atom.GetTotalNumHs() == 0:
            continue
        out.append(idx)
    return out


def _compute_per_ring_ih_locants(
    mol, ring_systems, per_system_locants,
) -> List[Optional[List[int]]]:
    """Per-component indicated-hydrogen locants in the ASSEMBLY numbering.

    IUPAC (the Blue Book): the indicated hydrogen of a ring assembly is
    cited per component ring at its own required position. For each SINGLE-RING
    component (Fix1 scope; fused components keep the front-replicate path) this
    returns the sorted list of assembly-numbering locants of its indicated-H
    atoms (see ``_ring_indicated_h_atoms`` for which atoms qualify, including the
    substitution rule that drops the aromatic ``1,1'-bipyrrole`` descriptor but
    keeps the sp3 ``2H,2'H-2,2'-bipyran`` one).

    Returns a list aligned to ``ring_systems``; each entry is the sorted locant
    list for a single-ring component, or ``None`` when the component is fused
    (multi-ring) or its numbering is unavailable — a ``None`` anywhere signals
    the caller to fall back to the front-replicate descriptor for the whole
    assembly.
    """
    ri = mol.GetRingInfo()

    result: List[Optional[List[int]]] = []
    for sys_idx, sys_atoms in enumerate(ring_systems):
        sys_rings = [r for r in ri.AtomRings() if set(r) <= sys_atoms]
        if len(sys_rings) != 1:
            result.append(None)  # fused component: keep front-replicate
            continue
        ring = sys_rings[0]
        ih_atoms = _ring_indicated_h_atoms(mol, ring)
        if (per_system_locants is None
                or sys_idx >= len(per_system_locants)
                or per_system_locants[sys_idx] is None):
            result.append(None)
            continue
        locmap = per_system_locants[sys_idx]
        if any(a not in locmap for a in ih_atoms):
            result.append(None)  # incomplete numbering: fall back
            continue
        result.append(sorted(locmap[a] for a in ih_atoms))
    return result


def _needs_von_baeyer_parens(name: str) -> bool:
    """True if a ring-assembly component name must be parenthesized.

    Triggers for cycloalkane / cycloalkene / ylidene monocycles and von Baeyer
    polycyclics (bicyclo[..]/tricyclo[..]/spiro[..]); mancude ring names (which
    never start with a 'cyclo' hydrocarbon stem) are left bare.
    """
    if not name:
        return False
    # von Baeyer descriptor: "...cyclo[<digit>..." or "spiro[<digit>...".
    if re.search(r"cyclo\[\d", name) or name.startswith("spiro["):
        return True
    # Monocyclic cycloalkane / cycloalkene / ylidene hydrocarbon stem.
    return bool(_VON_BAEYER_NAME_RE.match(name))


def _enclose_component(name: str) -> str:
    """Enclose a ring-assembly component name in parentheses when
    requires it (von Baeyer disambiguation); otherwise return it unchanged."""
    return f"({name})" if _needs_von_baeyer_parens(name) else name


def _to_ylidene(name: str) -> Optional[str]:
    """Convert a saturated carbocycle parent-hydride name to its ylidene
    substituent-group name for a double-bond junction
    (cyclopentane -> cyclopentylidene). Returns None if ``name`` is not a clean
    'cyclo...ane' hydrocarbon ring, so the caller fails closed (no wrong names)."""
    if name and name.startswith("cyclo") and name.endswith("ane"):
        return name[:-3] + "ylidene"
    return None


def _is_saturated_carbocycle(mol, system_atoms: Set[int]) -> bool:
    """True if a ring system is a SINGLE all-carbon, non-aromatic ring with no
    internal ring double bond -- the monocyclic cycloalkane class nameable as a
     ylidene assembly (matching ``_to_ylidene``'s 'cyclo...ane'
    capability). Multi-ring saturated carbocycles (norbornane, decalin) are
    EXCLUDED so detect_ring_assembly does not claim a double-bond junction it
    cannot name -- otherwise namer.py early-returns on the assembly and the
    namer's None falls through to a fail-closed 'unknown'. A10: claim only what
    we can name; everything else keeps its prior (von Baeyer / substituent) path."""
    ri = mol.GetRingInfo()
    rings_in_system = [r for r in ri.AtomRings() if set(r) <= system_atoms]
    if len(rings_in_system) != 1:
        return False
    for i in system_atoms:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != "C" or a.GetIsAromatic():
            return False
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            a1 in system_atoms
            and a2 in system_atoms
            and bond.GetBondType() == rdchem.BondType.DOUBLE
        ):
            return False
    return True


def _mancude_ring_stem(mol, system_atoms: Set[int]) -> Optional[str]:
    """Bare mancude parent-hydride stem of a monocyclic heterocyclic ring system.

    IUPAC /: a heterocyclic double-bond (ylidene) ring assembly
    such as ``2'H,3H-2,3'-bifuranylidene`` is named on the MANCUDE parent stem
    (``furan``), not the partially saturated component name that
    ``_get_ring_parent_name`` returns for the actual ylidene ring
    (``2,3-dihydrofuran``). Rebuild the ring as a fully aromatic system and let
    RDKit validate the aromaticity, then name it with the heterocycle namer. This
    handles the whole aromatizable-heterocycle class (furan, thiophene,...) and
    fails closed (``None``) for anything that is not a clean aromatic mancude ring
    or whose stem is not a bare ``^[a-z]+$`` name (a stem carrying its own
    indicated hydrogen or locant set, e.g. ``1H-pyrrole``, is not built here).
    """
    ri = mol.GetRingInfo()
    ring = next((r for r in ri.AtomRings() if set(r) == set(system_atoms)), None)
    if ring is None:
        return None
    if not any(mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring):
        return None  # carbocyclic ylidenes go through _to_ylidene, not here
    n = len(ring)
    ring_set = set(ring)
    adj: Dict[int, List[int]] = {i: [] for i in ring}
    for i in ring:
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if j in ring_set:
                adj[i].append(j)
    if any(len(v) != 2 for v in adj.values()):
        return None  # not a simple monocycle
    # Walk the ring in cyclic order.
    order = [ring[0]]
    prev, cur = None, ring[0]
    while len(order) < n:
        nxt = next((x for x in adj[cur] if x != prev), None)
        if nxt is None:
            return None
        order.append(nxt)
        prev, cur = cur, nxt
    rw = Chem.RWMol()
    idxmap: Dict[int, int] = {}
    for i in order:
        a = Chem.Atom(mol.GetAtomWithIdx(i).GetAtomicNum())
        a.SetIsAromatic(True)
        idxmap[i] = rw.AddAtom(a)
    for k in range(n):
        i, j = order[k], order[(k + 1) % n]
        rw.AddBond(idxmap[i], idxmap[j], rdchem.BondType.AROMATIC)
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return None
    if not all(m2.GetAtomWithIdx(i).GetIsAromatic() for i in range(n)):
        return None
    from .heterocycles import name_heterocycle
    m2_ring = next(
        (r for r in m2.GetRingInfo().AtomRings() if len(r) == n), None)
    if m2_ring is None:
        return None
    stem = name_heterocycle(m2, m2_ring)
    if not stem or not re.match(r"^[a-z]+$", stem):
        return None
    return stem


def _is_mancude_ylidene_component(mol, system_atoms: Set[int]) -> bool:
    """True if a ring system is a single mancude heterocyclic ring nameable as a
    / ylidene assembly component (``furan`` -> ``bifuranylidene``).

    Distinct from ``_is_saturated_carbocycle`` (carbocyclic ylidene, no
    heteroatom) and from ``_is_replacement_assembly_candidate`` (SATURATED
    heteromonocycle -> replacement 'a'-nomenclature). Delegates the clean-stem
    test to ``_mancude_ring_stem`` so the admitted set is exactly what the ylidene
    namer can spell."""
    ri = mol.GetRingInfo()
    rings = [r for r in ri.AtomRings() if set(r) <= system_atoms]
    if len(rings) != 1 or set(rings[0]) != set(system_atoms):
        return False
    return _mancude_ring_stem(mol, system_atoms) is not None


def _is_mancude_indicated_h_state(mol, system_atoms: Set[int]) -> bool:
    """True if a single monocyclic ring is in a MANCUDE state -- carrying the
    maximum number of noncumulative ring double bonds, so every saturated ring
    position is a genuine indicated-hydrogen position and NOT a hydro (di- /
    tetra- / perhydro) saturation.

    IUPAC / (the Blue Book): the indicated hydrogen of a ring
    assembly component (the ``2H`` of ``2H-1,2'-bipyridine``) is a property OF the
    one mancude parent; aromatic pyridine and its N-substituted 2H tautomer are
    the SAME mancude ring in two indicated-hydrogen states. A ring that could
    still accept a further noncumulative ring double bond -- two ADJACENT
    saturated ring atoms that both still bear a hydrogen -- is instead a HYDRO
    derivative (1,2-dihydropyridine, piperidine): a genuinely different parent,
    NOT another indicated-H tautomer. ``_mancude_ring_stem`` re-aromatizes ANY
    ring (piperidine normalizes to ``pyridine`` too), so this test is what keeps
    the assembly detector from merging a mancude ring with its hydro partner.

    A saturated ring atom = no ring OR exocyclic double bond in the kekulized
    structure. A further double bond is addable between two adjacent saturated
    ring atoms only when BOTH still carry a hydrogen -- a substituted nitrogen
    with no H (as at the ring-assembly junction) cannot, which is exactly why the
    N-substituted 2H-pyridine stays mancude. Fails soft to ``False`` if the ring
    cannot be kekulized or is not a simple monocycle.
    """
    ri = mol.GetRingInfo()
    ring = next((r for r in ri.AtomRings() if set(r) == set(system_atoms)), None)
    if ring is None:
        return False
    ring_set = set(ring)
    try:
        km = Chem.Mol(mol)
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:
        return False

    def _saturated(idx: int) -> bool:
        for bond in km.GetAtomWithIdx(idx).GetBonds():
            if bond.GetBondType() == rdchem.BondType.DOUBLE:
                return False  # a ring OR exocyclic double bond -> sp2, not saturated
        return True

    # NOT mancude if a further noncumulative ring double bond can be added: two
    # adjacent saturated ring atoms that both still bear a hydrogen.
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 not in ring_set or a2 not in ring_set:
            continue
        if not _saturated(a1) or not _saturated(a2):
            continue
        if (mol.GetAtomWithIdx(a1).GetTotalNumHs() >= 1
                and mol.GetAtomWithIdx(a2).GetTotalNumHs() >= 1):
            return False
    return True


def _system_signature(mol, system_atoms: Set[int]) -> Tuple:
    """
    Compute a ring system signature for identity comparison.

    The signature includes sorted element symbols, sorted ring sizes that
    are fully contained within the system, aromaticity of the system, and
    the intra-system double/triple bond counts. Two systems are identical
    if and only if their signatures match.

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in the ring system

    Returns:
        Tuple of (sorted_elements, sorted_ring_sizes, is_aromatic,
        double_bond_count, triple_bond_count)

    Note (W8-P11 leak fix): element/size/aromaticity alone cannot
    distinguish e.g. cyclooctane from cyclooctene (both all-carbon,
    ring_sizes=(8,), non-aromatic). Without a saturation check, a
    cyclooctene+cyclooctane pair joined by a single bond falsely compared
    EQUAL and was accepted as an identical-ring "bi-" assembly; the namer
    then applied the FIRST ring's (unsaturated) name to BOTH components,
    emitting e.g. "1,1'-bi(cyclooctene)" for a molecule where only one ring
    actually has the double bond -- a wrong-structure name that only
    OPSIN's self-consistency gate caught (which fails OPEN with no
    Java available). Counting intra-system double/triple bonds makes
    genuinely-identical rings compare equal (same connectivity -> same
    bond-order multiset) while rejecting rings that merely share element
    composition, size and aromaticity but differ in saturation.
    """
    elements = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in system_atoms)

    ri = mol.GetRingInfo()
    ring_sizes = []
    for ring in ri.AtomRings():
        if set(ring) <= system_atoms:
            ring_sizes.append(len(ring))

    is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in system_atoms)

    double_bonds = 0
    triple_bonds = 0
    for bond in bonds_of(mol):
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in system_atoms and a2 in system_atoms:
            bt = bond.GetBondType()
            if bt == rdchem.BondType.DOUBLE:
                double_bonds += 1
            elif bt == rdchem.BondType.TRIPLE:
                triple_bonds += 1

    return (tuple(elements), tuple(sorted(ring_sizes)), is_aromatic,
            double_bonds, triple_bonds)


# skeletal-replacement ring-assembly heteroatoms (element-seniority
# citation order O > S > Se > Te; reuse the 'a'-terms from skeletal_replacement).
_RA_REPL_TERMS = {'O': 'oxa', 'S': 'thia', 'Se': 'selena', 'Te': 'tellura'}
_RA_REPL_ORDER = ['O', 'S', 'Se', 'Te']  # element seniority set order)


def _is_replacement_assembly_candidate(
    mol, ring_systems: List[Set[int]], connections: List[Tuple[int, int, int, int]]
) -> bool:
    """/ gate: exactly two SAME-SIZE saturated monocyclic
    components of MORE THAN TEN ring atoms (the size at which the cycloalkane
    carries its heteroatoms by skeletal replacement rather than a Hantzsch-Widman
    stem), each all-carbon except AT MOST one neutral divalent ring heteroatom
    from {O,S,Se,Te}, joined by exactly one ring-to-ring SINGLE or DOUBLE
    (ylidene) bond. At least one component must carry a heteroatom (else it is a
    plain carbocyclic assembly handled elsewhere). The two rings may be identical
    (``3,3'-dioxa-1,1'-bi(cyclotetradecane)``,
    ``2,2'-dithia-1,1'-bi(cyclododecylidene)``) or differ only by the replacement
    heteroatom (``3'-oxa-2-thia-1,1'-bi(cyclotetradecane)``). Fail-closed for
    anything richer."""
    if len(ring_systems) != 2 or len(connections) != 1:
        return False
    a1, a2, _, _ = connections[0]
    bond = mol.GetBondBetweenAtoms(a1, a2)
    if bond is None or bond.GetBondType() not in (
        rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE
    ):
        return False
    ri = mol.GetRingInfo()
    sizes = []
    total_hetero = 0
    for sys_atoms in ring_systems:
        rings = [r for r in ri.AtomRings() if set(r) <= sys_atoms]
        if len(rings) != 1:
            return False  # multi-ring component -> not this class
        ring = rings[0]
        if set(ring) != set(sys_atoms):
            return False  # exocyclic atoms in the system -> not a clean monocycle
        sizes.append(len(ring))
        het = 0
        for i in ring:
            a = mol.GetAtomWithIdx(i)
            sym = a.GetSymbol()
            if sym == 'C':
                if a.GetIsAromatic():
                    return False
                continue
            if (sym not in _RA_REPL_TERMS or a.GetFormalCharge() != 0
                    or a.GetDegree() != 2 or a.GetIsAromatic()):
                return False
            het += 1
        if het > 1:
            return False  # >1 heteroatom per ring -> outside this narrow class
        total_hetero += het
        # No ring double bonds (saturated replacement ring).
        for b in bonds_of(mol):
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in sys_atoms and j in sys_atoms and \
                    b.GetBondType() == rdchem.BondType.DOUBLE:
                return False
    if sizes[0] != sizes[1] or total_hetero == 0:
        return False
    # Only rings LARGER than the Hantzsch-Widman range (>10) are named by
    # skeletal replacement; a small saturated heteromonocycle (oxane, thiane,...)
    # keeps its HW/retained assembly name and must NOT be diverted here.
    if sizes[0] <= 10:
        return False
    # No substituents beyond the inter-ring bond (each ring atom degree <=2
    # except the two attachment atoms which are degree 3 via the junction).
    attach = {a1, a2}
    for sys_atoms in ring_systems:
        for i in sys_atoms:
            deg = mol.GetAtomWithIdx(i).GetDegree()
            if i in attach:
                if deg != 3:
                    return False
            elif deg != 2:
                return False
    return True


def _name_replacement_ring_assembly(mol, assembly_info: Dict) -> Optional[str]:
    """: name a two-component same-skeleton monocyclic ring assembly by
    skeletal-replacement ('a') nomenclature. Each ring is numbered from its
    attachment atom (locant 1); the heteroatom takes the lower of the two ring
    directions. Across the two rings, the unprimed/primed assignment is the one
    giving the LOWEST COMBINED heteroatom locant set; 'a'-prefixes are cited in
    element-seniority order (O>S>Se>Te). Emits e.g.
    '3'-oxa-2-thia-1,1'-bi(cyclotetradecane)'. Fail-closed on any ambiguity."""
    from ..data.chain_names import get_chain_prefix
    ring_systems = assembly_info['ring_systems']
    connections = assembly_info['connections']
    a1, a2, s1, s2 = connections[0]
    ri = mol.GetRingInfo()

    def _component(sys_atoms, attach):
        """Return (ring_size, hetero_locant_or_None, hetero_symbol_or_None):
        number the monocycle from ``attach`` (=1), choosing the direction that
        gives the single heteroatom (if any) the lowest locant."""
        ring = next(r for r in ri.AtomRings() if set(r) <= sys_atoms)
        n = len(ring)
        # Build the two cyclic orderings starting at `attach`.
        # Neighbours of attach within the ring:
        nbrs = [x.GetIdx() for x in mol.GetAtomWithIdx(attach).GetNeighbors()
                if x.GetIdx() in sys_atoms]
        if len(nbrs) != 2:
            return None
        best = None  # (hetero_locant or n+1, ordering)
        for start_nbr in nbrs:
            order = [attach, start_nbr]
            prev, cur = attach, start_nbr
            while len(order) < n:
                nxt = next(
                    (x.GetIdx() for x in mol.GetAtomWithIdx(cur).GetNeighbors()
                     if x.GetIdx() in sys_atoms and x.GetIdx() != prev),
                    None)
                if nxt is None:
                    break
                order.append(nxt)
                prev, cur = cur, nxt
            if len(order) != n:
                continue
            het_loc = None
            het_sym = None
            for pos, idx in enumerate(order, start=1):
                if mol.GetAtomWithIdx(idx).GetSymbol() != 'C':
                    het_loc = pos
                    het_sym = mol.GetAtomWithIdx(idx).GetSymbol()
                    break
            key = het_loc if het_loc is not None else n + 1
            if best is None or key < best[0]:
                best = (key, het_loc, het_sym)
        if best is None:
            return None
        return (n, best[1], best[2])

    c1 = _component(ring_systems[s1], a1)
    c2 = _component(ring_systems[s2], a2)
    if c1 is None or c2 is None:
        return None
    n1, loc1, sym1 = c1
    n2, loc2, sym2 = c2
    if n1 != n2:
        return None

    # Assign which ring is unprimed to minimise the combined heteroatom locant
    # set: low locants to heteroatoms as a set). Each ring contributes
    # at most one heteroatom; compare the two orientations.
    def _combined(order):
        # order = list of (loc, sym, is_primed) sorted by (loc, prime)
        out = []
        for loc, sym, primed in order:
            if loc is None:
                continue
            out.append((loc, primed))
        return sorted(out)

    # Orientation A: ring1 unprimed, ring2 primed.
    optA = [(loc1, sym1, False), (loc2, sym2, True)]
    # Orientation B: ring2 unprimed, ring1 primed.
    optB = [(loc2, sym2, False), (loc1, sym1, True)]
    setA = _combined(optA)
    setB = _combined(optB)
    chosen = optA if setA <= setB else optB

    # Cite 'a'-prefixes in element-seniority order (O>S>Se>Te). When the SAME
    # replacement heteroatom appears in both rings it is cited ONCE with a
    # multiplying prefix and a combined locant set multiplication):
    # two ring oxygens at 3 and 3' -> '3,3'-dioxa', two ring sulfurs at 2 and 2'
    # -> '2,2'-dithia'; distinct elements stay separate ('3'-oxa-2-thia').
    by_elem: Dict[str, List[Tuple[int, bool]]] = {}
    for loc, sym, primed in chosen:
        if loc is None or sym is None:
            continue
        if sym not in _RA_REPL_TERMS:
            return None
        by_elem.setdefault(sym, []).append((loc, primed))
    if not by_elem:
        return None
    _MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}
    cited: List[Tuple[int, str]] = []
    for sym, entries in by_elem.items():
        entries.sort()
        mult = _MULT.get(len(entries))
        if mult is None:
            return None
        locant_str = ",".join(
            f"{loc}{chr(39) if primed else ''}" for loc, primed in entries)
        cited.append(
            (_RA_REPL_ORDER.index(sym),
             f"{locant_str}-{mult}{_RA_REPL_TERMS[sym]}"))
    cited.sort(key=lambda x: x[0])
    prefix = "-".join(part for _, part in cited)

    stem = get_chain_prefix(n1)  # e.g. 'tetradec' for 14
    #: a double-bond (ylidene) junction names the ring skeleton as
    # 'cyclo...ylidene' (2,2'-dithia-1,1'-bi(cyclododecylidene)); a single-bond
    # junction keeps 'cyclo...ane'. Both attachment atoms are locant 1 in their
    # rings -> '1,1'-bi(cyclo...)'.
    skeleton = (f"cyclo{stem}ylidene"
                if assembly_info.get('double_bond_junction')
                else f"cyclo{stem}ane")
    return f"{prefix}-1,1'-bi({skeleton})"


def _find_inter_system_bonds(
    mol, ring_systems: List[Set[int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Find all bonds connecting different ring systems.

    Returns list of (atom_idx_A, atom_idx_B, system_idx_A, system_idx_B).
    SINGLE and AROMATIC bonds are the single-bond junction; biphenyl's
    inter-ring bond may be typed as either depending on Kekulization). DOUBLE
    bonds are accepted for the double-bond junction (bi(...ylidene));
    detect_ring_assembly gates which double-bond junctions are actually claimed.
    """
    connections = []
    acceptable_types = {
        rdchem.BondType.SINGLE,
        rdchem.BondType.AROMATIC,
        rdchem.BondType.DOUBLE,
    }

    # Build atom -> set of system indices (an atom can be in multiple systems
    # for spiro compounds where the spiro center is shared)
    atom_to_systems: Dict[int, Set[int]] = {}
    for sys_idx, system in enumerate(ring_systems):
        for atom_idx in system:
            if atom_idx not in atom_to_systems:
                atom_to_systems[atom_idx] = set()
            atom_to_systems[atom_idx].add(sys_idx)

    # Atoms in multiple systems (spiro centers) -- skip bonds involving these
    shared_atoms = {idx for idx, systems in atom_to_systems.items() if len(systems) > 1}

    for bond in bonds_of(mol):
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # Skip bonds involving spiro/shared atoms
        if a1 in shared_atoms or a2 in shared_atoms:
            continue

        # Both atoms must be in ring systems, but different ones
        if a1 in atom_to_systems and a2 in atom_to_systems:
            systems1 = atom_to_systems[a1]
            systems2 = atom_to_systems[a2]
            # Each should be in exactly one system for a true assembly bond
            if len(systems1) == 1 and len(systems2) == 1:
                s1 = next(iter(systems1))
                s2 = next(iter(systems2))
                if s1 != s2 and bond.GetBondType() in acceptable_types:
                    connections.append((a1, a2, s1, s2))

    return connections


def _check_all_connected(
    num_systems: int, connections: List[Tuple[int, int, int, int]]
) -> bool:
    """
    Check that all ring systems are connected via inter-system bonds.

    Uses union-find to verify connectivity.
    """
    if num_systems <= 1:
        return True

    parent = list(range(num_systems))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for _, _, s1, s2 in connections:
        union(s1, s2)

    roots = set(find(i) for i in range(num_systems))
    return len(roots) == 1


def _check_path_topology(
    num_systems: int, connections: List[Tuple[int, int, int, int]]
) -> bool:
    """a phase-03: linear-path requirement for ring assemblies.

    Every system must have degree <= 2 in the inter-system bond graph.
    Branched arrangements (e.g., 1,3,5-triphenylbenzene where the central
    benzene has degree 3) are NOT ring assemblies per IUPAC; they
    fall through to substituent-based naming.

    Args:
        num_systems: number of ring systems.
        connections: list of (atom_a, atom_b, system_a, system_b) tuples
            from ``_find_inter_system_bonds``.

    Returns:
        True iff every system index appears at most twice across
        ``connections`` (a linear path / single bond ring assembly);
        False otherwise.

    Source: 151-internal notes.
    Source: internal notes §"Code Examples" Example 4.
    Source: AUTONOM-1990; IUPAC Blue Book.
    """
    if num_systems <= 1:
        return True
    deg: Counter = Counter()
    for _, _, s1, s2 in connections:
        deg[s1] += 1
        deg[s2] += 1
    if not all(d <= 2 for d in deg.values()):
        return False
    # a phase-04: reject cyclic ring-system arrangements.
    # A linear path of N systems has exactly N-1 inter-system bonds.
    # A cycle has N (every node degree 2 — passes the degree check).
    # IUPAC requires a linear path; a cyclic arrangement is a
    # different topology (would be a fused/bridged macrocyclic system).
    if len(connections) != num_systems - 1:
        return False
    return True


def _order_systems_along_path(
    num_systems: int,
    connections: List[Tuple[int, int, int, int]],
) -> Optional[List[int]]:
    """a phase-04 Scenario A root-cause fix.

    Given a linear-path inter-system bond graph (verified by
    ``_check_path_topology``), return the system indices in path order
    starting from one terminal end.

    For a path A—B—C, returns ``[A, B, C]`` where A and C are the
    terminal systems (degree 1 in the inter-system graph) and B is the
    middle system (degree 2). This is the canonical IUPAC
    ordering: terminal rings get unprimed and double-primed namespaces,
    middle rings get the single-prime namespace.

    Without this reordering, ``get_ring_systems`` returns ring systems
    in atom-traversal order — the middle ring of a substituted terphenyl
    can land at index 0, causing the connection-string emission to drop
    the prime on the middle ring's back-attachment locant
    (e.g., emitting ``1,1':4,1''-terphenyl`` instead of the canonical
    ``1,1':4',1''-terphenyl``). This was the defect captured at
    151-04-DIAGNOSTIC.md.

    Args:
        num_systems: number of ring systems.
        connections: list of (atom_a, atom_b, system_a, system_b) tuples
            from ``_find_inter_system_bonds``.

    Returns:
        List of system indices in path order, or None if a unique linear
        ordering cannot be determined (caller should fall through).

    Source: IUPAC Blue Book (single-prime namespace = middle ring).
    Source: 151-04-PLAN.md Task 3 Scenario A.
    """
    if num_systems <= 1:
        return list(range(num_systems))

    # Build adjacency map (system -> set of neighbor systems)
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_systems)}
    for _, _, s1, s2 in connections:
        adj[s1].add(s2)
        adj[s2].add(s1)

    # Tree-shape gate: linear path => exactly two terminals (degree 1).
    # If 0 terminals, graph is a cycle (rejected by _check_path_topology
    # tree-shape gate). If >2 terminals, branched (also rejected).
    terminals = [i for i, neighbors in adj.items() if len(neighbors) == 1]
    if len(terminals) != 2:
        return None

    # Pick the lower-indexed terminal as the start to make the ordering
    # deterministic. Walking from the OTHER terminal would give the
    # reversed path, but per IUPAC lowest-locant the symmetric
    # case is handled downstream by compare_locant_sets in
    # _compute_per_system_ring_locants.
    start = min(terminals)

    order: List[int] = [start]
    seen: Set[int] = {start}
    current = start
    while len(order) < num_systems:
        next_neighbors = adj[current] - seen
        if not next_neighbors:
            # Disconnected — should not happen given _check_all_connected
            return None
        # Linear-path invariant: at most one unseen neighbor.
        if len(next_neighbors) > 1:
            return None
        nxt = next(iter(next_neighbors))
        order.append(nxt)
        seen.add(nxt)
        current = nxt

    return order


def detect_ring_assembly(
    mol, ring_systems: List[Set[int]]
) -> Optional[Dict]:
    """
    Detect if a molecule is a ring assembly (identical ring systems joined
    by single bonds).

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets of atom indices, one per ring system
                      (from get_ring_systems)

    Returns:
        Dict with assembly info if detected, None otherwise.
        Dict keys:
          - ring_systems: list of sets of atom indices
          - connections: list of (atom_A, atom_B, system_A, system_B) tuples
          - count: number of identical ring systems
          - ring_type: 'carbocyclic' or 'heterocyclic'

    a phase.B cross-handler contract:
        Ring-assembly detection (this function) and multiplicative naming
        (rules.multiplicative.name_multiplicative) are MUTUALLY EXCLUSIVE
        by topology. Ring assemblies = identical rings joined directly by
        a single bond (no bridge atom). Multiplicative = identical parent
        units joined by 1+ bridge atoms (oxy / methylene / nitrilo /...).
        The split is enforced symmetrically:
          - this function rejects atom-bridged cases via
            _find_inter_system_bonds (which only matches ring-to-ring
            single bonds; bridge atoms are NOT in rings, so atom-bridged
            cases never produce inter-system bonds)
          - name_multiplicative rejects single-bond-only cases via
            _is_pure_single_bond_assembly at the entry point (guard)
        Cross-handler regression test:
        tests/integration/test_assembly_vs_multiplicative_dispatch.py.

    Source: 154-internal notes; 151-internal notes (path-topology contract).
    """
    if len(ring_systems) < 2:
        return None

    # Find inter-system single bonds
    connections = _find_inter_system_bonds(mol, ring_systems)
    if not connections:
        return None

    # Check that all systems are connected
    if not _check_all_connected(len(ring_systems), connections):
        return None

    # a phase-03: linear-path requirement (every system degree <= 2).
    # Rejects branched arrangements like 1,3,5-triphenylbenzene where the
    # central system has degree 3. AUTONOM-1990 + IUPAC.
    if not _check_path_topology(len(ring_systems), connections):
        return None

    # Compare signatures -- identical rings take the main naming path below.
    signatures = [_system_signature(mol, sys_atoms) for sys_atoms in ring_systems]
    identical = len(set(signatures)) == 1
    mancude_tautomer = False  # set when merged via the indicated-H tautomer path

    # IUPAC: a junction may be a DOUBLE bond (bi(...ylidene)).
    # _find_inter_system_bonds also returns double-bond junctions.
    double_bond_junction = any(
        (bond := mol.GetBondBetweenAtoms(a1, a2)) is not None
        and bond.GetBondType() == rdchem.BondType.DOUBLE
        for a1, a2, _, _ in connections
    )

    # /: skeletal-replacement ('a') nomenclature names a
    # two-component SATURATED heteromonocyclic assembly (>10-membered rings), be
    # the rings identical (share a signature: 3,3'-dioxa-1,1'-bi(cyclotetradecane),
    # 2,2'-dithia-1,1'-bi(cyclododecylidene)) or differing only by the replacement
    # heteroatom (3'-oxa-2-thia-1,1'-bi(cyclotetradecane), signatures differ). The
    # junction may be single (bi(cyclo...ane)) or double (bi(cyclo...ylidene)).
    # Checked here for BOTH signature cases; everything richer falls through.
    if _is_replacement_assembly_candidate(mol, ring_systems, connections):
        return {
            'ring_systems': ring_systems,
            'connections': connections,
            'count': len(ring_systems),
            'ring_type': 'heterocyclic',
            'double_bond_junction': double_bond_junction,
            'replacement': True,
        }

    if not identical:
        # (the Blue Book) /: two rings that are the SAME mancude
        # parent in different indicated-hydrogen states (aromatic pyridine + its
        # N-substituted 2H tautomer) form ONE ring assembly -- indicated H is a
        # property OF the mancude parent, not a distinct ring system. Their
        # _system_signature differs only in aromaticity / ring-double-bond count,
        # yet both normalize to the SAME mancude monocycle stem. Merge them --
        # SINGLE-bond junctions only, to leave the ylidene double-bond path (guard
        # below) exactly as it is -- when EVERY component is a known mancude
        # monocycle IN a mancude state (_is_mancude_indicated_h_state: not a hydro
        # derivative) and all normalize to the same stem. 9-B3a's per-ring
        # indicated-H recompute then cites the 2H on the correct component
        # (2H-1,2'-bipyridine). The mancude-state test is essential: bare stem
        # normalization also merges pyridine with piperidine / 1,2-dihydropyridine
        # (both re-aromatize to 'pyridine'), which are hydro derivatives, NOT
        # indicated-H tautomers.
        if double_bond_junction:
            return None
        stems = [_mancude_ring_stem(mol, sys_atoms) for sys_atoms in ring_systems]
        if not (all(s is not None for s in stems) and len(set(stems)) == 1):
            return None
        if not all(_is_mancude_indicated_h_state(mol, sys_atoms)
                   for sys_atoms in ring_systems):
            return None
        identical = True
        mancude_tautomer = True

    # Restrict the double-bond junctions this detector CLAIMS to the classes the
    # ylidene namer can build: the saturated-carbocycle class
    # (1,1'-bi(cyclopentylidene)) and the mancude heterocyclic class
    # (2'H,3H-2,3'-bifuranylidene). Other double-bond junctions return None here so
    # the molecule keeps its prior (substituent-based) naming instead of
    # early-returning in namer.py to a fail-closed dead end (A10, no wrong names).
    if double_bond_junction and not (
        all(_is_saturated_carbocycle(mol, sys_atoms)
            for sys_atoms in ring_systems)
        or all(_is_mancude_ylidene_component(mol, sys_atoms)
               for sys_atoms in ring_systems)
    ):
        return None

    # Additional guard: reject if there are non-ring atoms in the molecule
    # other than substituents (i.e., linker atoms between rings).
    # For true ring assemblies, the inter-ring bond is direct (no linker).
    # This is already ensured by _find_inter_system_bonds checking that
    # both atoms are IN ring systems.

    # a phase-04 Scenario A root-cause fix: reorder ring_systems
    # along the inter-system path so the middle ring of a ter-/quater-/
    # quinque- assembly lands at the correct primed-namespace index.
    # Without this reordering, ``get_ring_systems`` returns systems in
    # atom-traversal order — for substituted ring assemblies the middle
    # ring can land at index 0, causing the connection-string to drop
    # the prime on the middle ring's back-attachment locant
    # (emitting ``1,1':4,1''-terphenyl`` instead of canonical
    # ``1,1':4',1''-terphenyl``). See 151-04-DIAGNOSTIC.md.
    path_order = _order_systems_along_path(len(ring_systems), connections)
    if path_order is not None and path_order != list(range(len(ring_systems))):
        # Build remapping: old_idx -> new_idx
        new_index_of = {old: new for new, old in enumerate(path_order)}
        ring_systems = [ring_systems[old] for old in path_order]
        connections = [
            (a1, a2, new_index_of[s1], new_index_of[s2])
            for a1, a2, s1, s2 in connections
        ]

    # Determine ring type
    elements = signatures[0][0]
    has_heteroatom = any(e != 'C' for e in elements)
    ring_type = 'heterocyclic' if has_heteroatom else 'carbocyclic'

    return {
        'ring_systems': ring_systems,
        'connections': connections,
        'count': len(ring_systems),
        'ring_type': ring_type,
        'double_bond_junction': double_bond_junction,
        'mancude_tautomer': mancude_tautomer,
    }


def _get_ring_parent_name(mol, system_atoms: Set[int]) -> Optional[str]:
    """
    Determine the parent name for a ring system in an assembly context.

    Per IUPAC:
      - Benzene (6-membered all-C aromatic) -> "phenyl" in assemblies
      - Heterocycles -> their parent name (pyridine, thiophene, furan, etc.)
      - Cycloalkanes -> their parent name (cyclohexane, cyclopentane, etc.)

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in one ring system

    Returns:
        Parent name string, or None if cannot determine.
    """
    # Get the individual rings that belong to this system
    ri = mol.GetRingInfo()
    system_rings = [ring for ring in ri.AtomRings() if set(ring) <= system_atoms]

    if not system_rings:
        return None

    # For single-ring systems (most common in assemblies)
    if len(system_rings) == 1:
        ring = system_rings[0]
        ring_size = len(ring)

        # Check if heterocyclic
        has_heteroatom = any(
            mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring
        )

        if has_heteroatom:
            # Use the heterocycle naming infrastructure.
            from .heterocycles import name_heterocycle
            # /: the component name is the free PARENT
            # HYDRIDE name of the ring, so its indicated hydrogen must be
            # computed on the ISOLATED ring — not on the ring embedded in the
            # assembly, where the inter-ring bond at the connection atom
            # suppresses the indicated-H (e.g. the assembly's ring gave
            # '1,4-oxaphosphinine' with no '4H', but the parent hydride is
            # '4H-1,4-oxaphosphinine'). Extract the ring as a standalone mol,
            # name it there, then let the assembly composer add the connection
            # locant + enclosing parentheses. Fail-safe: if the isolated
            # fragment cannot be built or named, fall back to the embedded name.
            try:
                frag_smi = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=sorted(ring), canonical=True)
                frag = Chem.MolFromSmiles(frag_smi) if frag_smi else None
            except Exception:
                frag = None
            if frag is not None:
                frag_rings = [r for r in frag.GetRingInfo().AtomRings()
                              if len(r) == ring_size]
                if len(frag_rings) == 1:
                    iso = name_heterocycle(frag, frag_rings[0])
                    if iso:
                        return iso
            return name_heterocycle(mol, ring)

        # All-carbon ring
        is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)

        if is_aromatic and ring_size == 6:
            # IUPAC: benzene in assemblies uses "phenyl"
            return "phenyl"

        # Cycloalkane or cycloalkene
        from ..data.chain_names import get_chain_prefix
        prefix = get_chain_prefix(ring_size)

        # Check saturation
        has_double_bond = False
        ring_set = set(ring)
        for bond in bonds_of(mol):
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in ring_set and a2 in ring_set:
                if bond.GetBondType() == rdchem.BondType.DOUBLE:
                    has_double_bond = True
                    break

        if has_double_bond:
            return f"cyclo{prefix}ene"
        else:
            return f"cyclo{prefix}ane"

    # Multi-ring fused systems as assembly units (e.g., binaphthalene, biquinoline)
    # Per IUPAC: fused ring systems can serve as identical ring assembly units.
    # Use existing infrastructure: dictionary fast path (150+ entries), then algorithmic fallback.

    # Step 1: Extract subsystem SMILES for lookup
    system_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(system_atoms), canonical=True)
    if not system_smi:
        return None

    # Step 2: Create standalone mol for canonical SMILES matching
    sub_mol = Chem.MolFromSmiles(system_smi)
    if sub_mol is None:
        return None
    can_smi = Chem.MolToSmiles(sub_mol, canonical=True)

    # Step 3: Try fused heterocycle dictionary (fast path, 150+ entries from a phase)
    from ..data.fused_heterocycles import get_fused_heterocycle_name
    fused_result = get_fused_heterocycle_name(sub_mol)
    if fused_result:
        name, _tautomer_locant = fused_result
        # Include indicated H prefix if present (e.g., "1H-indole")
        return name

    # Step 4: Try retained names for carbocyclic fused systems (naphthalene, anthracene, etc.)
    from ..data.retained_names import get_retained_name
    retained = get_retained_name(can_smi)
    if retained:
        return retained

    # Step 5: Try algorithmic fused ring generator (a phase fallback)
    from .fused_rings import _try_algorithmic_fusion_name
    algo_name = _try_algorithmic_fusion_name(sub_mol)
    if algo_name:
        return algo_name

    return None


def _get_connection_locant(
    mol, connecting_atom: int, system_atoms: Set[int]
) -> int:
    """
    Determine the IUPAC locant for a connecting atom within its ring system.

    For heterocyclic rings, the locant follows IUPAC numbering with the
    direction chosen to give the LOWEST locant at the connection point
    (IUPAC lowest-locant rule for assemblies).

    For carbocyclic rings, position 1 is the connecting atom itself
    (for unsubstituted benzene, all positions are equivalent).

    Args:
        mol: RDKit Mol object
        connecting_atom: Atom index of the connection point
        system_atoms: Set of atom indices in the ring system

    Returns:
        IUPAC locant number (1-indexed)
    """
    # Get the individual ring(s) containing this atom
    ri = mol.GetRingInfo()
    atom_ring = None
    for ring in ri.AtomRings():
        if connecting_atom in ring and set(ring) <= system_atoms:
            atom_ring = ring
            break

    if atom_ring is None:
        return 1  # Fallback

    ring_list = list(atom_ring)

    # Check if heterocyclic
    has_heteroatom = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_list
    )

    if has_heteroatom:
        # For heterocyclic assemblies: apply standard IUPAC numbering
        # (heteroatom at position 1), then try both numbering directions
        # and pick the one giving the LOWEST locant at the connection point.
        # orient_heterocycle may pick an arbitrary direction for single-heteroatom
        # rings, so we need to try both explicitly.
        from ..data.hw_heteroatoms import get_heteroatom_priority
        from .heterocycles import (
            classify_heterocycle,
        )

        info = classify_heterocycle(mol, ring_list)
        heteroatoms = info['heteroatoms']

        if not heteroatoms:
            return 1

        # Find highest-priority heteroatom for position 1
        sorted_ha = sorted(
            heteroatoms, key=lambda x: get_heteroatom_priority(x[1])
        )
        start_idx = sorted_ha[0][0]
        n = len(ring_list)
        start_pos = ring_list.index(start_idx)

        # Try both directions, pick the one giving lowest connection locant
        best_locant = n  # Worst case

        for direction in (1, -1):
            oriented = []
            for i in range(n):
                oriented.append(ring_list[(start_pos + direction * i) % n])
            # Find connection locant in this orientation
            if connecting_atom in oriented:
                locant = oriented.index(connecting_atom) + 1
                best_locant = min(best_locant, locant)

        return best_locant

    # Carbocyclic: for unsubstituted benzene, connection atom = position 1
    # For substituted, we need to number from connection point to give lowest
    # locants. Start with connection = 1.
    return 1


def _number_carbocyclic_from_anchor(
    mol, system_atoms: Set[int], anchor_atom: int,
    other_inter_system_atoms: Set[int],
) -> Optional[Dict[int, int]]:
    """a phase-03 /: per-ring numbering for a carbocyclic
    assembly component anchored at ``anchor_atom`` (which becomes locant 1).

    Numbers the ring atoms in the direction that gives the lowest locant
    set for the OTHER inter-system bond atoms (first-point-of-difference
    via ``compare_locant_sets``). Used by ``_compute_per_system_ring_locants``
    to produce the per-system IUPAC locant maps that drive name emission
    AND the cascade-step-6 supplier.

    Args:
        mol: RDKit Mol object.
        system_atoms: atoms forming this ring system.
        anchor_atom: atom that becomes locant 1 (the inter-system bond
            attaching this ring to the previous chain ring; for terminal
            rings, the only inter-system bond atom).
        other_inter_system_atoms: other inter-system bond atoms in this
            same ring system (e.g., for the middle ring of terphenyl,
            this is the singleton {atom-bonded-to-ring-2}).

    Returns:
        Dict mapping atom_idx -> int locant covering all ring atoms in
        the (single-)ring system. Returns None for multi-ring systems
        (those use absolute numbering via _get_ring_parent_name). For
        single-ring systems, the lowest-locant direction wins per.

    Source: 151-internal notes,,.
    Source: AUTONOM-1990 (criterion order — lowest-locant tiebreak).
    Source: IUPAC Blue Book.
    """
    from .locants import compare_locant_sets

    ri = mol.GetRingInfo()
    rings = [ring for ring in ri.AtomRings() if set(ring) <= system_atoms]
    if len(rings) != 1:
        return None  # multi-ring fused systems use absolute numbering
    ring_list = list(rings[0])
    n = len(ring_list)
    if anchor_atom not in ring_list:
        return None
    start_pos = ring_list.index(anchor_atom)

    best_locants: Optional[Dict[int, int]] = None
    best_other_set: Optional[List[int]] = None

    for direction in (1, -1):
        # Build oriented sequence: anchor_atom -> locant 1, walking direction
        oriented = [
            ring_list[(start_pos + direction * i) % n] for i in range(n)
        ]
        atom_to_locant = {a: i + 1 for i, a in enumerate(oriented)}
        if not other_inter_system_atoms:
            # Terminal ring: any direction equally valid; pick deterministically
            # (direction +1 first; later substituent tiebreak handled elsewhere)
            return atom_to_locant
        other_set = sorted(
            atom_to_locant[a]
            for a in other_inter_system_atoms
            if a in atom_to_locant
        )
        if not other_set:
            continue
        if best_other_set is None or compare_locant_sets(
            other_set, best_other_set
        ) < 0:
            best_other_set = other_set
            best_locants = atom_to_locant

    return best_locants


def _compute_per_system_ring_locants(
    mol, assembly_info: Dict,
) -> Optional[List[Dict[int, int]]]:
    """a phase-03 /: per-system IUPAC locant maps for a ring
    assembly. Each entry is a Dict[int, int] mapping atom_idx -> locant
    (within that system's own numbering).

    For carbocyclic single-ring components: locant 1 = the inter-system
    bond atom going TOWARDS the lower-indexed neighbour ring; the second
    bond atom (if present) gets the lowest-locant ring-walk distance per
     ``compare_locant_sets`` tiebreak.

    For heterocyclic components: heteroatom-priority numbering via
    ``_get_connection_locant`` (existing logic, own-numbering).

    For multi-ring fused components (e.g., biindole): absolute numbering
    by per-component canonical SMILES lookup is delegated to
    ``_get_connection_locant`` (which falls through to fused-heterocycle
    catalog via ``name_heterocycle``).

    Args:
        mol: RDKit Mol object.
        assembly_info: dict from detect_ring_assembly.

    Returns:
        List of per-system Dict[int, int] locant maps, indexed by system
        position (matching assembly_info['ring_systems'] order). Returns
        None if any system cannot be numbered fully.

    Source: 151-internal notes,,.
    Source: internal notes Pattern S-3 (cascade-step-6 supplier).
    """
    ring_systems = assembly_info["ring_systems"]
    connections = assembly_info["connections"]

    # Build per-system bond inventory: for each system idx, the list of
    # (own_atom, other_system_idx) pairs. The "anchor" for numbering is
    # the bond going TOWARDS the lower-indexed neighbour system; for the
    # leftmost terminal (sys 0), this is just its single bond.
    bonds_per_sys: Dict[int, List[Tuple[int, int]]] = {
        i: [] for i in range(len(ring_systems))
    }
    for a1, a2, s1, s2 in connections:
        bonds_per_sys[s1].append((a1, s2))
        bonds_per_sys[s2].append((a2, s1))

    result: List[Dict[int, int]] = []
    for sys_idx, sys_atoms in enumerate(ring_systems):
        my_bonds = bonds_per_sys[sys_idx]
        # Pick anchor = bond to the lower-indexed neighbour system.
        # If ``sys_idx == 0`` there is no lower neighbour — pick its
        # single bond's own-atom as the anchor.
        anchor_atom: Optional[int] = None
        other_atoms: Set[int] = set()
        for own_atom, other_sys in my_bonds:
            if other_sys < sys_idx:
                anchor_atom = own_atom
            else:
                other_atoms.add(own_atom)
        if anchor_atom is None and my_bonds:
            # Sys 0 (no lower neighbour): use its only own-atom as anchor.
            anchor_atom = my_bonds[0][0]
            other_atoms = {a for a, _ in my_bonds[1:]}

        ri = mol.GetRingInfo()
        sys_rings = [r for r in ri.AtomRings() if set(r) <= sys_atoms]

        # Determine if single-ring carbocyclic vs everything else.
        if len(sys_rings) == 1 and anchor_atom is not None:
            ring = sys_rings[0]
            has_hetero = any(
                mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring
            )
            if not has_hetero:
                # Carbocyclic single ring: number from anchor with
                # lowest-locant tiebreak on other_atoms.
                m = _number_carbocyclic_from_anchor(
                    mol, sys_atoms, anchor_atom, other_atoms
                )
                if m is not None and set(m.keys()) == set(ring):
                    result.append(m)
                    continue

        # Heterocyclic single-ring: number the ring with heteroatom-priority
        # IUPAC absolute numbering (heteroatom = locant 1). Walking
        # direction is chosen to give the LOWEST locant set to the
        # inter-system bond atoms connection-locant lowest-locant
        # rule + first-point-of-difference via compare_locant_sets).
        sys_map: Dict[int, int] = {}
        if len(sys_rings) == 1:
            ring = sys_rings[0]
            n = len(ring)
            ring_list = list(ring)
            from ..data.hw_heteroatoms import get_heteroatom_priority
            from .heterocycles import classify_heterocycle
            from .locants import compare_locant_sets

            try:
                info_het = classify_heterocycle(mol, ring_list)
                heteroatoms = info_het.get("heteroatoms", [])
            except Exception:
                heteroatoms = []

            if heteroatoms:
                sorted_ha = sorted(
                    heteroatoms,
                    key=lambda x: get_heteroatom_priority(x[1]),
                )
                start_idx = sorted_ha[0][0]
                start_pos = ring_list.index(start_idx)
                conn_atoms = (
                    ([anchor_atom] if anchor_atom is not None else [])
                    + sorted(other_atoms)
                )
                # BLOCKER-3: the indicated-hydrogen atoms of this ring, so
                # their locant set can break a walking-direction tie.
                # (the Blue Book) orders numbering criteria "heteroatoms have the
                # lower possible locants, then indicated hydrogen atoms..." and
                # (f) (the Blue Book) "low locants are assigned to
                # indicated hydrogen atoms". For a 2H-pyridin-1-yl junction the
                # inter-system bond atom IS the heteroatom (locant 1 in BOTH
                # directions), so the connection-locant set ties and direction
                # used to fall to RDKit atom order -> 2H- or 6H- depending on the
                # SMILES writing (non-deterministic, non-PIN). The iH locant set
                # is the next tiebreak: 2 < 6 selects 2H-1,2'-bipyridine for
                # every writing.
                ih_atoms = _ring_indicated_h_atoms(mol, ring)
                best_map: Optional[Dict[int, int]] = None
                best_locant_set: Optional[List[int]] = None
                best_ih_set: Optional[List[int]] = None
                for direction in (1, -1):
                    oriented = [
                        ring_list[(start_pos + direction * i) % n]
                        for i in range(n)
                    ]
                    cand_map = {a: i + 1 for i, a in enumerate(oriented)}
                    cand_set = sorted(cand_map[a] for a in conn_atoms if a in cand_map)
                    if not cand_set:
                        continue
                    cand_ih_set = sorted(
                        cand_map[a] for a in ih_atoms if a in cand_map
                    )
                    if best_locant_set is None:
                        take = True
                    else:
                        c = compare_locant_sets(cand_set, best_locant_set)
                        # Primary: lowest connection-locant set. When
                        # it ties, secondary: lowest indicated-H locant set
                        #. direction +1 (tried first) is the final
                        # stable tiebreak, so the order is total & deterministic.
                        take = c < 0 or (
                            c == 0
                            and compare_locant_sets(cand_ih_set, best_ih_set) < 0
                        )
                    if take:
                        best_locant_set = cand_set
                        best_ih_set = cand_ih_set
                        best_map = cand_map
                if best_map is not None:
                    sys_map = best_map
            if not sys_map:
                # No heteroatoms classified or numbering failed — fall
                # back to anchor-based carbocyclic numbering.
                if anchor_atom is not None and anchor_atom in ring:
                    fallback = _number_carbocyclic_from_anchor(
                        mol, sys_atoms, anchor_atom, other_atoms
                    )
                    if fallback is not None:
                        sys_map = fallback
            result.append(sys_map)
        else:
            # Multi-ring fused: defer to absolute numbering via
            # _get_connection_locant per atom. Coverage may be partial
            # for some catalogue entries; the supplier checks coverage
            # downstream and returns None if incomplete.
            for atom_idx in sys_atoms:
                loc = _get_connection_locant(mol, atom_idx, sys_atoms)
                sys_map[atom_idx] = loc
            result.append(sys_map)

    return result


def _get_substituent_locant(
    mol, sub_atom: int, ring_atom: int, system_atoms: Set[int],
    connecting_atom: int
) -> int:
    """
    Determine the IUPAC locant for a substituent attached to a ring in an assembly.

    For carbocyclic rings, numbering starts from the connection point (locant 1)
    and follows the direction giving the lowest locants for substituents.

    For heterocyclic rings, standard IUPAC numbering is used.

    Args:
        mol: RDKit Mol object
        sub_atom: Atom index of the substituent attachment point (in ring)
        ring_atom: Same as sub_atom (where substituent connects to ring)
        system_atoms: Set of atom indices in the ring system
        connecting_atom: Atom index of the inter-ring connection in this system

    Returns:
        IUPAC locant (1-indexed)
    """
    ri = mol.GetRingInfo()
    atom_ring = None
    for ring in ri.AtomRings():
        if sub_atom in ring and set(ring) <= system_atoms:
            atom_ring = ring
            break

    if atom_ring is None:
        return 1

    ring_list = list(atom_ring)
    ring_size = len(ring_list)

    has_heteroatom = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_list
    )

    if has_heteroatom:
        from .heterocycles import orient_heterocycle
        _, atom_to_locant = orient_heterocycle(mol, ring_list)
        return atom_to_locant.get(sub_atom, 1)

    # Carbocyclic: number from the connection atom (= locant 1)
    # Find position of connection atom and sub_atom in the ring adjacency
    # Build adjacency for the ring
    ring_set = set(ring_list)
    adj = {idx: [] for idx in ring_list}
    for bond in bonds_of(mol):
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in ring_set and a2 in ring_set:
            adj[a1].append(a2)
            adj[a2].append(a1)

    # Walk from connection_atom in both directions
    def walk(start, direction_neighbor):
        """Walk ring from start through direction_neighbor, return ordered atoms.

        Only step to UNVISITED neighbours: the old ``or n == start`` clause let
        the walk pick the start atom back as ``nexts[0]`` on the second step and
        terminate prematurely, so the short path to a META substituent was never
        traversed (it returned the long-way locant 5 instead of the correct 3
        for '[1,1'-biphenyl]-3-…'). The loop now enumerates every ring atom once
        in ring order and stops when no unvisited neighbour remains."""
        visited = [start]
        current = direction_neighbor
        while current != start:
            visited.append(current)
            nexts = [n for n in adj[current] if n not in visited]
            if not nexts:
                break
            current = nexts[0]
        return visited

    neighbors_of_conn = adj.get(connecting_atom, [])
    if len(neighbors_of_conn) < 2:
        # Shouldn't happen for a ring atom, fallback
        if connecting_atom == sub_atom:
            return 1
        return 2

    path_a = walk(connecting_atom, neighbors_of_conn[0])
    path_b = walk(connecting_atom, neighbors_of_conn[1])

    # Determine locant of sub_atom in each direction
    locant_a = path_a.index(sub_atom) + 1 if sub_atom in path_a else ring_size
    locant_b = path_b.index(sub_atom) + 1 if sub_atom in path_b else ring_size

    # Return the lower locant (IUPAC lowest locant rule)
    return min(locant_a, locant_b)


def _reassign_carbocyclic_locants(mol, ring_systems, connections, substituents):
    """: number each CARBOCYCLIC ring system ONCE (connection atom = 1),
    choosing the single direction giving the lowest locant SET to ALL its
    substituents together.

    :func:`_get_substituent_locant` picks each substituent's own min-locant
    direction independently; when two substituents sit on the SAME ring that
    collides (3,5-disubstituted benzene ring -> each mins to 3 -> the invalid
    '3,3' set, which OPSIN rejects). Choosing one shared direction per ring makes
    the set '3,5'. Heterocyclic rings keep their fixed orient_heterocycle
    numbering (handled in _get_substituent_locant) and are skipped here."""
    ri = mol.GetRingInfo()
    by_sys: Dict[int, List[Dict]] = {}
    for s in substituents:
        by_sys.setdefault(s['system_idx'], []).append(s)
    for sys_idx, subs in by_sys.items():
        if len(subs) < 2:
            continue  # single substituent: per-atom min is already correct
        sys_atoms = ring_systems[sys_idx]
        # ALL inter-system junction atoms for this system. A MIDDLE ring in a
        # linear assembly has TWO junctions; either may be numbered locant 1
        # when the flanking rings are equivalent, so both must be enumerated
        # as origins for a deterministic, lowest-locant choice. Using only the
        # first junction made the disubstituted-middle-ring numbering
        # SMILES-order dependent (2'-carboxy-3'-chloro vs 3'-carboxy-2'-chloro;
        # #41, the >=2-substituent sibling of the single-substituent fix in
        # _get_substituent_info).
        conn_atoms = []
        for a1, a2, s1, s2 in connections:
            if s1 == sys_idx:
                conn_atoms.append(a1)
            elif s2 == sys_idx:
                conn_atoms.append(a2)
        if not conn_atoms:
            continue
        ring = None
        for r in ri.AtomRings():
            if conn_atoms[0] in r and set(r) <= sys_atoms:
                ring = r
                break
        if ring is None:
            continue
        # carbocyclic only (heteroatom rings keep orient_heterocycle numbering)
        if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring):
            continue
        rset = set(ring)
        adj: Dict[int, List[int]] = {i: [] for i in ring}
        for b in bonds_of(mol):
            x, y = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if x in rset and y in rset:
                adj[x].append(y)
                adj[y].append(x)

        def _positions(conn, first):
            visited = [conn]
            cur = first
            while cur != conn:
                visited.append(cur)
                nxt = [n for n in adj[cur] if n not in visited]
                if not nxt:
                    break
                cur = nxt[0]
            return {a: i + 1 for i, a in enumerate(visited)}

        # Every candidate numbering: each junction origin x each ring
        # direction. Choose by IUPAC: (1) lowest locant SET to the
        # substituents together, then (2) lowest locant to the
        # substituent cited first in alphanumerical order.
        candidates = []
        for conn in conn_atoms:
            for nb in adj.get(conn, []):
                candidates.append(_positions(conn, nb))
        if not candidates:
            continue
        big = len(ring) + 1
        from ..assembly.naming_utils import alpha_sort_key
        subs_alpha = sorted(subs, key=lambda s: alpha_sort_key(s.get('name') or ''))

        def _rank(p):
            return (
                sorted(p.get(s['ring_atom'], big) for s in subs),
                [p.get(s['ring_atom'], big) for s in subs_alpha],
            )

        chosen = min(candidates, key=_rank)
        for s in subs:
            if s['ring_atom'] in chosen:
                s['locant'] = chosen[s['ring_atom']]


def _get_substituent_info(
    mol, ring_systems: List[Set[int]], connections: List[Tuple[int, int, int, int]]
) -> List[Dict]:
    """
    Find substituents on ring assembly systems and determine their locants.

    Args:
        mol: RDKit Mol object
        ring_systems: List of ring system atom sets
        connections: Inter-system bonds

    Returns:
        List of dicts with:
          - system_idx: which ring system the substituent is on
          - ring_atom: atom in ring where substituent attaches
          - sub_atoms: list of atom indices in substituent
          - name: substituent name (e.g., "chloro", "methyl")
          - locant: IUPAC locant on the ring
    """
    all_ring_atoms = set()
    for sys_atoms in ring_systems:
        all_ring_atoms.update(sys_atoms)

    # Connection atoms (not substituents)
    connection_atoms = set()
    for a1, a2, _, _ in connections:
        connection_atoms.add(a1)
        connection_atoms.add(a2)

    # Build atom -> system index
    atom_to_system = {}
    for sys_idx, sys_atoms in enumerate(ring_systems):
        for atom_idx in sys_atoms:
            atom_to_system[atom_idx] = sys_idx

    substituents = []
    visited_sub_atoms = set()

    for atom_idx in all_ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        sys_idx = atom_to_system[atom_idx]

        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in all_ring_atoms or n_idx in visited_sub_atoms:
                continue

            # This is a substituent atom
            # Walk the substituent to find all atoms
            sub_atoms = _walk_substituent(mol, n_idx, all_ring_atoms)
            visited_sub_atoms.update(sub_atoms)

            # Name the substituent
            sub_name = _name_substituent(mol, sub_atoms, n_idx)

            # Get the inter-system connection atoms for this ring system.
            # A MIDDLE ring in a linear assembly (e.g. terphenyl) has TWO
            # junctions; either may be numbered locant 1 when the flanking
            # rings are equivalent, so the substituent's lowest achievable
            # locant is the min over all junction origins. Using only the
            # FIRST connection made the locant SMILES-order-dependent
            # (2' vs 3'; #41, a review a998bea712). Taking the min is both
            # deterministic and the IUPAC lowest-locant choice.
            conn_atoms = []
            for a1, a2, s1, s2 in connections:
                if s1 == sys_idx:
                    conn_atoms.append(a1)
                elif s2 == sys_idx:
                    conn_atoms.append(a2)

            if not conn_atoms:
                conn_atoms = [atom_idx]

            locant = min(
                _get_substituent_locant(
                    mol, atom_idx, atom_idx, ring_systems[sys_idx], c
                )
                for c in conn_atoms
            )

            substituents.append({
                'system_idx': sys_idx,
                'ring_atom': atom_idx,
                'sub_atoms': sub_atoms,
                'name': sub_name,
                'locant': locant,
            })

    # set-lowest numbering for rings bearing >=2 substituents (fixes the
    # per-substituent min collision, e.g. 3,5-disubstituted -> '3,3').
    _reassign_carbocyclic_locants(mol, ring_systems, connections, substituents)
    return substituents


def _walk_substituent(mol, start_idx: int, ring_atoms: Set[int]) -> List[int]:
    """Walk from start_idx through non-ring atoms to collect substituent."""
    visited = set()
    stack = [start_idx]
    result = []

    while stack:
        current = stack.pop()
        if current in visited or current in ring_atoms:
            continue
        visited.add(current)
        result.append(current)

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in visited and n_idx not in ring_atoms:
                stack.append(n_idx)

    return result


def _name_substituent(mol, sub_atoms: List[int],
                      attachment_atom: int) -> Optional[str]:
    """
    Name a substituent attached to a ring assembly.

    Handles common cases: halogens, simple alkyl groups.

    Args:
        mol: RDKit Mol object
        sub_atoms: List of atom indices in substituent
        attachment_atom: First atom of the substituent (bonded to ring)

    Returns:
        Substituent name (prefix form, e.g., "chloro", "methyl")
    """
    from ..assembly.naming_utils import get_alkyl_name

    # free-valence gate. Every route below names the fragment from its
    # ATOM COUNT, which cannot distinguish -CH3 from =CH2:
    # 'C=C1CCC(C2CCCCC2)CC1' came out as 4-methyl-1,1'-bi(cyclohexane), a
    # different molecule. Shared primitive, same three-way contract as the two
    # general chokepoints; a single bond defers and changes nothing.
    from ..assembly.substituent_enumerator import carbon_free_valence_prefix
    from ..data.chain_names import get_chain_prefix

    _fv = carbon_free_valence_prefix(mol, sub_atoms, attachment_atom)
    if _fv.prefix is not None:
        return _fv.prefix
    if _fv.must_fail_closed:
        return None

    if len(sub_atoms) == 1:
        atom = mol.GetAtomWithIdx(sub_atoms[0])
        symbol = atom.GetSymbol()

        # Halogens
        halogen_names = {
            'F': 'fluoro',
            'Cl': 'chloro',
            'Br': 'bromo',
            'I': 'iodo',
        }
        if symbol in halogen_names:
            return halogen_names[symbol]

        # Single non-halogen atoms
        if symbol == 'O' and atom.GetTotalNumHs() >= 1:
            return 'hydroxy'
        if symbol == 'N' and atom.GetTotalNumHs() >= 2:
            return 'amino'

    # Check for alkyl groups (all carbon + hydrogen only)
    all_c_h = all(
        mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H') for i in sub_atoms
    )
    if all_c_h and sub_atoms:
        n_carbons = sum(
            1 for i in sub_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if n_carbons > 0:
            # Try recursive naming (handles retained names + branched subs)
            if len(sub_atoms) > 1:
                from ..assembly.substituent_naming import name_substituent_fragment
                rec_name = name_substituent_fragment(
                    mol, sub_atoms, sub_atoms[0], []
                )
                if rec_name:
                    return rec_name
            try:
                return get_alkyl_name(n_carbons)
            except (ValueError, KeyError):
                prefix = get_chain_prefix(n_carbons)
                return f"{prefix}yl"

    # Check for alkoxy groups (O + alkyl chain)
    if sub_atoms:
        first_atom = mol.GetAtomWithIdx(attachment_atom)
        if first_atom.GetSymbol() == 'O' and attachment_atom in sub_atoms:
            other_atoms = [i for i in sub_atoms if i != attachment_atom]
            carbon_count = sum(
                1 for i in other_atoms
                if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
            )
            all_simple = all(
                mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H')
                for i in other_atoms
            )
            if all_simple and carbon_count > 0:
                _ALKOXY = {1: 'methoxy', 2: 'ethoxy', 3: 'propoxy',
                           4: 'butoxy', 5: 'pentyloxy'}
                if carbon_count in _ALKOXY:
                    return _ALKOXY[carbon_count]
                prefix = get_chain_prefix(carbon_count)
                return f"{prefix}oxy"

    # Explicit -C(=O)OH recognizer (Wave-2 completion): the recursive
    # fallback below mis-prefixed a carboxyl as 'formyl' (the -OH oxygen was
    # dropped in the fragment round-trip — a wrong name the RT gate had to
    # suppress).: the prefix for -COOH is 'carboxy'.
    if _is_carboxyl_substituent(mol, sub_atoms, attachment_atom):
        return 'carboxy'

    # Fallback: use recursive naming via name_fragment_recursively
    # This handles compound substituents (C + heteroatoms) on ring assemblies,
    # such as COOH, CONH2, CHO, etc., that the simple patterns above miss.
    if sub_atoms and len(sub_atoms) <= 25:
        try:
            frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sub_atoms)
            if frag_smiles:
                from ..assembly.fragment_naming import name_fragment_recursively
                from ..assembly.substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix
                frag_name = name_fragment_recursively(frag_smiles)
                if frag_name:
                    carbon_count = sum(
                        1 for i in sub_atoms
                        if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                    )
                    prefix = parent_to_prefix(
                        frag_name, chain_length=carbon_count,
                        attach_locant=ATTACH_LOCANT_UNKNOWN)
                    if prefix:
                        return prefix
        except Exception:
            pass

    # Last resort: FAIL CLOSED. This used to return the literal word
    # "substituent", which is not a nomenclature term at all -- it named
    # nothing and merely deferred the failure to whoever spelled it into a
    # name. made ``parent_to_prefix`` legitimately return None for a
    # functional parent that has no '-yl' form, which put this branch back on
    # the execution path, so the placeholder is now reachable rather than
    # theoretical. Returning None makes every caller abstain: a name we
    # cannot construct must not be emitted); ``_get_substituent_info`` and
    # ``name_ring_assembly`` propagate the abstention.
    return None


def _is_carboxyl_substituent(mol, sub_atoms: List[int],
                             attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -C(=O)OH group attached via
    its carbon: {C, =O (degree 1), -OH (degree 1)}."""
    if len(sub_atoms) != 3:
        return False
    c = mol.GetAtomWithIdx(attachment_atom)
    if c.GetSymbol() != 'C' or c.GetFormalCharge() != 0:
        return False
    others = [i for i in sub_atoms if i != attachment_atom]
    if len(others) != 2:
        return False
    oxo = oh = None
    for i in others:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != 'O' or a.GetDegree() != 1 or a.GetFormalCharge():
            return False
        bond = mol.GetBondBetweenAtoms(attachment_atom, i)
        if bond is None:
            return False
        if bond.GetBondType() == Chem.BondType.DOUBLE and a.GetTotalNumHs() == 0:
            oxo = i
        elif bond.GetBondType() == Chem.BondType.SINGLE and a.GetTotalNumHs() == 1:
            oh = i
    return oxo is not None and oh is not None


def _is_formyl_substituent(mol, sub_atoms: List[int],
                           attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -CHO (formyl) group attached
    via its carbon: {C(H)(=O)}. Named as the added-carbon suffix -carbaldehyde
     on a ring assembly."""
    if len(sub_atoms) != 2:
        return False
    c = mol.GetAtomWithIdx(attachment_atom)
    if (c.GetSymbol() != 'C' or c.GetFormalCharge() != 0
            or c.GetTotalNumHs() != 1):
        return False
    others = [i for i in sub_atoms if i != attachment_atom]
    if len(others) != 1:
        return False
    o = mol.GetAtomWithIdx(others[0])
    if (o.GetSymbol() != 'O' or o.GetDegree() != 1 or o.GetFormalCharge()
            or o.GetTotalNumHs() != 0):
        return False
    bond = mol.GetBondBetweenAtoms(attachment_atom, others[0])
    return bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE


def _is_cyano_substituent(mol, sub_atoms: List[int],
                          attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -C#N (cyano) group attached via
    its carbon. Named as the added-carbon suffix -carbonitrile on a
    ring assembly."""
    if len(sub_atoms) != 2:
        return False
    c = mol.GetAtomWithIdx(attachment_atom)
    if (c.GetSymbol() != 'C' or c.GetFormalCharge() != 0
            or c.GetTotalNumHs() != 0):
        return False
    others = [i for i in sub_atoms if i != attachment_atom]
    if len(others) != 1:
        return False
    n = mol.GetAtomWithIdx(others[0])
    if (n.GetSymbol() != 'N' or n.GetDegree() != 1 or n.GetFormalCharge()
            or n.GetTotalNumHs() != 0):
        return False
    bond = mol.GetBondBetweenAtoms(attachment_atom, others[0])
    return bond is not None and bond.GetBondType() == Chem.BondType.TRIPLE


def _is_primary_amine_substituent(mol, sub_atoms: List[int],
                                  attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral primary -NH2 attached directly
    to the ring. Named as the suffix -amine."""
    if len(sub_atoms) != 1 or sub_atoms[0] != attachment_atom:
        return False
    n = mol.GetAtomWithIdx(attachment_atom)
    return (n.GetSymbol() == 'N' and n.GetFormalCharge() == 0
            and n.GetDegree() == 1 and n.GetTotalNumHs() == 2)


def _is_hydroxy_substituent(mol, sub_atoms: List[int],
                            attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -OH attached directly to the
    ring. Named as the suffix -ol."""
    if len(sub_atoms) != 1 or sub_atoms[0] != attachment_atom:
        return False
    o = mol.GetAtomWithIdx(attachment_atom)
    return (o.GetSymbol() == 'O' and o.GetFormalCharge() == 0
            and o.GetDegree() == 1 and o.GetTotalNumHs() == 1)


def _ring_assembly_pcg_suffix(mol, sub_atoms: List[int],
                              attachment_atom: int) -> Optional[str]:
    """Classify a ring-assembly substituent as a PRINCIPAL characteristic group
    expressible as a SUFFIX on the enclosed assembly parent + the
    suffix table). Returns the suffix stem, or None if the group is not a
    single-kind suffix-expressible PCG handled here.

      -C(=O)OH -> 'carboxylic acid' (added exocyclic C)
      -CHO -> 'carbaldehyde' (added exocyclic C)
      -C#N -> 'carbonitrile' (added exocyclic C)
      -NH2 -> 'amine' (direct ring attachment)
      -OH -> 'ol' (direct ring attachment)
    """
    if _is_carboxyl_substituent(mol, sub_atoms, attachment_atom):
        return 'carboxylic acid'
    if _is_formyl_substituent(mol, sub_atoms, attachment_atom):
        return 'carbaldehyde'
    if _is_cyano_substituent(mol, sub_atoms, attachment_atom):
        return 'carbonitrile'
    if _is_primary_amine_substituent(mol, sub_atoms, attachment_atom):
        return 'amine'
    if _is_hydroxy_substituent(mol, sub_atoms, attachment_atom):
        return 'ol'
    return None


# Suffix-expressible ring-assembly PCG kind -> its detachable PREFIX form, used
# when a group is DEMOTED (a more-senior PCG on the same assembly takes the
# suffix). All RT-verified against OPSIN 2.9 carboxy, aldehyde->formyl,
# nitrile->cyano, hydroxy, amino).
_RING_ASSEMBLY_PCG_PREFIX = {
    'carboxylic acid': 'carboxy',
    'carbaldehyde': 'formyl',
    'carbonitrile': 'cyano',
    'amine': 'amino',
    'ol': 'hydroxy',
}

# Suffix-expressible ring-assembly PCG kind -> its rules.seniority key (so the
# senior group is picked with the SHARED seniority order, never a hand-coded one).
_RING_ASSEMBLY_PCG_SENIORITY_KEY = {
    'carboxylic acid': 'carboxylic_acid',
    'carbonitrile': 'nitrile',
    'carbaldehyde': 'aldehyde',
    'ol': 'alcohol',
    'amine': 'primary_amine',
}


def _is_nitro_substituent(mol, sub_atoms: List[int],
                          attachment_atom: int) -> bool:
    """True iff *sub_atoms* is a nitro group -NO2 attached via its nitrogen
    (matches both the [N+](=O)[O-] and neutral N(=O)=O drawings). Prefix
    'nitro'."""
    if len(sub_atoms) != 3:
        return False
    n = mol.GetAtomWithIdx(attachment_atom)
    if n.GetSymbol() != 'N':
        return False
    oxygens = [i for i in sub_atoms if i != attachment_atom]
    if len(oxygens) != 2:
        return False
    return all(
        mol.GetAtomWithIdx(i).GetSymbol() == 'O'
        and mol.GetAtomWithIdx(i).GetDegree() == 1
        for i in oxygens
    )


def _mixed_ring_assembly_prefix_name(
    mol, sub_atoms: List[int], attachment_atom: int, pcg_kind: Optional[str]
) -> Optional[str]:
    """Prefix name for a NON-principal substituent in the mixed prefix+suffix
    ring-assembly builder. Fails CLOSED (returns None) for anything outside the
    supported carbocyclic-biaryl class, so the caller abstains rather than ship
    a garbage/dropped prefix (the local ``_name_substituent`` can still return an
    ``"unknown organic compound"`` mangle for groups it cannot name -- a no-Java
    leak. Its former ``"substituent"`` placeholder is gone: made that
    branch return None, so an unnameable group now fails closed at the source).

    Supported: demoted suffix-expressible PCG kinds (carboxy/formyl/cyano/
    amino/hydroxy), nitro, single halogen, all-C/H alkyl, and O-attached alkoxy.
    Every suffix-expressible group is one of the recognised PCG kinds, so a group
    MORE senior than the chosen suffix can only reach here via an unrecognised
    heteroatom pattern -> None -> abstain (never a wrong suffix)."""
    if pcg_kind is not None:
        return _RING_ASSEMBLY_PCG_PREFIX.get(pcg_kind)  # None -> abstain
    if _is_nitro_substituent(mol, sub_atoms, attachment_atom):
        return 'nitro'
    # Single halogen.
    if len(sub_atoms) == 1:
        sym = mol.GetAtomWithIdx(sub_atoms[0]).GetSymbol()
        halo = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
        if sym in halo:
            return halo[sym]
    # All-carbon/hydrogen alkyl.
    if all(mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H') for i in sub_atoms):
        nm = _name_substituent(mol, sub_atoms, attachment_atom)
        if nm and 'unknown' not in nm:
            return nm
        return None
    # O-attached alkoxy (O then only C/H).
    a = mol.GetAtomWithIdx(attachment_atom)
    if a.GetSymbol() == 'O' and attachment_atom in sub_atoms:
        rest = [i for i in sub_atoms if i != attachment_atom]
        if rest and all(
            mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H') for i in rest
        ):
            nm = _name_substituent(mol, sub_atoms, attachment_atom)
            if nm and 'unknown' not in nm:
                return nm
    return None


def _build_mixed_pcg_ring_assembly(
    mol, substituent_list: List[Dict], pcg_kinds: List[Optional[str]],
    ring_systems: List[Set[int]], connections: List[Tuple[int, int, int, int]],
    ring_name: str, multiplier: str, connection_str: str, count: int,
) -> Optional[str]:
    """Name a 2-ring CARBOCYCLIC assembly (biphenyl class) that bears a
    suffix-expressible PCG mixed with other substituents +.

    The most SENIOR PCG (via rules.seniority) is expressed as the assembly
    SUFFIX on the enclosed parent; every other substituent becomes a detachable
    PREFIX. The PCG-bearing ring is numbered UNPRIMED so the suffix takes the
    lowest locant; the numbering direction of each ring is chosen
    by first-point-of-difference on (suffix locants, then all substituent
    locants, then the alphabetically-first prefix). PIN form places NO hyphen
    between the prefix block and the opening bracket (`****`, BB 6968:
    no hyphen after a numerical prefix before an enclosing mark). The witness
    6,6'-dinitro[1,1'-biphenyl]-2,2'-dicarboxylic acid is real but lives at BB
    49803 under (axial chirality) -- it is NOT a example, and
     contains no biphenyl.

    Returns the PIN, or None to fail CLOSED (out-of-class / un-nameable prefix),
    letting ``name_ring_assembly`` fall through to its veto / prefix-only path.
    """
    from ..assembly.naming_utils import alpha_sort_key, get_multiplier_prefix
    from .seniority import compare_seniority

    # Scope: exactly a 2-component, single-bond, all-carbon single-ring assembly.
    if count != 2 or len(connections) != 1 or len(ring_systems) != 2:
        return None
    present = [k for k in pcg_kinds if k is not None]
    if not present:
        return None  # no suffix-expressible PCG -> not this builder's job

    ri = mol.GetRingInfo()
    a1, a2, s1, s2 = connections[0]
    junction = {s1: a1, s2: a2}
    ring_of: Dict[int, Tuple[int, ...]] = {}
    for sidx, satoms in enumerate(ring_systems):
        rings = [r for r in ri.AtomRings() if set(r) <= set(satoms)]
        if len(rings) != 1:
            return None  # fused component -> out of class
        ring = rings[0]
        if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring):
            return None  # heterocyclic assembly numbers junction != 1 -> abstain
        if sidx not in junction or junction[sidx] not in ring:
            return None
        ring_of[sidx] = ring

    # Senior PCG kind (shared seniority order; lower index = more senior).
    senior = present[0]
    for k in present[1:]:
        if compare_seniority(
            _RING_ASSEMBLY_PCG_SENIORITY_KEY[k],
            _RING_ASSEMBLY_PCG_SENIORITY_KEY[senior],
        ) < 0:
            senior = k

    # Partition; compute prefix names now (abstain on any un-nameable group).
    suffix_subs, prefix_subs = [], []
    for s, k in zip(substituent_list, pcg_kinds):
        if k == senior:
            suffix_subs.append(s)
        else:
            nm = _mixed_ring_assembly_prefix_name(
                mol, s['sub_atoms'], s['sub_atoms'][0], k)
            if nm is None:
                return None  # un-nameable substituent -> fail closed
            prefix_subs.append((s, nm))
    if not suffix_subs:
        return None

    # Per-ring position maps (junction = locant 1; both walk directions).
    def _position_maps(ring: Tuple[int, ...], conn: int) -> List[Dict[int, int]]:
        rset = set(ring)
        adj: Dict[int, List[int]] = {i: [] for i in ring}
        for b in mol.GetBonds():
            x, y = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if x in rset and y in rset:
                adj[x].append(y)
                adj[y].append(x)
        maps: List[Dict[int, int]] = []
        for first in adj[conn]:
            visited = [conn]
            cur = first
            while cur != conn:
                visited.append(cur)
                nxt = [nn for nn in adj[cur] if nn not in visited]
                if not nxt:
                    break
                cur = nxt[0]
            if len(visited) == len(ring):
                maps.append({a: i + 1 for i, a in enumerate(visited)})
        return maps or [{a: i + 1 for i, a in enumerate(ring)}]

    pos_maps = {
        sidx: _position_maps(ring_of[sidx], junction[sidx])
        for sidx in ring_of
    }

    # Search priming (which system is unprimed) x direction per ring; pick the
    # assignment with lowest (suffix keys, all-substituent keys, alpha-first).
    # A locant key is (number, prime_count): unprimed (0) < primed (1) at equal
    # number, so plain tuple sort implements the lowest-locant rule.
    best = None
    best_key = None
    systems = list(ring_of)  # [0, 1]
    for unprimed in systems:
        prime_of = {unprimed: 0, [x for x in systems if x != unprimed][0]: 1}
        for m0 in pos_maps[systems[0]]:
            for m1 in pos_maps[systems[1]]:
                pmap = {systems[0]: m0, systems[1]: m1}

                def _key(s):
                    sy = s['system_idx']
                    return (pmap[sy][s['ring_atom']], prime_of[sy])

                suffix_keys = sorted(_key(s) for s in suffix_subs)
                all_keys = sorted(
                    [_key(s) for s in suffix_subs]
                    + [_key(s) for s, _ in prefix_subs]
                )
                alpha_first = None
                if prefix_subs:
                    first_sub = min(prefix_subs, key=lambda sn: alpha_sort_key(sn[1]))
                    alpha_first = _key(first_sub[0])
                cand = (suffix_keys, all_keys, alpha_first or (0, 0))
                if best_key is None or cand < best_key:
                    best_key = cand
                    best = (prime_of, pmap)

    prime_of, pmap = best

    def _loc(s):
        sy = s['system_idx']
        return pmap[sy][s['ring_atom']], prime_of[sy]

    # Suffix block.
    suffix_keys = sorted(_loc(s) for s in suffix_subs)
    suffix_loc_str = ",".join(
        f"{num}{_format_prime(pr)}" for num, pr in suffix_keys)
    smult = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}.get(len(suffix_keys))
    if smult is None:
        return None
    # (c) (BB 7623/7625): the terminal 'a' of a numerical multiplier is
    # elided before a vowel-initial suffix ('tetra'+'ol' -> 'tetrol',
    # 'tetra'+'amine' -> 'tetramine'); '-carboxylic acid'/'-carbaldehyde'/
    # '-carbonitrile' (consonant-initial) and 'di'/'tri' (no terminal 'a') are
    # left untouched. Route through the shared elision primitive rather than the
    # raw f-string so ``[1,1'-biphenyl]-2,4,4',6-tetrol`` (PIN) is emitted.
    from ..assembly.naming_utils import _join_multiplied_suffix
    core = (f"[{connection_str}-{multiplier}{ring_name}]"
            f"-{suffix_loc_str}-{_join_multiplied_suffix(smult, senior)}")

    # Prefix block (alphanumerical, grouped multipliers). Directly abuts '[' with
    # NO hyphen: no hyphen before an opening enclosing mark).
    by_name: Dict[str, List[Tuple[int, int]]] = {}
    for s, nm in prefix_subs:
        by_name.setdefault(nm, []).append(_loc(s))
    prefix_parts = []
    for nm in sorted(by_name, key=alpha_sort_key):
        keys = sorted(by_name[nm])
        loc_str = ",".join(f"{num}{_format_prime(pr)}" for num, pr in keys)
        if len(keys) > 1:
            prefix_parts.append(f"{loc_str}-{get_multiplier_prefix(len(keys), nm)}{nm}")
        else:
            prefix_parts.append(f"{loc_str}-{nm}")
    prefix_block = "-".join(prefix_parts)

    return f"{prefix_block}{core}" if prefix_block else core


def _format_prime(ring_index: int) -> str:
    """
    Format primed notation for a ring index.

    Ring 0 -> "" (unprimed)
    Ring 1 -> "'" (single prime)
    Ring 2 -> "''" (double prime)

    Uses ASCII apostrophe (U+0027) for OPSIN compatibility.
    """
    return "'" * ring_index


def name_ring_assembly(
    mol, assembly_info: Dict, features: Any
) -> Optional[str]:
    """
    Generate IUPAC name for a ring assembly.

    IUPAC: Ring assemblies are named with multiplicative prefixes
    (bi-, ter-, quater-) before the parent ring name, with connection
    locants using primed notation.

    Args:
        mol: RDKit Mol object
        assembly_info: Dict from detect_ring_assembly with keys:
            ring_systems, connections, count, ring_type
        features: MolecularFeatures object (for substituent context)

    Returns:
        Complete IUPAC name string, or None if naming fails.

    Examples:
        biphenyl -> "1,1'-biphenyl"
        2,2'-bipyridine -> "2,2'-bipyridine"
        4-chlorobiphenyl -> "4-chloro-1,1'-biphenyl"
    """
    #: skeletal-replacement ring assembly (mixed/single-heteroatom
    # same-skeleton monocycles) -> dedicated 'a'-nomenclature namer.
    if assembly_info.get('replacement'):
        return _name_replacement_ring_assembly(mol, assembly_info)

    ring_systems = assembly_info['ring_systems']
    connections = assembly_info['connections']
    count = assembly_info['count']

    # Get the multiplier prefix
    multiplier = ASSEMBLY_MULTIPLIERS.get(count)
    if multiplier is None:
        return None  # Unsupported assembly size

    # Determine parent ring name from the first system
    ring_name = _get_ring_parent_name(mol, ring_systems[0])
    if ring_name is None:
        return None

    # a phase: ring stems carrying an embedded indicated-H
    # descriptor (e.g. "phenanthridin-6(5H)-one", "acridin-9(10H)-one") need a
    # per-ring threading strategy the placement-subset fix doesn't implement;
    # fail closed (return None) so a higher-level fallback can attempt naming
    # rather than emit the buggy mid-string `2,2'-biphenanthridin-6(5H)-one`
    # form. Deeper rewrite is a phase Follow-up 12 main-thread territory.
    if _INDICATED_H_EMBEDDED_RE.search(ring_name):
        return None

    # Build connection locant string per IUPAC.
    # a phase-03 /: use per-system numbering so ter-/quater-
    # assemblies emit the correct middle-ring back-attachment locant
    # (4 for para-terphenyl, 6 for terpyridine, 5 for terthiophene).
    # The pre-151-03 path called _get_connection_locant per pair, which
    # always returned 1 for carbocyclic atoms — yielding the buggy
    # "1,1':1',1''-terphenyl" output captured in internal notes-C.md.
    per_system_locants = _compute_per_system_ring_locants(mol, assembly_info)

    def _lookup_locant(per_system, sys_idx, atom_idx, sys_atoms):
        """Locant lookup with safe fallback to legacy single-bond helper."""
        if (
            per_system is not None
            and 0 <= sys_idx < len(per_system)
            and atom_idx in per_system[sys_idx]
        ):
            return per_system[sys_idx][atom_idx]
        return _get_connection_locant(mol, atom_idx, sys_atoms)

    # Substituents on the assembly, computed once and reused below (the ylidene
    # branch, the citation-order tiebreak, and the prefix builder).
    substituent_list = _get_substituent_info(mol, ring_systems, connections)

    # ⚠ Do NOT hoist a ``s['name'] is None`` abstention to here. ``s['name']``
    # is ONE producer's opinion (``_name_substituent``), not the question
    # "is this group nameable". Two builders below name substituents WITHOUT
    # consulting it -- the single-kind PCG suffix branch, which expresses the
    # group as a SUFFIX and so legitimately needs no prefix form at all, and
    # ``_build_mixed_pcg_ring_assembly`` via
    # ``_mixed_ring_assembly_prefix_name`` (carboxy/formyl/cyano/amino/hydroxy,
    # nitro, halogen, alkyl, alkoxy). A blanket check here short-circuits both.
    # The abstention belongs where the prefix-only path actually consumes
    # ``s['name']`` -- see the veto loop below, which has fail-closed on a
    # missing name since c6b9f6ba. added the
    # hoisted duplicate and it cost the P2-RING-ASSEMBLY-MIXED gold target
    # (4'-nitro[1,1'-biphenyl]-4-carboxylic acid -> abstention): nitro reaches
    # ``_name_substituent`` first, gets None, and the assembly died before the
    # producer that spells 'nitro' ever ran.

    # (connection/"free valence" locant) + (lowest
    # locants to substituents): for a 2-component assembly, decide ONCE,
    # deterministically, which physical ring system is UNPRIMED by
    # first-point-of-difference on the COMBINED citation set the assembly
    # actually uses -- connection locants first (tier 1,, then
    # substituent-prefix locants (tier 2,. ``ring_systems``/
    # ``connections`` order coming out of ``detect_ring_assembly`` is RDKit
    # atom-index (SMILES-spelling) dependent, so without this the choice of
    # which ring is "system 0" (unprimed) was non-deterministic AND, whenever
    # a substituent broke the tie, sometimes wrong (a substituent on the ring
    # that happened to land at index 1 was cited with a prime it should never
    # carry -- "4'-chloro-1,1'-biphenyl" instead of the PIN
    # "4-chloro-1,1'-biphenyl"). Mirrors the analogous heteroatom
    # combined-locant-set tiebreak in ``_name_replacement_ring_assembly``
    #, generalised from heteroatoms to substituents. Relabeling
    # here -- BEFORE the connection string and substituent locants are built
    # below -- keeps every downstream consumer (the connection-locant loop,
    # the single-kind-suffix branch, the mixed prefix+suffix builder, and the
    # plain substituent-prefix branch) automatically consistent: they all key
    # off the same (now canonical) ``system_idx``.
    # (the Blue Book): per-component indicated-hydrogen locants, computed in
    # the assembly numbering, drive both the priming tiebreak below (indicated H
    # is senior to substituent prefixes, (b) ahead of (g)) and the
    # per-ring descriptor in the indicated-H branch further down.
    per_ring_ih_locants = _compute_per_ring_ih_locants(
        mol, ring_systems, per_system_locants)

    if count == 2 and len(ring_systems) == 2:
        def _combined_key(swap: bool):
            conn_key = []
            for a1, a2, s1, s2 in connections:
                for atom, sys in ((a1, s1), (a2, s2)):
                    eff_sys = (1 - sys) if swap else sys
                    loc = _lookup_locant(
                        per_system_locants, sys, atom, ring_systems[sys])
                    conn_key.append((loc, eff_sys))
            # / (b): indicated hydrogen is senior to substituent
            # prefixes when choosing which ring is unprimed. Only a tiebreak when
            # the assembly is unsubstituted (all our per-ring iH targets are), so
            # substituted rows keep their established (conn, subst) priming.
            ih_key = []
            if not substituent_list:
                for sys_idx, locs in enumerate(per_ring_ih_locants):
                    if locs:
                        for loc in locs:
                            eff_sys = (1 - sys_idx) if swap else sys_idx
                            ih_key.append((loc, eff_sys))
            sub_key = []
            for s in substituent_list:
                eff_sys = (1 - s['system_idx']) if swap else s['system_idx']
                sub_key.append((s['locant'], eff_sys))
            return (sorted(conn_key), sorted(ih_key), sorted(sub_key))

        if _combined_key(True) < _combined_key(False):
            ring_systems = [ring_systems[1], ring_systems[0]]
            connections = [(a1, a2, 1 - s1, 1 - s2)
                           for (a1, a2, s1, s2) in connections]
            if per_system_locants is not None and len(per_system_locants) == 2:
                per_system_locants = [per_system_locants[1], per_system_locants[0]]
            per_ring_ih_locants = [per_ring_ih_locants[1], per_ring_ih_locants[0]]
            for s in substituent_list:
                s['system_idx'] = 1 - s['system_idx']

    # Sort connections by system indices to ensure consistent ordering.
    # For bi- assemblies: one connection -> "X,X'"
    # For ter- assemblies: two connections -> "X,X':X',X''"
    connection_parts = []
    sorted_connections = []
    for a1, a2, s1, s2 in connections:
        if s1 > s2:
            a1, a2, s1, s2 = a2, a1, s2, s1
        sorted_connections.append((a1, a2, s1, s2))
    sorted_connections.sort(key=lambda c: (c[2], c[3]))
    for a1, a2, s1, s2 in sorted_connections:
        loc1 = _lookup_locant(per_system_locants, s1, a1, ring_systems[s1])
        loc2 = _lookup_locant(per_system_locants, s2, a2, ring_systems[s2])
        prime1 = _format_prime(s1)
        prime2 = _format_prime(s2)
        # IUPAC + (8 Oct 2025 erratum): "lowest locants...
        # then order of citation" — the unprimed (first-cited) ring takes the
        # lower attachment locant. For a 2-component assembly of identical rings
        # with no distinguishing substituent, get_ring_systems order is
        # atom-index (SMILES-spelling) dependent, so without this tiebreak the
        # asymmetric case is order-dependent (2,3'-bifuran vs 3,2'-bifuran).
        # Relabeling the unprimed ring is valid only when no substituent
        # differentiates the two identical rings.
        if count == 2 and not substituent_list and loc2 < loc1:
            loc1, loc2 = loc2, loc1
        connection_parts.append(f"{loc1}{prime1},{loc2}{prime2}")

    connection_str = ":".join(connection_parts)

    # IUPAC: a double-bond junction is named with the ylidene
    # substituent-group form (method 2). Two classes: the saturated-carbocycle
    # ring, enclosed in parentheses to avoid confusion with von Baeyer names
    # (1,1'-bi(cyclopentylidene)); and the mancude heterocyclic ring, named on its
    # mancude parent stem with per-component indicated hydrogen
    # (2'H,3H-2,3'-bifuranylidene). Fail closed (return None -> prior naming) if
    # the component carries substituents or is neither class (no wrong names).
    if assembly_info.get("double_bond_junction"):
        if substituent_list:
            return None
        ylidene = _to_ylidene(ring_name)
        if ylidene is not None:
            return f"{connection_str}-{multiplier}{_enclose_component(ylidene)}"
        # Mancude heterocyclic ylidene: stem is the MANCUDE parent
        # (furan, not the 2,3-dihydrofuran ring_name), with per-ring indicated
        # hydrogen recomputed at each component's saturated position. Both stems
        # must agree (identical rings) or fail closed.
        stems = [_mancude_ring_stem(mol, s) for s in ring_systems]
        if not stems[0] or any(s != stems[0] for s in stems):
            return None
        if any(locs is None for locs in per_ring_ih_locants):
            return None
        ih_tokens = sorted(
            (loc, sys_idx)
            for sys_idx, locs in enumerate(per_ring_ih_locants)
            for loc in locs
        )
        ih_prefix = (",".join(f"{loc}{_format_prime(sys_idx)}H"
                              for loc, sys_idx in ih_tokens) + "-"
                     ) if ih_tokens else ""
        return (f"{ih_prefix}{connection_str}-{multiplier}"
                f"{stems[0]}ylidene")

    # (the Blue Book) /: a SINGLE-bond assembly of the SAME mancude
    # parent in two indicated-hydrogen states (aromatic pyridine + its
    # N-substituted 2H tautomer -> 2H-1,2'-bipyridine). detect_ring_assembly
    # flags this 'mancude_tautomer'. Name it on the MANCUDE parent stem
    # (pyridine), NOT the per-component partially-saturated name that
    # _get_ring_parent_name returns for the actual ring (1,2-dihydropyridine),
    # which would emit the WRONG-molecule "1,2'-bi(1,2-dihydropyridine)". The
    # per-ring indicated hydrogen is recomputed at each component's saturated
    # position (2H on the 2H-pyridine ring, none on the aromatic ring). Mirrors
    # the ylidene mancude path above but for a single bond (no "ylidene"). Fail
    # closed to the established path if the mancude stems disagree or numbering is
    # missing (no wrong names); substituted rows are out of this narrow scope.
    if assembly_info.get("mancude_tautomer") and not substituent_list:
        stems = [_mancude_ring_stem(mol, s) for s in ring_systems]
        if (stems[0] and all(s == stems[0] for s in stems)
                and all(locs is not None for locs in per_ring_ih_locants)):
            ih_tokens = sorted(
                (loc, sys_idx)
                for sys_idx, locs in enumerate(per_ring_ih_locants)
                for loc in locs
            )
            ih_prefix = (",".join(f"{loc}{_format_prime(sys_idx)}H"
                                  for loc, sys_idx in ih_tokens) + "-"
                         ) if ih_tokens else ""
            return f"{ih_prefix}{connection_str}-{multiplier}{stems[0]}"

    # a phase.B: indicated-H placement subset for ring assemblies.
    # If ring_name carries an indicated-H prefix like "1H-indole", emit the
    # descriptor once per primed ring ("1H,1'H-2,2'-biindole") instead of
    # leaving it embedded inside the multiplied stem ("2,2'-bi1H-indole",
    # the buggy pre-fix output). The deeper assembly-builder rewrite
    # (per-ring indicated-H locants generically threaded into the assembly
    # base-name for biindole-class assemblies) stays in a phase's deferred-
    # warnings backlog (Follow-up 12 main thread).
    # Source: 155-internal notes; AUTONOM-followups.md Follow-up 12 placement
    # subset; IUPAC. Reuses _format_prime above.
    indicated_h_match = _INDICATED_H_RE.match(ring_name)
    if indicated_h_match:
        locant_int, ring_stem = (
            indicated_h_match.group(1),
            indicated_h_match.group(2),
        )
        # /: a COMPOUND component name (a skeletal-'a'
        # mancude heterocycle carrying its own heteroatom-locant set, e.g.
        # '4H-1,4-oxaphosphinine') is ENCLOSED in parentheses with its
        # indicated-H kept INSIDE, cited once — '4,4'-bi(4H-1,4-oxaphosphinine)'.
        # This differs from the biindole class (bare stem 'indole', no internal
        # locants), which front-replicates the indicated-H per primed ring
        # ("1H,1'H-2,2'-biindole",. Distinguish by whether the stem
        # itself carries a locant set (a comma-locant '<d>,<d>-' prefix).
        if re.match(r"^\d[\d,]*-", ring_stem):
            base_name = (
                f"{connection_str}-{multiplier}({ring_name})"
            )
        elif all(locs is not None for locs in per_ring_ih_locants):
            # IUPAC (the Blue Book) "Indicated hydrogen... in a ring
            # assembly is added... to each component ring as required": recompute
            # each SINGLE-RING component's indicated hydrogen at its own position
            # in the assembly numbering, rather than front-replicating the first
            # ring's descriptor. This is required for asymmetric assemblies
            # (``1H,3'H-4,4'-biazepine`` — the two rings differ) and to DROP a
            # spurious descriptor when the indicated-H atom is the inter-ring
            # attachment (``1,1'-bipyrrole``, cited with NO indicated H although
            # the isolated component is 1H-pyrrole;. ``per_ring_ih_locants``
            # already excludes attachment atoms and follows the priming chosen by
            # the tiebreak above. Fused components (any entry None) fall through to
            # the front-replicate branch, which stays correct for the symmetric
            # biindole/biindene class it was built for.
            ih_tokens = sorted(
                ((loc, sys_idx)
                 for sys_idx, locs in enumerate(per_ring_ih_locants)
                 for loc in locs),
            )
            if ih_tokens:
                ih_prefix = ",".join(
                    f"{loc}{_format_prime(sys_idx)}H"
                    for loc, sys_idx in ih_tokens
                ) + "-"
            else:
                ih_prefix = ""
            base_name = (
                f"{ih_prefix}{connection_str}-{multiplier}{ring_stem}"
            )
        else:
            indicated_h_replicated = ",".join(
                f"{locant_int}{_format_prime(i)}H" for i in range(count)
            ) + "-"
            base_name = (
                f"{indicated_h_replicated}{connection_str}-{multiplier}{ring_stem}"
            )
    else:
        # IUPAC: enclose the component in parentheses when needed to
        # avoid confusion with von Baeyer names (cycloalkanes / spiro / bicyclo);
        # mancude rings (phenyl/pyridine/furan/...) stay bare.
        base_name = f"{connection_str}-{multiplier}{_enclose_component(ring_name)}"

    if not substituent_list:
        return base_name

    # / / suffix table: a PRINCIPAL characteristic
    # group on a ring assembly must be expressed as a SUFFIX on the enclosed
    # assembly parent, never as a prefix. Added-carbon groups —
    # '[1,1'-biphenyl]-4,4'-dicarboxylic acid', '[1,1'-biphenyl]-4-carbaldehyde',
    # '[1,1'-biphenyl]-4-carbonitrile' — and direct groups —
    # '[1,1'-biphenyl]-4-amine', '[1,1'-biphenyl]-4,4'-diol' — all follow the
    # same enclosed-parent + locant + multiplied-suffix shape. Expressing the
    # PRINCIPAL characteristic group as a prefix would be wrong here regardless
    # of spelling — it is the suffix rule above that forces the enclosed-parent
    # shape, not any defect in the prefix vocabulary.
    # (Historical note, corrected Task 3: this comment used to justify
    # itself by claiming the generic namer emits 'formaldehydyl' / 'hydrogen
    # cyanidyl'. It did, but that was a defect in parent_to_prefix, now fixed —
    # CHO -> 'formyl' and HCN -> 'cyano'. Do not re-derive a nomenclature rule
    # from that former misbehaviour.)
    # Scope (fail-closed): EVERY substituent is the SAME suffix-expressible PCG;
    # mixed / other decorations keep the established prefix-only path below.
    _pcg_kinds = [
        _ring_assembly_pcg_suffix(mol, s['sub_atoms'], s['sub_atoms'][0])
        for s in substituent_list
    ]
    if _pcg_kinds and all(k is not None for k in _pcg_kinds) and \
            len(set(_pcg_kinds)) == 1:
        _suffix_stem = _pcg_kinds[0]
        suffix_pairs = [(s['locant'], s['system_idx'])
                        for s in substituent_list]
        # (c) determinism: the suffix takes the LOWEST locants -- the
        # ring carrying it must be the UNPRIMED one ([1,1'-biphenyl]-4-
        # carboxylic acid, never -4'-). get_ring_systems order is SMILES-
        # spelling dependent, so for a symmetric 2-ring connection (equal
        # attachment locants) relabel the priming when that lowers the
        # suffix locant set; asymmetric connections keep their citation-
        # order labels (relabelling would alter the connection locants).
        if count == 2 and sorted_connections:
            _a1, _a2, _s1, _s2 = sorted_connections[0]
            _l1 = _lookup_locant(per_system_locants, _s1, _a1,
                                 ring_systems[_s1])
            _l2 = _lookup_locant(per_system_locants, _s2, _a2,
                                 ring_systems[_s2])
            if _l1 == _l2:
                _swapped = [(loc, 1 - sys_idx) for loc, sys_idx in
                            suffix_pairs]
                if sorted(_swapped) < sorted(suffix_pairs):
                    suffix_pairs = _swapped
        suffix_locants = sorted(suffix_pairs)
        locant_str = ",".join(
            f"{loc}{_format_prime(sys_idx)}" for loc, sys_idx in suffix_locants
        )
        n = len(suffix_locants)
        mult = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}.get(n)
        if mult is None:
            return None
        if indicated_h_match:
            return None  # indicated-H + suffix threading not built
        # (c) (BB 7623/7625): elide the multiplier's terminal 'a' before a
        # vowel-initial suffix -- 'tetra'+'ol' -> 'tetrol', 'tetra'+'amine' ->
        # 'tetramine' ([1,1'-biphenyl]-3,3',4,4'-tetramine, PIN) -- while leaving
        # '-carboxylic acid'/'-carbaldehyde'/'-carbonitrile' and 'di'/'tri'
        # untouched. Route through the shared elision primitive, not a raw f-string.
        from ..assembly.naming_utils import _join_multiplied_suffix
        return (f"[{connection_str}-{multiplier}{ring_name}]"
                f"-{locant_str}-{_join_multiplied_suffix(mult, _suffix_stem)}")

    # +: MIXED prefix+suffix assembly (a suffix-expressible PCG
    # coexists with other substituents). The senior PCG becomes the suffix, the
    # rest become prefixes. Only fires for the 2-ring carbocyclic (biphenyl)
    # class; fails closed (returns None) otherwise, falling through to the veto /
    # prefix-only path below. Purely additive: upgrades wrong-PIN prefix names
    # ('4'-chloro-4-hydroxy-1,1'-biphenyl') and fail-closed 'unknown' cases to
    # the PIN, and closes the nitro gate-off leak; never regresses a working name.
    if any(k is not None for k in _pcg_kinds):
        _mixed = _build_mixed_pcg_ring_assembly(
            mol, substituent_list, _pcg_kinds, ring_systems, connections,
            ring_name, multiplier, connection_str, count)
        if _mixed is not None:
            return _mixed

    # Fail-closed veto: a MIXED -CHO / -C#N on a ring assembly.
    #
    # ⚠ Corrected Task 3 — this comment previously read "a -CHO / -C#N
    # on a ring assembly has NO valid prefix form". That is FALSE as a
    # nomenclature claim, and the Blue Book says so directly:
    # (the Blue Book, under ALDEHYDES): "... a -CHO group is
    # expressed by the preferred prefix 'oxo' if located at an end of a
    # carbon chain, or, otherwise, by the preferred prefix 'formyl'."
    # the Blue Book is a verbatim ring example: '4-formylcyclohexane-1-
    # carboxylic acid (PIN)'.
    # (the Blue Book): the -CN group "is designated by the preferred
    # prefix 'cyano'".
    # Both groups DO have valid preferred prefixes. What was true was the
    # empirical half: parent_to_prefix emitted 'formaldehydyl' / 'hydrogen
    # cyanidyl'. That was a defect in the converter, not a fact about IUPAC,
    # and it is fixed (see _FUNCTIONAL_PARENT_NO_YL_FORM in
    # assembly/substituent_naming.py).
    #
    # The veto nevertheless STAYS, for the reason stated in the next sentence
    # and only that reason: the single-kind suffix path above already consumed
    # the pure cases, so reaching here means the group is MIXED with others,
    # which needs the prefix+suffix ring-assembly builder (not yet built).
    # Refuse rather than ship a wrong name (accuracy #1).
    for _s in substituent_list:
        _a = _s['sub_atoms'][0]
        if (_is_formyl_substituent(mol, _s['sub_atoms'], _a)
                or _is_cyano_substituent(mol, _s['sub_atoms'], _a)):
            return None
        #: _name_substituent returns None when the attachment bond is
        # double/triple and no prefix spells that free valence. Refuse for the
        # same reason as the two cases above -- a name built around a missing
        # or single-valence token describes a different molecule.
        #
        # This is the ONE correct home for the missing-name abstention, and it
        # is load-bearing: it is the last statement before ``s['name']`` is
        # actually spelled into the prefix string below, and everything above
        # it (the suffix and mixed prefix+suffix builders) can still name a
        # group this producer could not. Do not hoist it -- see the note at
        # ``substituent_list = _get_substituent_info(...)``.
        if not _s.get('name'):
            return None

    # Build substituent prefix
    # Group by name for multipliers

    from ..assembly.naming_utils import (
        alpha_sort_key,
        get_multiplier_prefix,
    )

    # Sort substituents alphabetically by name
    substituent_list.sort(key=lambda s: alpha_sort_key(s['name']))

    # Group identical substituents. Locant/prime pairs are collected as
    # (locant, system_idx) keys -- NOT pre-formatted strings -- and sorted
    # numerically (ascending locant, unprimed before primed at equal locant)
    # before formatting. ``substituent_list`` iteration order upstream traces
    # back to a Python-set walk over ring atoms in ``_get_substituent_info``,
    # which is atom-index (SMILES-spelling) dependent; without this explicit
    # sort, two chemically-identical substituents (e.g. both ring chloro
    # atoms in 4,4'-dichlorobiphenyl) could be cited in either order,
    # producing non-deterministic "4,4'-dichloro-..." vs "4',4-dichloro-..."
    # output for the same molecule.
    sub_groups: Dict[str, List[Tuple[int, int]]] = {}
    for sub in substituent_list:
        name = sub['name']
        sub_groups.setdefault(name, []).append((sub['locant'], sub['system_idx']))

    # Build prefix parts
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        keys = sorted(sub_groups[name])
        n = len(keys)
        locant_str = ",".join(
            f"{loc}{_format_prime(sys_idx)}" for loc, sys_idx in keys)
        if n > 1:
            mult = get_multiplier_prefix(n, name)
            prefix_parts.append(f"{locant_str}-{mult}{name}")
        else:
            prefix_parts.append(f"{locant_str}-{name}")

    sub_prefix = "-".join(prefix_parts)

    return f"{sub_prefix}-{base_name}"


def name_ring_assembly_prefix(
    mol, assembly_info: Dict, attachment_atom_idx: int
) -> Optional[str]:
    """Generate ring assembly substituent prefix per IUPAC.

    When a ring assembly (identical rings joined by single bonds) appears as
    a substituent on a parent chain, the prefix uses square-bracket notation
    with primed connection locants and an attachment locant with -yl suffix.

    Format: ``[connection_locants-multiplierRing_name]-attach_locant-yl``

    Args:
        mol: RDKit Mol object
        assembly_info: Dict from detect_ring_assembly with keys:
            ring_systems, connections, count, ring_type
        attachment_atom_idx: Atom index where assembly connects to parent chain

    Returns:
        Prefix string like ``[1,1'-biphenyl]-4-yl`` or None on failure.

    Examples:
        biphenyl attached at para position -> "[1,1'-biphenyl]-4-yl"
        bipyridine attached at position 5 -> "[2,2'-bipyridin]-5-yl"
    """
    ring_systems = assembly_info['ring_systems']
    connections = assembly_info['connections']
    count = assembly_info['count']

    multiplier = ASSEMBLY_MULTIPLIERS.get(count)
    if multiplier is None:
        return None

    ring_name = _get_ring_parent_name(mol, ring_systems[0])
    if ring_name is None:
        return None

    # a phase: same fail-closed as in `name_ring_assembly` for embedded
    # descriptors. Both call sites must agree on the contract.
    if _INDICATED_H_EMBEDDED_RE.search(ring_name):
        return None

    # Build connection locant string per IUPAC, using PER-SYSTEM
    # numbering exactly as ``name_ring_assembly`` (the parent path) does. The
    # legacy per-pair ``_get_connection_locant`` returned 1 for every
    # carbocyclic connection atom, so a ter-/quater- assembly SUBSTITUENT
    # emitted the buggy '1,1':1',1''-terphenyl' (the middle ring's back-
    # attachment locant must be 4' for para-terphenyl). This bug was fixed on
    # the parent path (151-03 /) but the substituent-prefix path kept
    # the old code; a biphenyl (single junction) is unaffected either way.
    per_system_locants = _compute_per_system_ring_locants(mol, assembly_info)

    def _lookup_locant(sys_idx, atom_idx, sys_atoms):
        if (per_system_locants is not None
                and 0 <= sys_idx < len(per_system_locants)
                and atom_idx in per_system_locants[sys_idx]):
            return per_system_locants[sys_idx][atom_idx]
        return _get_connection_locant(mol, atom_idx, sys_atoms)

    sorted_connections = []
    for a1, a2, s1, s2 in connections:
        if s1 > s2:
            a1, a2, s1, s2 = a2, a1, s2, s1
        sorted_connections.append((a1, a2, s1, s2))
    sorted_connections.sort(key=lambda c: (c[2], c[3]))
    connection_parts = []
    for a1, a2, s1, s2 in sorted_connections:
        loc1 = _lookup_locant(s1, a1, ring_systems[s1])
        loc2 = _lookup_locant(s2, a2, ring_systems[s2])
        connection_parts.append(
            f"{loc1}{_format_prime(s1)},{loc2}{_format_prime(s2)}")
    connection_str = ":".join(connection_parts)

    # Find which ring system the attachment atom belongs to
    attach_system_idx = None
    for i, sys_atoms in enumerate(ring_systems):
        if attachment_atom_idx in sys_atoms:
            attach_system_idx = i
            break
    if attach_system_idx is None:
        return None

    # Free-valence (-yl) locant. Two a review BLOCKERs live here because the legacy
    # _get_substituent_locant numbered the attachment ring in ISOLATION:
    # (1) for a HETEROCYCLE it used orient_heterocycle ignorant of the junction
    # and placed the -yl on a JUNCTION atom ('[2,2'-bithiophen]-2-yl',
    # correct '5-yl'); and
    # (3) for a MULTI-RING FUSED component it emitted a PARSEABLE-WRONG name
    # ('[1,1'-binaphthalen]-2-yl' for a 2,2'-binaphthalene) that
    # masks — the producer must be honest WITHOUT the gate.
    # Fix, by attachment-ring kind (carbocyclic monocycle is UNCHANGED, so
    # '[1,1'-biphenyl]-3-yl' etc. stay byte-identical):
    # * fused / multi-ring -> fail closed (never guess, finding 3);
    # * heterocyclic monocycle -> the per-system heteroatom-priority map,
    # which numbers junction-aware and consistent with the connection string
    # (finding 1: bithiophene 5, bipyridine 5);
    # * carbocyclic monocycle -> the existing _get_substituent_locant,
    # which applies the lowest-free-valence-locant rule the
    # per-system connection-locant minimiser does not.
    _attach_sys_atoms = ring_systems[attach_system_idx]
    _attach_sys_rings = [
        r for r in mol.GetRingInfo().AtomRings() if set(r) <= _attach_sys_atoms]
    if len(_attach_sys_rings) != 1:
        return None  # fused / multi-ring attachment system -> fail closed (BLOCKER 3)
    _is_hetero_attach = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in _attach_sys_rings[0])

    inter_ring_conn_atom = None
    for a1, a2, s1, s2 in connections:
        if s1 == attach_system_idx:
            inter_ring_conn_atom = a1
            break
        elif s2 == attach_system_idx:
            inter_ring_conn_atom = a2
            break

    if _is_hetero_attach:
        if (per_system_locants is None
                or attach_system_idx >= len(per_system_locants)
                or attachment_atom_idx not in per_system_locants[attach_system_idx]):
            return None  # heteroatom ring not numbered -> fail closed (BLOCKER 1)
        attach_locant = per_system_locants[attach_system_idx][attachment_atom_idx]
    elif inter_ring_conn_atom is not None:
        attach_locant = _get_substituent_locant(
            mol, attachment_atom_idx, attachment_atom_idx,
            ring_systems[attach_system_idx], inter_ring_conn_atom
        )
    else:
        attach_locant = _get_connection_locant(
            mol, attachment_atom_idx, ring_systems[attach_system_idx]
        )
    attach_prime = _format_prime(attach_system_idx)

    # a phase.B: indicated-H placement subset for ring-assembly
    # SUBSTITUENT prefix path. Mirrors the parent-path replication in
    # `name_ring_assembly` (line 1201-1212) so a biindole-bearing
    # substituent emits `[1H,1'H-2,2'-biindol]-5-yl` instead of the pre-fix
    # buggy form `[2,2'-bi1H-indol]-5-yl`. Both call sites now agree on
    # the replication contract -- a biindole-bearing molecule names
    # consistently whether the assembly is parent or substituent.
    # Source: internal notes; AUTONOM-followups.md Follow-up 12
    # placement subset; IUPAC. Reuses _format_prime
    # and _INDICATED_H_RE.
    indicated_h_match = _INDICATED_H_RE.match(ring_name)
    if indicated_h_match:
        locant_int, ring_stem = (
            indicated_h_match.group(1),
            indicated_h_match.group(2),
        )
        indicated_h_replicated = ",".join(
            f"{locant_int}{_format_prime(i)}H" for i in range(count)
        ) + "-"
        # Vowel elision for the -yl form: "indole" -> "indol".
        display_stem = ring_stem[:-1] if ring_stem.endswith('e') else ring_stem
        # Assembly base: "1H,1'H-2,2'-biindol"
        assembly_base = (
            f"{indicated_h_replicated}{connection_str}-{multiplier}{display_stem}"
        )
    else:
        # For heterocyclic rings, apply vowel elision: "pyridine" -> "pyridin" before -yl
        # (IUPAC: terminal 'e' dropped before '-yl')
        display_name = ring_name
        if display_name.endswith('e'):
            display_name = display_name[:-1]

        #: a NON-retained component (cycloalkane / von Baeyer / spiro)
        # is enclosed in parentheses to disambiguate from a von Baeyer name,
        # EXACTLY as the parent path does via _enclose_component (line ~2168) —
        # 'cyclohexan' -> '(cyclohexan)', so '[1,1'-bi(cyclohexan)]-2-yl'
        # (the Blue Book preferred prefix), not the buggy '[1,1'-bicyclohexan]-2-yl'.
        # Retained mancude stems (phenyl/pyridin/thiophen) never match and stay
        # bare -> '[1,1'-biphenyl]-4-yl' byte-identical. The decision keys off the
        # ORIGINAL ring_name (the predicate matches 'cyclohexane', not the elided
        # stem); the marks wrap the elided display_name.
        if _needs_von_baeyer_parens(ring_name):
            display_name = f"({display_name})"

        # Assembly base: "1,1'-biphenyl", "2,2'-bipyridin", "1,1'-bi(cyclohexan)"
        assembly_base = f"{connection_str}-{multiplier}{display_name}"

    # Full prefix: "[1,1'-biphenyl]-4-yl" or "[1H,1'H-2,2'-biindol]-5-yl"
    return f"[{assembly_base}]-{attach_locant}{attach_prime}-yl"


def name_mixed_ring_prefix(
    mol,
    ring_systems_list: List[Set[int]],
    inter_system_bonds: List[Tuple[int, int, int, int]],
    attachment_atom_idx: int,
) -> Optional[str]:
    """Generate compound substituent prefix for non-identical connected rings.

    When two or more non-identical ring systems are connected by single bonds
    and appear as a substituent on a parent chain, the ring carrying the free
    valence (chain attachment) is the parent of the compound substituent prefix.
    The other ring(s) become simple substituents on it.

    For the parent ring of the compound prefix:
    - Carbocyclic: numbering gives chain attachment locant 1 (lowest locant rule)
    - Heterocyclic: standard IUPAC numbering (heteroatom at position 1)

    Args:
        mol: RDKit Mol object
        ring_systems_list: List of sets of atom indices, one per ring system
        inter_system_bonds: List of (atom_A, atom_B, system_A, system_B) tuples
            from _find_inter_system_bonds
        attachment_atom_idx: Atom index where the multi-ring fragment connects
            to the parent chain

    Returns:
        Compound prefix string like ``(4-(pyridin-2-yl)phenyl)`` or None.
    """
    if len(ring_systems_list) < 2:
        return None

    from .ring_substituents import get_ring_substituent_name, identify_ring_system

    # Determine which ring the chain attaches to -- that becomes the parent
    # of the compound substituent prefix (it carries the free valence = -yl)
    parent_idx = None
    sub_idx = None
    for i, sys_atoms in enumerate(ring_systems_list):
        if attachment_atom_idx in sys_atoms:
            parent_idx = i
            break
    if parent_idx is None:
        return None

    # For 2-ring systems, the other ring is the substituent
    sub_idx = 1 - parent_idx if len(ring_systems_list) == 2 else None
    if sub_idx is None:
        # For 3+ non-identical rings, not yet supported
        return None

    parent_atoms = ring_systems_list[parent_idx]
    sub_atoms = ring_systems_list[sub_idx]

    # Get the parent ring's system name (for stem)
    parent_ring_tuple = tuple(sorted(parent_atoms))
    parent_ring_name = identify_ring_system(mol, parent_ring_tuple)
    if not parent_ring_name:
        return None

    # Get the substituent ring's -yl name
    sub_ring_tuple = tuple(sorted(sub_atoms))

    # For the substituent ring, get its prefix name with position if applicable
    # Find where the inter-ring bond attaches to the sub ring
    sub_attach_atom = None
    for a1, a2, s1, s2 in inter_system_bonds:
        if s1 == sub_idx:
            sub_attach_atom = a1
            break
        elif s2 == sub_idx:
            sub_attach_atom = a2
            break

    sub_yl_name = get_ring_substituent_name(
        mol, sub_ring_tuple, attachment_point=sub_attach_atom
    )
    if not sub_yl_name:
        return None

    # Check if sub ring name needs parenthesization (if it contains locants/hyphens)
    # e.g., "pyridin-2-yl" needs parentheses: "(pyridin-2-yl)"
    # but "phenyl" does not
    if '-' in sub_yl_name and not sub_yl_name.startswith('('):
        sub_display = f"({sub_yl_name})"
    else:
        sub_display = sub_yl_name

    # Parent ring stem: drop terminal 'e' before -yl (vowel elision per IUPAC)
    parent_stem = parent_ring_name
    if parent_stem.endswith('e'):
        parent_stem = parent_stem[:-1]

    # Determine locants on the parent ring
    # Find the inter-ring bond atom on the parent ring
    parent_conn_atom = None
    for a1, a2, s1, s2 in inter_system_bonds:
        if s1 == parent_idx:
            parent_conn_atom = a1
            break
        elif s2 == parent_idx:
            parent_conn_atom = a2
            break
    if parent_conn_atom is None:
        return None

    # For locant calculation, use _get_substituent_locant which numbers
    # carbocyclic rings from the chain attachment point (locant 1) and
    # heterocyclic rings from standard IUPAC numbering
    connection_locant = _get_substituent_locant(
        mol, parent_conn_atom, parent_conn_atom,
        parent_atoms, attachment_atom_idx
    )
    attach_locant = _get_substituent_locant(
        mol, attachment_atom_idx, attachment_atom_idx,
        parent_atoms, attachment_atom_idx
    )

    # For heterocyclic parent rings, use the standard IUPAC numbering
    # (not relative to attachment point). Check if parent is heterocyclic.
    parent_has_het = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in parent_atoms
    )
    if parent_has_het:
        # Standard IUPAC numbering from _get_connection_locant
        connection_locant = _get_connection_locant(mol, parent_conn_atom, parent_atoms)
        attach_locant = _get_connection_locant(mol, attachment_atom_idx, parent_atoms)

    # Build compound prefix
    # Format: "(connection_locant-sub_displaystem-attach_locant-yl)"
    # Example with phenyl parent: "(4-(pyridin-2-yl)phenyl)" -- attach_locant omitted when 1
    # Example with pyridine parent: "(2-phenylpyridin-3-yl)"

    # For benzene, use "phenyl" directly (retained substituent name)
    # For other rings, use "stem-attach_locant-yl"
    if parent_ring_name == 'benzene':
        # Benzene as parent: "phenyl" is the standard substituent name
        # Locant for connection: the position where sub ring attaches
        # When chain attachment = position 1, connection locant is meaningful
        if attach_locant == 1:
            # Simple case: chain at position 1
            return f"({connection_locant}-{sub_display}phenyl)"
        else:
            return f"({connection_locant}-{sub_display}phenyl)"
    else:
        # General case: stem-attach_locant-yl
        return f"({connection_locant}-{sub_display}{parent_stem}-{attach_locant}-yl)"


# ============================================================================
# a phase-03: cascade-step-6 supplier (get_ring_assembly_iupac_locants)
# ============================================================================

def get_ring_assembly_iupac_locants(mol) -> Optional[Dict[int, _Locant]]:
    """a phase-03 cascade-step-6 supplier for ring assemblies size >= 2.

    Returns the per-system IUPAC numbering of every ring atom merged into a
    single ``Dict[int, int]`` covering ALL ring atoms in the assembly.
    Primes are NAME-format-layer concerns only (in ``_format_prime``), so
    the supplier emits plain integer locants — the comparator in
    ``compare_locant_sets`` and ``_build_ring_pos`` sees the integer base.

    Coverage invariant per Pitfall 7: returns ``None`` when partial
    coverage would otherwise leak into the cascade-step-6 gate. The gate
    in ``candidate_pool.py:634::_has_iupac_locants`` checks dict
    truthiness only; a partial map would silently mis-rank candidates.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping atom_idx -> int locant covering all ring atoms in
        every system of the assembly. Returns ``None`` when:
          * the molecule is not a ring assembly per ``detect_ring_assembly``
            (size < 2, mixed signatures, branched topology, oversize, etc.),
          * any system's per-ring numbering produces partial coverage.

    Source: 151-internal notes,,.
    Source: internal notes Pattern S-3 (cascade-step-6 supplier contract).
    Source: internal notes §"OPSIN Compatibility Evidence" (12 named cases).
    """
    from ..perception.rings import get_ring_systems

    if mol is None:
        return None
    ring_systems = get_ring_systems(mol, include_spiro=False)
    if len(ring_systems) < 2:
        return None

    info = detect_ring_assembly(mol, ring_systems)
    if info is None:
        return None

    per_system = _compute_per_system_ring_locants(mol, info)
    if per_system is None:
        return None

    # Merge per-system maps into a single atom_idx -> locant map.
    merged: Dict[int, int] = {}
    for sys_map in per_system:
        for atom_idx, loc in sys_map.items():
            merged[atom_idx] = loc

    # Coverage invariant (Pitfall 7): every ring atom in the assembly
    # must be covered. Partial coverage returns None so the cascade
    # falls through to the sorted-int proxy (no silent mis-ranking).
    ri = mol.GetRingInfo()
    all_ring_atoms: Set[int] = set()
    for r in ri.AtomRings():
        all_ring_atoms.update(r)
    if not (set(merged.keys()) >= all_ring_atoms):
        return None

    # Filter to ring atoms only (cascade-step-6 contract; no acyclic atoms).
    return {k: v for k, v in merged.items() if k in all_ring_atoms}
