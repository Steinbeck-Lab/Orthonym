"""Which fused ring system of a bridged fused ring system is the one that is bridged.

 "Selection of bridges" (the Blue Book): "Bridges are selected so that a
recommended fused ring system as described in through is the parent fused
ring system that is bridged." (:14259) ranks the choices: (a) "contain the
maximum number of rings" (:14261), (b) "include the maximum number of skeletal atoms"
(:14271), (c) "have the fewer heteroatoms in the fused ring system before bridging"
(:14279), (d) the most senior ring system (:14289), (e) "have the minimum number of
polyvalent bridges",..., (h) "have the maximum number of divalent bridges". This module
applies (a)-(c) over every fused residual, (d) between different parents (``_senior``),
and (e)-(h) as one count of bridges that are not divalent; ``build`` applies (i) and (j)
(:14395), which need the numbering and the double bonds.

A split whose best-ranked competitor is a residual this module cannot name, or a bridge
that is not divalent (three attachments, or a double bond to the fused ring system, as in
a 'metheno' bridge), is declined: the von Baeyer name stays the fallback.
"""
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from ...perception.mancude import max_matching_size

#: at most this many bridges (one, two or three independent divalent bridges)
MAX_BRIDGES = 3
#: a bridge set that is not one chain (a polyvalent, dependent or ring bridge) is only a
#: competitor in the ranking; one that removes at most MAX_BRIDGES rings holds at most this
#: many branch atoms (``_competitor_sets`` derives the bound), whatever its atom count
MAX_COMPETITOR_BRANCH_ATOMS = 2 * MAX_BRIDGES - 2
#: enumeration bounds: ring-system atoms and rings
MAX_SYSTEM_ATOMS = 40
MAX_SYSTEM_RINGS = 9


@dataclass(frozen=True)
class Split:
    """One way to read a ring system as a fused parent plus divalent bridges.

    ``bridges[i]`` lists the atoms of bridge i in chain order; its first atom is bonded
    to ``bridgeheads[i][0]`` and its last atom to ``bridgeheads[i][1]`` (a one-atom
    bridge is bonded to both). ``n_double`` is the number of noncumulative double bonds
    the input has in the parent ring system (j)). ``unsaturated`` holds the
    ring-system atoms that carry a ring double bond in the input."""
    residual: FrozenSet[int]
    bridges: Tuple[Tuple[int, ...], ...]
    bridgeheads: Tuple[Tuple[int, int], ...]
    parent: Optional[str]
    n_double: int
    unsaturated: FrozenSet[int]


def ring_systems(mol) -> List[Set[int]]:
    """The sets of ring atoms connected through ring bonds, largest first."""
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    out: List[Set[int]] = []
    seen: Set[int] = set()
    for start in sorted(ring_atoms):
        if start in seen:
            continue
        comp, stack = {start}, [start]
        while stack:
            cur = stack.pop()
            for b in mol.GetAtomWithIdx(cur).GetBonds():
                j = b.GetOtherAtomIdx(cur)
                if b.IsInRing() and j in ring_atoms and j not in comp:
                    comp.add(j)
                    stack.append(j)
        seen |= comp
        out.append(comp)
    return sorted(out, key=lambda c: (-len(c), min(c)))


def ring_system(mol) -> Set[int]:
    """The largest ring system."""
    systems = ring_systems(mol)
    return systems[0] if systems else set()


def cycle_rank(mol, atoms: Set[int]) -> int:
    """The number of rings of a connected atom set (bonds - atoms + 1)."""
    return len(_bonds_within(mol, atoms)) - len(atoms) + 1


def _bonds_within(mol, atoms: Set[int]) -> List[Tuple[int, int]]:
    return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]


def _skeleton(mol, atoms: Set[int]):
    """The element-aware, all-single-bond graph induced by ``atoms``."""
    idx = sorted(atoms)
    pos = {a: i for i, a in enumerate(idx)}
    rw = Chem.RWMol()
    for a in idx:
        rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum()))
    for i, j in _bonds_within(mol, atoms):
        rw.AddBond(pos[i], pos[j], Chem.BondType.SINGLE)
    sk = rw.GetMol()
    sk.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(sk)
    return sk


def fused_rings(mol, atoms: Set[int]) -> Optional[List[FrozenSet[int]]]:
    """The rings of ``atoms`` when they form an ortho- or ortho- and peri-fused ring
    system, the Blue Book: "Two rings that have only two atoms and
    one bond in common are said to be ortho-fused"), else None: one connected graph,
    every atom in a ring, as many smallest rings as the cycle rank (a bridged graph has
    more), every pair of rings sharing at most one atom or exactly one bond, the rings
    connected through shared bonds, and no bond in three rings (a ring closed across a
    bond common to two rings is a bridge, (d):14030)."""
    if not atoms:
        return None
    sk = _skeleton(mol, atoms)
    if len(Chem.GetMolFrags(sk)) != 1:
        return None
    rings = [frozenset(r) for r in Chem.GetSymmSSSR(sk)]
    if len(rings) < 2 or len(rings) != sk.GetNumBonds() - sk.GetNumAtoms() + 1:
        return None
    if set().union(*rings) != set(range(sk.GetNumAtoms())):
        return None

    def one_bond(x, y):
        s = x & y
        return len(s) == 2 and sk.GetBondBetweenAtoms(*tuple(s)) is not None

    for x, y in combinations(rings, 2):
        if len(x & y) > 1 and not one_bond(x, y):
            return None
    seen, stack = {0}, [0]
    while stack:
        c = stack.pop()
        for j in range(len(rings)):
            if j not in seen and one_bond(rings[c], rings[j]):
                seen.add(j)
                stack.append(j)
    if len(seen) != len(rings):
        return None
    for b in sk.GetBonds():
        x, y = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if sum(1 for r in rings if x in r and y in r) >= 3:
            return None
    idx = sorted(atoms)
    return [frozenset(idx[k] for k in r) for r in rings]


def ring_unsaturation(mol, system: Set[int]) -> Optional[FrozenSet[int]]:
    """The ring-system atoms that carry a double bond to another ring-system atom on a
    Kekule structure. The set does not depend on which Kekule structure RDKit picks
    (every carbon of an aromatic ring carries one in each); None when the input cannot
    be kekulized or has a ring triple bond."""
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    out = set()
    for b in kek.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in system and j in system:
            if b.GetBondType() == Chem.BondType.TRIPLE:
                return None
            if b.GetBondType() == Chem.BondType.DOUBLE:
                out.update((i, j))
    return frozenset(out)


def _has_perfect_matching(mol, atoms: Set[int]) -> bool:
    adj = {a: set() for a in atoms}
    for i, j in _bonds_within(mol, atoms):
        adj[i].add(j)
        adj[j].add(i)
    return 2 * max_matching_size(atoms, adj) == len(atoms)


def _parent_name(mol, residual: Set[int]) -> Optional[str]:
    """The name of the fused parent on ``residual`` (``parents.fused_parent``), or None
    when the residual is not a parent this package names."""
    from .parents import fused_parent
    got = fused_parent(mol, residual)
    return got.name if got is not None else None


def _five_membered_rule(rings: Sequence[FrozenSet[int]]) -> bool:
    """ (the Blue Book): "Fusion nomenclature gives preferred IUPAC names
    only to compounds having at least two rings of at least five or more members." A
    fused parent with fewer is not a parent of a bridged fused PIN (von Baeyer,:23718,
    :23725)."""
    return sum(1 for r in rings if len(r) >= 5) >= 2


def ears(mol, system: Set[int]) -> List[FrozenSet[int]]:
    """Every maximal chain of ring-system atoms that have exactly two ring-system
    neighbours, between two atoms with three or more,:14043: a divalent
    bridge "is connected by single bonds to two different positions of a fused ring
    system"; its atoms have no other ring-system neighbour, so it is a whole such chain:
    a shorter piece would leave a residual atom outside every ring)."""
    deg = {a: sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in system)
           for a in system}
    out, seen = [], set()
    for a in sorted(system):
        if deg[a] != 2 or a in seen:
            continue
        chain, stack = {a}, [a]
        while stack:
            c = stack.pop()
            for nb in mol.GetAtomWithIdx(c).GetNeighbors():
                j = nb.GetIdx()
                if j in system and deg[j] == 2 and j not in chain:
                    chain.add(j)
                    stack.append(j)
        seen |= chain
        ends = {nb.GetIdx() for c in chain for nb in mol.GetAtomWithIdx(c).GetNeighbors()
                if nb.GetIdx() in system and nb.GetIdx() not in chain}
        if len(ends) == 2:
            out.append(frozenset(chain))
    return out


def _divalent_chain(mol, comp: FrozenSet[int], residual: Set[int]):
    """(chain, (bh0, bh1)) when ``comp`` is an unbranched chain whose first atom is bonded
    to exactly one residual atom bh0 and whose last atom to exactly one other residual
    atom bh1, no inner atom bonded to the residual (a one-atom bridge: bonded to exactly
    two residual atoms). Else None."""
    res_nbrs = {a: [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                    if nb.GetIdx() in residual] for a in comp}
    if len(comp) == 1:
        (a,) = tuple(comp)
        nb = res_nbrs[a]
        return ((a,), (nb[0], nb[1])) if len(nb) == 2 and nb[0] != nb[1] else None
    inner = _bonds_within(mol, set(comp))
    deg = {a: 0 for a in comp}
    for i, j in inner:
        deg[i] += 1
        deg[j] += 1
    if len(inner) != len(comp) - 1 or max(deg.values()) > 2:
        return None
    ends = [a for a in comp if deg[a] == 1]
    if len(ends) != 2 or any(len(res_nbrs[a]) != 1 for a in ends):
        return None
    if any(res_nbrs[a] for a in comp if a not in ends):
        return None
    chain, prev = [min(ends)], None
    while len(chain) < len(comp):
        nxt = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(chain[-1]).GetNeighbors()
                   if nb.GetIdx() in comp and nb.GetIdx() != prev and nb.GetIdx() not in chain)
        prev = chain[-1]
        chain.append(nxt)
    bh0, bh1 = res_nbrs[chain[0]][0], res_nbrs[chain[-1]][0]
    return (tuple(chain), (bh0, bh1)) if bh0 != bh1 else None


def _candidate(mol, system: Set[int], unsat: FrozenSet[int], comps: Sequence[FrozenSet[int]]):
    """(key, Split) for one choice of bridge atom sets, or None when the residual is not a
    fused ring system with two rings of five or more members, or a set is not a bridge.
    ``key`` ranks by (a), (b), (c); ``not_divalent`` counts the bridges that
    are not divalent (three or more attachments, or attached through a double bond: the
    input's ring double bonds must lie inside the residual or inside one bridge,
     for (e)-(h). A Split with no bridges stands for such a competitor;
    ``parent`` is filled in by ``best_splits`` for the best-ranked splits only."""
    residual = set(system) - set().union(*comps)
    for a in residual:
        if sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in residual) < 2:
            return None
    rings = fused_rings(mol, residual)
    if rings is None or not _five_membered_rule(rings):
        return None
    n_hetero = sum(1 for a in residual if mol.GetAtomWithIdx(a).GetAtomicNum() != 6)
    chains, heads, not_divalent = [], [], 0
    for comp in comps:
        n_att = len({nb.GetIdx() for a in comp for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                     if nb.GetIdx() in residual})
        if n_att < 2:
            return None                 # a ring hanging on one atom is not a bridge
        dv = _divalent_chain(mol, comp, residual)
        if dv is None or not _has_perfect_matching(mol, set(comp) & unsat):
            not_divalent += 1
            continue
        chains.append(dv[0])
        heads.append(dv[1])
    res_unsat = set(residual) & unsat
    if not _has_perfect_matching(mol, res_unsat):
        not_divalent += 1
    key = (-len(rings), -len(residual), n_hetero)
    if not_divalent:
        return key, not_divalent, Split(frozenset(residual), (), (), None, 0, unsat)
    return key, 0, Split(frozenset(residual), tuple(chains), tuple(heads),
                         None, len(res_unsat) // 2, unsat)


#: atoms that never carry a ring double bond in a mancude ring: the divalent chalcogens (the O
#: of furan or of 2H-pyran)
_NO_RING_DOUBLE_BOND = frozenset({8, 16, 34, 52})


def _mancude_ring(mol, ring: Sequence[int], unsaturated: FrozenSet[int]) -> bool:
    """True when ``ring`` is a mancude ring of the input: (the Blue Book) "a
    mancude ring or ring system, i.e., one that contains the maximum number of
    noncumulative double bonds", which may have "one or more positions where no multiple
    bond is attached" (the indicated hydrogen). A ring RDKit perceives as aromatic is one.
    Otherwise every atom of the ring that can carry a double bond (all but the divalent
    chalcogens) carries a ring double bond (``unsaturated``), except one when their number
    is odd: the rings of heptalene,:11455) and the eight-membered ring of
    'benzo[8]annulene (PIN)' (:2662), which RDKit does not call aromatic (4n pi electrons),
    and the seven-membered ring of 5H-benzo[7]annulene or the azepine ring of
    1H-1-benzazepine, whose CH2 or NH is the indicated hydrogen. An even number of such
    atoms leaves none over: the six-membered ring of 1,4-dihydronaphthalene is not one."""
    if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
        return True
    able = [a for a in ring if mol.GetAtomWithIdx(a).GetAtomicNum() not in _NO_RING_DOUBLE_BOND]
    return sum(1 for a in able if a not in unsaturated) <= len(able) % 2


def _in_mancude_ring(mol, atom: int, unsaturated: FrozenSet[int]) -> bool:
    """True when ``atom`` is part of the double-bond system of a mancude ring of the input
    (``_mancude_ring``): it is aromatic, or it carries a ring double bond and lies in such a
    ring. An atom without one is at most that ring's indicated hydrogen: the methano carbon
    of '4,7-methanocyclopenta[a]indene (PIN)' (:19829) is the only such atom of the two
    five-membered rings its bridge closes, and it is a bridge, not a cut ring (so are the
    methano carbons of '1,4-ethano-5,8-methanoanthracene (PIN)':14235 and of
    '1,4-methano-10,13-pentanonaphtho[2,3-c][1]benzazocine (PIN)':19904)."""
    if mol.GetAtomWithIdx(atom).GetIsAromatic():
        return True
    return atom in unsaturated and any(atom in ring and _mancude_ring(mol, ring, unsaturated)
                                       for ring in mol.GetRingInfo().AtomRings())


def _sp2_atom(atom) -> bool:
    """An atom of the double-bond system of a mancude parent: aromatic, or carrying a double
    bond in the ring or to an exocyclic =O, =S, =N- or =C< (a ketone or ylidene of a mancude
    parent keeps its ring a ring of that parent:, the Blue Book)."""
    return atom.GetIsAromatic() or any(
        b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC) for b in atom.GetBonds())


def _cuts_a_crossed_mancude_system(mol, split: Split) -> bool:
    """True when a carbon bridge atom of ``split`` lies in a ring system of the input whose
    atoms are all ``_sp2_atom`` (fused rings sharing two or more atoms grouped) and a ring of
    the input outside that system shares three or more of its atoms: the system is linked to
    a chain at nonadjacent positions, (1) (the Blue Book) "at least one ring
    or ring system of which must be a mancude system attached to adjacent atoms or chains at
    nonadjacent ring positions", and the reading cuts it into a bridge (a quinone ring of an
    ansa macrocycle cut into a methano bridge; a para-phenylene linked by -O- and -O-O-S- cut
    into an etheno bridge). (:23843): "cyclic phane systems > fused ring systems >
    bridged fused systems"."""
    rings = [frozenset(r) for r in mol.GetRingInfo().AtomRings()]
    systems: List[Set[int]] = []
    for ring in rings:
        if not all(_sp2_atom(mol.GetAtomWithIdx(a)) for a in ring):
            continue
        joined = [s for s in systems if len(s & ring) >= 2]
        merged = set(ring).union(*joined)
        systems = [s for s in systems if not any(s is j for j in joined)] + [merged]
    carbons = {a for chain in split.bridges for a in chain
               if mol.GetAtomWithIdx(a).GetAtomicNum() == 6}
    return any(system & carbons
               and any(not ring <= system and len(ring & system) >= 3 for ring in rings)
               for system in systems)


def _on_a_produced_parent(mol, split: Split) -> bool:
    """True when the fused parent of ``split`` comes from the two-component producer of slice
    S2c-1 (``hetero_fusion``, source 'hetero_fusion_name+opsin'), a parent no Blue Book row
    reads as a bridged fused PIN. The boundary rows of ``phane_reading`` below are the book's
    and hold for the table parents; on a produced parent the literal reading of (1)
    applies (``_cuts_a_crossed_mancude_system``) until a cyclophane namer decides the class."""
    from .parents import fused_parent
    try:
        got = fused_parent(mol, set(split.residual))
    except Exception:
        return False
    return got is not None and got.source == "hetero_fusion_name+opsin"


def phane_reading(mol, split: Split) -> bool:
    """ (the Blue Book-:23897): True when ``split`` reads a cyclophane as a
    bridged fused system -- a carbon atom of a bridge belongs to a mancude ring of the input
    (``_in_mancude_ring``: the reading cuts a mancude ring into a bridge) while the ring
    system also holds atoms that lie in no mancude ring (the chains that link the mancude
    units; an indicated-hydrogen atom of a mancude ring is not one).

     (1) (:23828): "cyclophanes are cyclic phane structures containing one or
    more rings or ring systems, at least one ring or ring system of which must be a mancude
    system attached to adjacent atoms or chains at nonadjacent ring positions";
    (:23843): "cyclic phane systems > fused ring systems > bridged fused systems > non-fused
    bridged systems". The book's rows fix where the test can apply:
    -:23895-:23897 '(I) 3,7-dithia-1(1,7),5(7,1)-dinaphthalenacyclooctaphane (PIN; a phane
      name) (II) 5,7,14,16-tetrahydro-1,17:8,10-diethenodibenzo[c,j][1,8]
      dithiacyclotetradecine (a bridged fused ring name)' "A phane name is senior to a
      bridged fused ring name." (II) cuts each naphthalene into a benzo ring of the parent
      and an etheno bridge, and its parent ring holds the -CH2-S-CH2- chains.
    - Bridged fused PINs whose saturated chain is the bridge and whose mancude system stays
      whole in the parent: '1,4-ethanonaphthalene (PIN)' (:14407), '9,10-ethanoanthracene
      (PIN)' (:14183), '1,4-methano-10,13-pentanonaphtho[2,3-c][1]benzazocine (PIN)'
      (:19904). A literal reading of (1) would make these cyclophanes; the book does not.
    - Bridged fused PINs whose one-atom heteroatom bridge closes a furan of the formal
      parent: '6,9-epoxy-1,4-methanobenzo[8]annulene (PIN)' (:14189), '1,4-epoxy-4a,8a-
      ethanonaphthalene (PIN)' (:14587). The test looks at carbon bridge atoms only.
    - A bridged fused PIN cut out of one mancude polycycle with no chain: (b)
      (:14271) '6,7-(epiprop[1]en[1]yl[3]ylidene)benzo[a]cyclohepta[e][8]annulene (I)
      (PIN) [not... 4,5-buta[1,3]dienodibenzo[a,d][8]annulene (III)' (:14277): the bridge
      of (I) is three carbon atoms of a benzene ring, and every atom of the ring system
      carries a ring double bond. So does '5,10-ethenobenzo[8]annulene' (the gate's
      W2E-P5BR-2, the same (b) choice); neither is a phane case.
    - The mancude unit need not be aromatic, and it may carry an indicated hydrogen: a
      heptalene joined at its 1,4-positions by a saturated chain, or a 5H-benzo[7]annulene
      joined at its 5,8-positions, is the same shape as the naphthalene of:23895 (II), and
       (1) asks for "a mancude system", not an aromatic one. In the same way a
      polycycle whose only atom without a ring double bond is the indicated hydrogen of a
      mancude ring has no chain: it is the all-mancude case above with an odd atom count.
    - A reading on a parent of the S2c-1 producer (``_on_a_produced_parent``) that cuts a
      carbon bridge out of an sp2 ring system a chain-linked ring crosses
      (``_cuts_a_crossed_mancude_system``) is a cyclophane by (1) read literally."""
    if _on_a_produced_parent(mol, split) and _cuts_a_crossed_mancude_system(mol, split):
        return True
    unsaturated = frozenset(split.unsaturated)
    if not any(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 and _in_mancude_ring(mol, a, unsaturated)
               for chain in split.bridges for a in chain):
        return False
    system = set(split.residual).union(*(set(chain) for chain in split.bridges))
    in_mancude = set()
    for ring in mol.GetRingInfo().AtomRings():
        if system.issuperset(ring) and _mancude_ring(mol, ring, unsaturated):
            in_mancude.update(ring)
    return not system <= in_mancude


#: (c) (the Blue Book) and (g) (:19414): the heteroatom orders
_ORDER_C = ("O", "S", "Se", "Te", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al",
            "Ga", "In", "Tl")
_ORDER_G = ("F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi", "Si",
            "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl")


def p44_2_1_key(mol, atoms) -> Tuple:
    """The key of a ring system (smaller = more senior), the Blue Book-:19418:
    "(a) is a heterocycle; (b) has at least one nitrogen atom; (c) has at least one
    heteroatom (in the absence of nitrogen) that occurs earlier in the following sequence:
    F > Cl > Br > I > O > S > Se > Te > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga
    > In > Tl; (d) has the greater number of rings; (e) has the greater number of skeletal
    atoms; (f) has the greater number of heteroatoms of any kind; (g) has the greater
    number of heteroatoms occurring earlier in the sequence: F > Cl > Br > I > O > S > Se >
    Te > N > P >..."."""
    syms = [mol.GetAtomWithIdx(a).GetSymbol() for a in atoms]
    hetero = [s for s in syms if s != "C"]
    first = min((_ORDER_C.index(s) for s in hetero if s in _ORDER_C), default=len(_ORDER_C))
    counts = tuple(-sum(1 for s in hetero if s == e) for e in _ORDER_G)
    return (not hetero, "N" not in hetero, 0 if "N" in hetero else first,
            -cycle_rank(mol, set(atoms)), -len(syms), -len(hetero), counts)


def _seniority_key(mol, split: Split):
    """A key whose smaller value is the more senior parent ring system by (d)
    (:14289, "see "): (a)-(g) (``p44_2_1_key``; rings, atoms and the
    heteroatom count are equal after (a)-(c)); (a) (:19739) "has
    the larger individual ring component at first point of difference when their ring sizes
    are compared in order of decreasing size"; then the lower heteroatom locant set. That
    last step is not a criterion: it is how the book's own (d) examples decide
    (:14327-:14329 '1,7-ethano[4,1,2]benzoxadiazine (I) (PIN) [not 4,6-ethanopyrido[1,2-d]
    [1,3,4]oxadiazine (II); the locant set '1,2,4' for the heteroatoms in (I) is lower than
    the locant set '1,3,9' in (II) (see ]', and:14253 '1,5-methanoindole (PIN)',
    whose indolizine reading has N4), both between two-ring parents. ``_senior`` uses it only
    there: (b) (:19740) "has the greater number of rings in a horizontal row"
    comes first and is not computed, and it can only separate parents of three or more
    rings (two ortho-fused rings are always one row of two)."""
    from .parents import fused_parent
    rings = fused_rings(mol, set(split.residual)) or []
    sizes = tuple(-len(r) for r in sorted(rings, key=len, reverse=True))
    fp = fused_parent(mol, split.residual)
    if fp is None:
        return None
    het_locs = tuple(sorted(fp.numberings[0][a] for a in split.residual
                            if mol.GetAtomWithIdx(a).GetAtomicNum() != 6))
    if any(not isinstance(x, int) for x in het_locs):
        return None
    return (p44_2_1_key(mol, split.residual), sizes, het_locs)


def _senior(mol, splits: List[Split]) -> Optional[List[Split]]:
    """The splits whose parent is the most senior ring system (d)), or None.

     (:19737-:19743) after (a): "(b) has the greater number of rings in a
    horizontal row; (c) has the lower letter(s) in the fusion descriptor...; (d) has the
    lower number(s) in the fusion descriptor...; (e) has the senior ring system component
    according to ". None of (b)-(e) is applied, so a tie they would break declines:
    two parents left after ``_seniority_key``; two parents of three or more rings that only
    the heteroatom locant step separates ((b) comes first, ``_seniority_key``); two
    fusion-descriptor names that and (a), (b) leave tied ((c), (d))."""
    keyed = []
    for sp in splits:
        k = _seniority_key(mol, sp)
        if k is None:
            return None
        keyed.append((k, sp))
    best = min(k for k, _ in keyed)
    top = [sp for k, sp in keyed if k == best]
    if len({sp.parent for sp in top}) != 1:
        return None
    # the parents that (a)-(g) and (a) leave tied with the best one
    rivals = [sp for k, sp in keyed if k != best and k[:2] == best[:2]]
    if rivals and len(fused_rings(mol, set(top[0].residual)) or ()) != 2:
        # only the heteroatom locants separate them; for three or more rings
        # (b) (:19740, rings in a horizontal row) decides first, which is not computed
        return None
    if "[" in top[0].parent and any("[" in sp.parent for sp in rivals):
        # two fusion-descriptor names still tied after (a) and (b) (two ortho-fused rings
        # are one row): (c), (d) (:19741-:19742, letters, numbers) decide
        # between them (the book's examples:14207,:14211: furo[3,4-b]pyran over
        # furo[2,3-c]pyran), which this package does not apply. A parent that (a)-(g) of
        # or (a) of already ranked below never reaches (c) or (d)
        # (:19737 "applied successively until no alternatives remain").
        return None
    return top


def _competitor_sets(system: Set[int], adj: Dict[int, Set[int]],
                     chains: Sequence[FrozenSet[int]]) -> List[Tuple[FrozenSet[int], int]]:
    """(set, rings it removes) for every bridge set that is not one of ``chains`` and can
    leave a residual ``_candidate`` accepts: polyvalent, dependent and ring bridges
     :14045,:14055,:14075), of any atom count. The
    package spells none of them; they are competitors in the ranking, so a reading that
    loses to one is declined.

    Complete by construction. A residual atom needs two residual neighbours, so a chain of
    two-neighbour atoms is wholly in the set or wholly outside it, and a chain that touches
    a branch atom (three or more ring-system neighbours) of the set is in it; a connected
    set without a branch atom is a single chain. So each competitor is a connected set of
    branch atoms (connected through chains or direct bonds) plus every chain touching them.
    With m branch atoms, c its cycle rank and k its bonds to the residual it removes
    c + k - 1 rings, and three neighbours per branch atom give m <= 2c + k - 2; with
    c + k - 1 <= MAX_BRIDGES and k >= 2 that is m <= 2 * MAX_BRIDGES - 2."""
    branch = sorted(a for a in system if len(adj[a]) >= 3)
    touching: Dict[int, Set[int]] = {a: set() for a in branch}   # chain atoms at a branch atom
    linked: Dict[int, Set[int]] = {a: {j for j in adj[a] if j in touching} for a in branch}
    seen: Set[int] = set()
    for start in sorted(system):
        if len(adj[start]) != 2 or start in seen:
            continue
        chain, stack = {start}, [start]
        while stack:
            cur = stack.pop()
            for j in adj[cur]:
                if len(adj[j]) == 2 and j not in chain:
                    chain.add(j)
                    stack.append(j)
        seen |= chain
        ends = {j for a in chain for j in adj[a] if j not in chain}
        for e in ends:
            touching[e] |= chain
            linked[e] |= ends - {e}
    chain_set = set(chains)
    out = []
    frontier = {frozenset([a]) for a in branch}
    for _size in range(MAX_COMPETITOR_BRANCH_ATOMS):
        for nodes in frontier:
            sub = frozenset(nodes.union(*(touching[a] for a in nodes)))
            if sub in chain_set:
                continue
            att = {j for a in sub for j in adj[a] if j not in sub}
            if len(att) >= 2:
                out.append((sub, _rings_removed(sub, adj)))
        frontier = {s | {j} for s in frontier for a in s for j in linked[a] if j not in s}
    return sorted(out, key=lambda e: (len(e[0]), sorted(e[0])))


def _rings_removed(removed: FrozenSet[int], adj: Dict[int, Set[int]]) -> int:
    """How many rings fewer the residual has than the ring system: removing atoms removes
    their bonds (the bonds inside the set once), and the cycle rank is bonds - atoms + 1."""
    lost = sum(len(adj[a]) for a in removed)
    inner = sum(1 for a in removed for j in adj[a] if j in removed) // 2
    return (lost - inner) - len(removed)


#: ring systems best_splits has declined, keyed by the molecule and the system (the engine
#: asks for the same ring system many times while it ranks parents and substituents)
_DECLINED: Dict[Tuple[str, Tuple[int, ...]], bool] = {}


def best_splits(mol, system: Set[int]) -> Optional[List[Split]]:
    """The splits that tie on (a)-(h), all with one nameable parent and
    divalent bridges only, or None.

    (a) "contain the maximum number of rings": one divalent bridge leaves one ring fewer
    than the ring system, two bridges (or one bridge with three attachments) two fewer,
    three bridges three fewer, so the stages are tried in that order and the first stage
    with a fused residual decides. Within a stage every reading is ranked: (b) atoms, (c)
    heteroatoms, (d) the senior parent, (e)-(h) the bridges that are not divalent. None
    when the system is itself a fused ring system:14241 applies "When a
    polycyclic ring system cannot be named completely as a fused ring system"), when no
    split is found, or when a best-ranked split cannot be named or ranked."""
    system = set(system)
    n_rings = cycle_rank(mol, system)
    if not (3 <= n_rings <= MAX_SYSTEM_RINGS) or len(system) > MAX_SYSTEM_ATOMS:
        return None
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    memo_key = (Chem.MolToSmiles(mol), tuple(sorted(ranks[a] for a in system)))
    if memo_key in _DECLINED:
        return None
    got = _best_splits(mol, system, n_rings)
    if got is None:
        if len(_DECLINED) > 4096:
            _DECLINED.clear()
        _DECLINED[memo_key] = True
    return got


def _best_splits(mol, system: Set[int], n_rings: int) -> Optional[List[Split]]:
    if fused_rings(mol, system) is not None:
        return None
    unsat = ring_unsaturation(mol, system)
    if unsat is None:
        return None
    adj = {a: {nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() in system}
           for a in system}
    chains = ears(mol, system)

    def apart(x, y):
        return not (x & y) and not any(adj[i] & y for i in x)

    def rank(found):
        best = min(k for k, _, _ in found)
        top = [(nd, replace(s, parent=_parent_name(mol, set(s.residual))))
               for k, nd, s in found if k == best]
        if any(s.parent is None for _, s in top):
            return None
        splits = [s for _, s in top]
        if len({s.parent for s in splits}) != 1:
            splits = _senior(mol, splits)                      # (d)
            if splits is None:
                return None
            top = [(nd, s) for nd, s in top if s in splits]
        least = min(nd for nd, _ in top)                       # (e)-(h)
        if least:
            return None
        return [s for nd, s in top if nd == 0]

    others: List[Tuple[FrozenSet[int], int]] = []

    def readings(j):
        """Bridge sets that remove exactly j rings: j divalent chains, or chains with one
        competitor set."""
        if j == 1:
            yield from ((c,) for c in chains)
            return
        if not others:
            others.extend(_competitor_sets(system, adj, chains))
        for combo in combinations(chains, j):
            if all(apart(x, y) for x, y in combinations(combo, 2)):
                yield combo
        for o, removes in others:
            if removes == j:
                yield (o,)
            elif removes == j - 1 and j == 3:
                for c in chains:
                    if apart(c, o):
                        yield (c, o)

    for j in range(1, MAX_BRIDGES + 1):
        found = []
        for comps in readings(j):
            c = _candidate(mol, system, unsat, comps)
            if c is not None and c[0][0] == -(n_rings - j):
                found.append(c)
        if found:
            return rank(found)
    return None
