"""Fullerene cage detection (structural, deterministic, no naming).

A fullerene is a closed cage of 12 pentagons and (n/2 - 10) hexagons in which every cage atom
has three cage neighbours (Blue Book, fullerenes). The engine declines them (no
fullerene naming), and each used to cost 11-18 s before the decline. This detector reads the
cage from the ring graph so the decline is made at once.

Rule (per input molecule; nothing is cached):

1. ``G`` is the graph of ring bonds.
2. ``K`` is the 3-core of ``G`` (atoms with fewer than three ring-bond neighbours are dropped
   repeatedly).
3. Every simple 5- and 6-cycle of ``K`` is listed; an edge in >= 2 of them borders two faces.
4. A connected component of such edges is a cage when it has at least ``MIN_CAGE_ATOMS`` atoms,
   every atom has exactly three neighbours in it, it has 12 pentagons and ``V/2 - 10``
   hexagons, and every edge borders exactly two of those faces.

``MIN_CAGE_ATOMS`` is 60. The C20 dodecahedrane cage has the same face count but its carbons
are sp3 and the Blue Book does not treat it as a fullerene; it has a von Baeyer name that
the engine builds, so it must not be matched. Cages of 20 to 58 atoms are not detected.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Set, Tuple

from rdkit import Chem

MIN_CAGE_ATOMS = 60
MAX_CORE_DEGREE = 6


def _small_cycles(adj: Dict[int, Set[int]], maxlen: int = 6) -> List[tuple]:
    """Every simple cycle of length 5..maxlen of ``adj``, once each."""
    order = {a: i for i, a in enumerate(sorted(adj))}
    seen = {}
    for s in adj:
        stack = [(s, (s,))]
        while stack:
            v, path = stack.pop()
            for w in adj[v]:
                if w == s and len(path) >= 5:
                    key = frozenset(frozenset((path[i], path[(i + 1) % len(path)]))
                                    for i in range(len(path)))
                    seen[key] = path
                elif w not in path and len(path) < maxlen and order[w] > order[s]:
                    stack.append((w, path + (w,)))
    return list(seen.values())


def fullerene_cages(mol: Chem.Mol) -> List[frozenset]:
    """The atom-index sets of the fullerene cages of ``mol`` (empty list when none)."""
    adj: Dict[int, Set[int]] = defaultdict(set)
    for b in mol.GetBonds():
        if b.IsInRing():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            adj[i].add(j)
            adj[j].add(i)
    if len(adj) < MIN_CAGE_ATOMS:
        return []
    core = set(adj)
    changed = True
    while changed:
        changed = False
        for a in list(core):
            if len(adj[a] & core) < 3:
                core.discard(a)
                changed = True
    if len(core) < MIN_CAGE_ATOMS:
        return []
    cadj = {a: adj[a] & core for a in core}
    # a cage atom has three ring neighbours (an addend can add one); a denser core is no fullerene, and listing its
    # cycles grows with the fifth power of the degree (B12 icosahedra, complete graphs)
    if max(len(n) for n in cadj.values()) > MAX_CORE_DEGREE:
        return []
    cycles = _small_cycles(cadj)
    ecount: Dict[frozenset, int] = defaultdict(int)
    for p in cycles:
        for i in range(len(p)):
            ecount[frozenset((p[i], p[(i + 1) % len(p)]))] += 1
    hadj: Dict[int, Set[int]] = defaultdict(set)
    for e, n in ecount.items():
        if n >= 2:
            i, j = tuple(e)
            hadj[i].add(j)
            hadj[j].add(i)
    cages, seen = [], set()
    for a in list(hadj):
        if a in seen:
            continue
        comp, st = set(), [a]
        while st:
            v = st.pop()
            if v in comp:
                continue
            comp.add(v)
            st.extend(hadj[v] - comp)
        seen |= comp
        if len(comp) < MIN_CAGE_ATOMS or len(comp) % 2:
            continue
        if any(len(hadj[v] & comp) != 3 for v in comp):
            continue
        faces = [p for p in cycles
                 if all(x in comp for x in p)
                 and all(ecount[frozenset((p[i], p[(i + 1) % len(p)]))] >= 2
                         for i in range(len(p)))]
        n5 = sum(1 for p in faces if len(p) == 5)
        n6 = sum(1 for p in faces if len(p) == 6)
        fe: Dict[frozenset, int] = defaultdict(int)
        for p in faces:
            for i in range(len(p)):
                fe[frozenset((p[i], p[(i + 1) % len(p)]))] += 1
        if (n5 == 12 and n6 == len(comp) // 2 - 10
                and all(v == 2 for v in fe.values()) and len(fe) == 3 * len(comp) // 2):
            cages.append(frozenset(comp))
    return cages


def cage_atoms(mol: Chem.Mol) -> FrozenSet[int]:
    """The atoms of the fullerene cages of ``mol`` (empty for every molecule without one). A chain search that
    enumerates simple paths leaves them out: a cage is a ring system, never part of a chain, and the paths of its 31
    rings are too many to list."""
    if mol is None or mol.GetNumAtoms() < MIN_CAGE_ATOMS:
        return frozenset()
    try:
        return frozenset().union(*fullerene_cages(mol))
    except RuntimeError:            # a molecule whose rings were never perceived has none to list
        return frozenset()


def has_fullerene_cage(mol: Chem.Mol) -> bool:
    """True when ``mol`` contains a fullerene cage (see the module docstring)."""
    return bool(mol is not None and fullerene_cages(mol))


# --------------------------------------------------------------------------------------
# the one C60 skeleton with at most one addend on a [6,6] bond (lane J1 scope)
# --------------------------------------------------------------------------------------

#: the attached ring of a fused addend has 3 to 7 atoms, two of them the cage atoms of the fusion bond
MAX_ATTACHED_RING_ATOMS = 5


@dataclass(frozen=True)
class C60Adduct:
    """A (C60-Ih)[5,6]fullerene with at most one addend on a [6,6] bond.

    ``cage`` holds the 60 atoms of the cage; ``sp3`` the two cage atoms of the addend's bond (empty for
    the bare cage); ``ring`` the atoms of the attached monocycle in order from ``sp3[0]`` to ``sp3[1]``
    (empty when the addend is two exocyclic groups, hydrogen atoms included); ``locants`` every numbering
    of the cage that puts the addend's bond on 1,9 (all 120 for the bare cage), as {atom index: locant}."""
    cage: FrozenSet[int]
    sp3: Tuple[int, ...]
    ring: Tuple[int, ...]
    locants: Tuple[Dict[int, int], ...] = ()


@lru_cache(maxsize=1)
def _template_skeleton() -> Chem.Mol:
    """The C60 template as carbon atoms joined by single bonds, atom index ``i`` = locant ``i + 1``."""
    from ..data.fullerene_c60 import C60_ATOMS, neighbour_pairs
    rw = Chem.RWMol()
    for _ in range(C60_ATOMS):
        rw.AddAtom(Chem.Atom(6))
    for i, j in neighbour_pairs():
        rw.AddBond(i - 1, j - 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def _cage_to_template(mol: Chem.Mol, cage: FrozenSet[int]) -> List[Dict[int, int]]:
    """Every isomorphism of the cage skeleton onto the C60 template, as dicts {atom index: locant}
    (120 for a C60 skeleton, none for any other 60-atom fullerene isomer)."""
    order = sorted(cage)
    pos = {a: i for i, a in enumerate(order)}
    rw = Chem.RWMol()
    for _ in order:
        rw.AddAtom(Chem.Atom(6))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            rw.AddBond(pos[i], pos[j], Chem.BondType.SINGLE)
    skeleton = rw.GetMol()
    skeleton.UpdatePropertyCache(strict=False)
    params = Chem.SubstructMatchParameters()
    params.uniquify = False
    params.maxMatches = 121
    # query = the template: match[q] is the cage position of the atom with locant q + 1
    matches = skeleton.GetSubstructMatches(_template_skeleton(), params)
    return [{order[m[q]]: q + 1 for q in range(len(m))} for m in matches]


def c60_adduct(mol: Chem.Mol):
    """The ``C60Adduct`` reading of ``mol``, or None when ``mol`` is anything else: one fragment, one
    fullerene cage that is the C60 skeleton of ``data.fullerene_c60`` (no heteroatom in it), all atoms neutral
    without isotope or radical, no stereo layer (this lane reads back no descriptor), no atom outside the cage and the attached ring in the ring system of the
    cage, and at most one addend: none, or two adjacent cage atoms on a [6,6] bond that carry the two ends
    of one attached monocycle or two exocyclic groups, every other cage atom carrying no group.

     "Fullerenes and modified fullerenes ortho fused to organic rings or ring systems"
    (the Blue Book): "the fusion bond is always a single bond and the fusion atoms cannot accept an
    'exo' double bond"; an addend that is not a pair of groups on one [6,6] bond, or a ring system that holds
    more than the cage and one monocycle, is not read here."""
    if mol is None or mol.GetNumAtoms() < MIN_CAGE_ATOMS or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return None
    # no stereo layer: a drawn centre or double-bond geometry needs descriptors this lane does not read back
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()) or any(
            b.GetStereo() != Chem.BondStereo.STEREONONE or b.GetBondDir() != Chem.BondDir.NONE for b in mol.GetBonds()):
        return None
    cages = fullerene_cages(mol)
    if len(cages) != 1 or len(cages[0]) != 60:
        return None
    cage = frozenset(cages[0])
    if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in cage):
        return None
    maps = _cage_to_template(mol, cage)
    if not maps:
        return None
    # the cage atoms that carry anything but their three cage neighbours: a group or a hydrogen atom
    sp3 = sorted(a for a in cage if mol.GetAtomWithIdx(a).GetTotalNumHs()
                 or any(n.GetIdx() not in cage for n in mol.GetAtomWithIdx(a).GetNeighbors()))
    # the ring system holding the cage
    system, stack = set(cage), list(cage)
    while stack:
        cur = stack.pop()
        for b in mol.GetAtomWithIdx(cur).GetBonds():
            j = b.GetOtherAtomIdx(cur)
            if b.IsInRing() and j not in system:
                system.add(j)
                stack.append(j)
    attached = system - cage
    if not sp3:
        return C60Adduct(cage, (), (), tuple(maps)) if not attached and mol.GetNumAtoms() == 60 else None
    if len(sp3) != 2 or mol.GetBondBetweenAtoms(*sp3) is None:
        return None
    first = maps[0]
    from ..data.fullerene_c60 import C60_DOUBLE_BONDS
    if frozenset((first[sp3[0]], first[sp3[1]])) not in {frozenset(p) for p in C60_DOUBLE_BONDS}:
        return None                                  # a [5,6] bond
    onto = tuple(m for m in maps if {m[sp3[0]], m[sp3[1]]} == {1, 9})
    if not attached:
        return C60Adduct(cage, tuple(sp3), (), onto)
    # the attached atoms are one path from one sp3 atom to the other, closed by the bond between the two
    if not 1 <= len(attached) <= MAX_ATTACHED_RING_ATOMS:
        return None

    def ring_neighbours(a):
        return [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()
                if n.GetIdx() in system and mol.GetBondBetweenAtoms(a, n.GetIdx()).IsInRing()]

    if any(len([n for n in ring_neighbours(s) if n in attached]) != 1 for s in sp3):
        return None
    if any(len(ring_neighbours(a)) != 2 for a in attached):
        return None
    path, prev = [], sp3[0]
    cur = next(n for n in ring_neighbours(prev) if n in attached)
    while cur in attached and cur not in path:
        path.append(cur)
        step = [n for n in ring_neighbours(cur) if n != prev]
        prev, cur = cur, step[0]
    if cur != sp3[1] or len(path) != len(attached):
        return None
    return C60Adduct(cage, tuple(sp3), tuple(path), onto)
