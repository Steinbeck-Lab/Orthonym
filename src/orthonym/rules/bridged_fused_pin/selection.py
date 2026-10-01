"""Which fused ring system of a bridged fused ring system is the one that is bridged.

 "Selection of bridges" (the Blue Book): "Bridges are selected so that a
recommended fused ring system as described in through is the parent fused
ring system that is bridged." (:14259) ranks the choices: (a) "contain the
maximum number of rings" (:14261), (b) "include the maximum number of skeletal atoms"
(:14271), (c) "have the fewer heteroatoms in the fused ring system before bridging"
(:14279), (d) the most senior ring system (:14289), (e) "have the minimum number of
polyvalent bridges",..., (h) "have the maximum number of divalent bridges". This module
applies (a)-(c) over every fused residual and (d)-(h) as far as slice S1 needs them (one
parent, divalent bridges); ``build`` applies (i) and (j) (:14395), which need the
numbering and the double bonds.

A split whose best-ranked competitor is a residual this module cannot name, or a bridge
that is not divalent (three attachments, or a double bond to the fused ring system, as in
a 'metheno' bridge), is declined: the von Baeyer name stays the fallback.
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from ...perception.mancude import max_matching_size

#: the fused parents slice S1 names retained names), by skeleton SMILES
S1_PARENT_SKELETONS: Dict[str, str] = {
    "C1CCC2CCCCC2C1": "naphthalene",
    "C1CCC2CC3CCCCC3CC2C1": "anthracene",
}
#: at most this many atoms in one bridge, and at most two bridges (slice S1 scope)
MAX_BRIDGE_ATOMS = 4
MAX_BRIDGES = 2
#: the largest ring system that can hold an anthracene and two four-atom bridges
MAX_SYSTEM_ATOMS = 14 + MAX_BRIDGES * MAX_BRIDGE_ATOMS


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
    if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in residual):
        return None
    return S1_PARENT_SKELETONS.get(Chem.MolToSmiles(_skeleton(mol, residual)))


_NAPHTHALENE_QUERY = None


def _holds_s1_parent(mol, system: Set[int]) -> bool:
    """A necessary condition, checked first because it is cheap: the ring system holds a
    naphthalene skeleton (anthracene holds one too) on its own atoms."""
    global _NAPHTHALENE_QUERY
    if _NAPHTHALENE_QUERY is None:
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        params.aromatizeIfPossible = False
        _NAPHTHALENE_QUERY = Chem.AdjustQueryProperties(Chem.MolFromSmiles("C1CCC2CCCCC2C1"), params)
    return any(set(m) <= system for m in mol.GetSubstructMatches(_NAPHTHALENE_QUERY, maxMatches=64))


def _connected_subsets(mol, atoms: Set[int], kmax: int) -> Dict[int, List[FrozenSet[int]]]:
    by_size: Dict[int, List[FrozenSet[int]]] = {}
    frontier = {frozenset([a]) for a in atoms}
    for size in range(1, kmax + 1):
        by_size[size] = sorted(frontier, key=sorted)
        grown = set()
        for s in frontier:
            for a in s:
                for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                    j = nb.GetIdx()
                    if j in atoms and j not in s:
                        grown.add(s | {j})
        frontier = grown
    return by_size


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
    fused ring system or a set is not a bridge. ``key`` ranks by (a), (b),
    (c) and then (e)-(h) as one count: the bridges that are not divalent (three or more
    attachments, or attached through a double bond: the input's ring double bonds must
    lie inside the residual or inside one bridge,. A Split with no bridges
    stands for such a competitor, and ``parent=None`` for a residual slice S1 does not
    name."""
    residual = set(system) - set().union(*comps)
    rings = fused_rings(mol, residual)
    if rings is None:
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
    key = (-len(rings), -len(residual), n_hetero, not_divalent)
    if not_divalent:
        return key, Split(frozenset(residual), (), (), None, 0, unsat)
    return key, Split(frozenset(residual), tuple(chains), tuple(heads),
                      _parent_name(mol, residual), len(res_unsat) // 2, unsat)


def best_splits(mol, system: Set[int]) -> Optional[List[Split]]:
    """The splits that tie on (a)-(h), all nameable by slice S1, or None.

    One divalent bridge leaves one ring more than two bridges or one polyvalent bridge,
    so (a) is settled by the first stage that finds a fused residual: stage 1 tries one
    bridge atom set, stage 2 two sets together with the one-set competitors that have
    three attachments. Within a stage fewer bridge atoms leave more skeletal atoms (b), so
    sizes are tried from the smallest up and the first size with a fused residual
    decides. None when the system is itself a fused ring system:14241 applies
    "When a polycyclic ring system cannot be named completely as a fused ring system"),
    when no split is found, or when a best-ranked split is one slice S1 cannot name."""
    system = set(system)
    n_rings = cycle_rank(mol, system)
    if not (3 <= n_rings <= 3 + MAX_BRIDGES) or len(system) > MAX_SYSTEM_ATOMS:
        return None
    if fused_rings(mol, system) is not None:
        return None
    if not _holds_s1_parent(mol, system):
        return None
    unsat = ring_unsaturation(mol, system)
    if unsat is None:
        return None
    by_size = _connected_subsets(mol, system, MAX_BRIDGE_ATOMS)

    def rank(found):
        best = min(k for k, _ in found)
        top = [s for k, s in found if k == best]
        if any(s.parent is None for s in top) or len({s.parent for s in top}) != 1:
            return None
        return top

    for t in range(1, MAX_BRIDGE_ATOMS + 1):                         # stage 1
        found = [c for s in by_size.get(t, ())
                 for c in [_candidate(mol, system, unsat, (s,))]
                 if c is not None and c[0][0] == -(n_rings - 1)]
        if found:
            return rank(found)
    for t in range(1, 2 * MAX_BRIDGE_ATOMS + 1):                     # stage 2
        found = [c for s in by_size.get(t, ())
                 for c in [_candidate(mol, system, unsat, (s,))] if c is not None]
        for t1 in range(1, t // 2 + 1):
            for x in by_size.get(t1, ()):
                for y in by_size.get(t - t1, ()):
                    if t1 == t - t1 and sorted(x) >= sorted(y):
                        continue
                    if x & y or any(mol.GetBondBetweenAtoms(i, j) is not None
                                    for i in x for j in y):
                        continue
                    c = _candidate(mol, system, unsat, (x, y))
                    if c is not None:
                        found.append(c)
        if found:
            return rank(found)
    return None
