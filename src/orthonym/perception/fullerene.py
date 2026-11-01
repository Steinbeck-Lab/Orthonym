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
from typing import Dict, List, Set

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


def has_fullerene_cage(mol: Chem.Mol) -> bool:
    """True when ``mol`` contains a fullerene cage (see the module docstring)."""
    return bool(mol is not None and fullerene_cages(mol))
