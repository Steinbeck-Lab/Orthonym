"""Path enumeration of the von Baeyer main-ring search, bounded by the ring size it can still reach.

``VonBaeyerAnalyzer._find_main_ring_impl`` looks at every pair of bridgeheads (bh1, bh2) and, for
each, at every simple bh1->bh2 path of the ring system: a pair of interior-disjoint paths is a
candidate main ring (the two paths and the two bridgeheads), the main bridge is a third disjoint
path. Listing ALL the paths is what makes a ring system of 12 or more rings cost minutes, and
nearly all of them are useless: the incumbent main ring already has most of the skeletal atoms of
the system, and a pair of paths below the incumbent is dropped by ``_MainRingBest.consider``
whatever else it is.

 "Selection of the main ring" (``the Blue Book Blue Book``): "The main ring of a
polycyclic hydrocarbon ring system is selected so as to include as many skeletal atoms of the
structure as possible." The ring size is the first criterion of the score the search keeps
(``(ring size, main bridge, balance)``, and after it), so a candidate below
the best ring size seen cannot be, or tie, the result. This module lists only the paths that can
still take part in a pair of at least a given ring size, in the order the full enumeration would
list them, so the caller offers to ``consider`` exactly the candidates it needs and nothing else.

The bound (exact, a necessary condition). Let ``need`` be the number of interior atoms (atoms other
than bh1 and bh2) the two paths of the pair must cover together, ``ring size - 2``. A partial path
P from bh1 to v, with ``V`` its atoms, can still be extended to a path p of a pair (p, q) of that
size only if

    (atoms of P other than bh1) + (atoms outside V that p's rest or q can still use) >= need.

p's rest runs from v to bh2, and q from bh1 to bh2, both through the atoms outside V; neither
passes through bh2 before its end. So an atom outside V is out of reach of both unless it lies in
the connected component of bh2 in the atoms outside V and that component touches v or bh1 (a
component of the atoms outside V and bh2 that does not touch bh2 can be left only through V; one
that touches neither v nor bh1 can be entered by neither path). The bound counts every atom of that
component but bh2 as usable, and nothing else: an upper bound, so a partial path below ``need`` is
cut with its whole subtree. A complete path p is kept only if ``len(p interior) + (the largest
component of the atoms outside p that touches bh1 and bh2, or 0)`` reaches ``need``, q's interior
being inside such a component. A path that survives is not necessarily in a pair of the size (the
bound is not tight); a path in a pair of the size always survives. The surviving paths are listed
in the DFS order of the full enumeration: the neighbour order is the same and a cut removes whole
subtrees, so no path moves relative to another.

The components of the atoms outside V depend on V alone, not on the pair or on v: the DFS of every
pair that starts at one bridgehead walks the same prefixes, so they are kept per ring system
(``RingGraph``) and a flood fill is paid once per set V, not once per evaluation.
"""
from typing import Callable, List, Tuple

#: The bound is evaluated only for a ring size that leaves out at most this many ring atoms (ring
#: atoms - ring size) and at most ``LEVEL_PRUNE_MAX_FRACTION`` of them. Measured (UN lane B2) on the
#: main-ring search of 14 saved ring systems of 38-73 atoms and 11-17 rings: the bound cuts the paths
#: listed more the fewer atoms the main ring leaves out, and it never cost more than it saved up to
#: 37 left out of 70. A 71-atom system whose main ring leaves 21 atoms out went from 382M units
#: (133 s) with the bound limited to 16 atoms out to 34M (22 s); a 50-atom one leaving 20 out from
#: 12.7M units to 2.0M. Far below that (no incumbent yet: a ring of 3 atoms) the bound cannot cut,
#: and it is not asked for; one that stops cutting is switched off by ``BOUND_USELESS_PER``.
LEVEL_PRUNE_MAX_FREE = 48
LEVEL_PRUNE_MAX_FRACTION = 0.6

#: The bound is switched off for a ring system once it has been evaluated this many times and
#: has cut less than one evaluation in ``BOUND_USELESS_PER``: a system on which it cuts nothing
#: pays an evaluation per decision of the DFS for no listed path less.
BOUND_MIN_CHECKS = 50_000
BOUND_USELESS_PER = 100

#: The levelled search runs only on a ring system of at least this many rings (circuit rank, the
#: ring count of. Below it the full enumeration of the main-ring search is cheap (about
#: 0.05 s for 29 atoms and 12 bridgeheads) and stays as it was, so every system within the 8 rings
#: of the PIN tier (``vonbaeyer_universal.MAX_CAGE_RINGS``) is searched exactly as before, with the
#: same result and the units it charged, less one enumeration where Cases 1 and 3 share a list.
LEVEL_MIN_RINGS = 9

#: Atoms a flood fill covers per perf-budget unit: a flood fill takes about half a microsecond an atom,
#: and the unit of the plain DFS (a frame) or of a pair loop is one to two microseconds.
FLOOD_ATOMS_PER_UNIT = 2

#: The sets of visited atoms whose components are kept per ring system; beyond it a set is
#: computed and not kept, which bounds the memory of a search that visits very many.
COMPONENT_CACHE_MAX = 150_000


class RingGraph:
    """Bitmask view of one ring system: its atoms, each atom's neighbours inside it in the
    ``GetNeighbors`` order of the molecule (the order the DFS visits them in), and the components
    of the ring atoms outside a set of atoms, kept as the DFS asks for them."""

    __slots__ = ("size", "rings", "mask", "nbrs", "nbr_mask", "outside", "bound_checks", "bound_cuts")

    def __init__(self, adj_list, ring_atoms, n_atoms):
        self.size = len(ring_atoms)
        self.mask = 0
        self.nbrs = [()] * n_atoms
        self.nbr_mask = [0] * n_atoms
        self.outside = {}
        self.bound_checks = 0
        self.bound_cuts = 0
        for a in ring_atoms:
            self.mask |= 1 << a
        for a in ring_atoms:
            inside = tuple(n for n in adj_list[a] if n in ring_atoms)
            self.nbrs[a] = inside
            m = 0
            for n in inside:
                m |= 1 << n
            self.nbr_mask[a] = m
        bonds = sum(self.nbr_mask[a].bit_count() for a in ring_atoms) // 2
        self.rings = bonds - self.size + 1

    def bound_is_useless(self) -> bool:
        """True once the bound has been evaluated ``BOUND_MIN_CHECKS`` times on this ring system and
        has cut less than one evaluation in ``BOUND_USELESS_PER``."""
        return (self.bound_checks >= BOUND_MIN_CHECKS
                and self.bound_cuts * BOUND_USELESS_PER < self.bound_checks)

    def components_outside(self, visited: int, spend: Callable[..., None]) -> Tuple[int, ...]:
        """The connected components (atom bitmasks) of the ring atoms outside ``visited``. A set not
        seen before costs a flood fill over its atoms, charged as one perf-budget unit per
        ``FLOOD_ATOMS_PER_UNIT`` atoms (and one more)."""
        known = self.outside.get(visited)
        if known is not None:
            return known
        rest = self.mask & ~visited
        spend(1 + rest.bit_count() // FLOOD_ATOMS_PER_UNIT)
        nmask = self.nbr_mask
        comps = []
        while rest:
            low = rest & -rest
            rest ^= low
            comp = frontier = low
            while frontier:
                lb = frontier & -frontier
                frontier ^= lb
                new = nmask[lb.bit_length() - 1] & rest
                if new:
                    rest ^= new
                    comp |= new
                    frontier |= new
            comps.append(comp)
        comps = tuple(comps)
        if len(self.outside) < COMPONENT_CACHE_MAX:
            self.outside[visited] = comps
        return comps


def uses_bound(graph: RingGraph, need: int) -> bool:
    """True if ``enumerate_paths`` evaluates the bound for ``need`` interior atoms on this ring system:
    a bound is asked for, the main ring leaves out few enough ring atoms (``LEVEL_PRUNE_MAX_FREE``
    and ``LEVEL_PRUNE_MAX_FRACTION``), and the bound has not been found useless here. The list is
    then not the full enumeration's."""
    free = graph.size - 2 - need
    return (need > 0 and free <= LEVEL_PRUNE_MAX_FREE and free <= graph.size * LEVEL_PRUNE_MAX_FRACTION
            and not graph.bound_is_useless())


def enumerate_paths(graph: RingGraph, bh1: int, bh2: int, need: int,
                    spend: Callable[..., None], cap: int) -> Tuple[List[List[int]], bool]:
    """Simple bh1->bh2 paths through the ring system that can take part in a pair of interior-disjoint
    paths covering ``need`` interior atoms, in DFS order; ``(paths, capped)``.

    ``need <= 0`` (or a ring size that ``uses_bound`` does not bound) lists every path:
    the DFS of ``polycyclic._find_all_simple_paths_impl``, frame for frame. One perf-budget unit is
    charged per frame, and ``FLOOD_ATOMS_PER_UNIT`` atoms per unit for each set of visited atoms whose
    components are computed (``RingGraph.components_outside``). ``capped`` is True when ``cap`` frames were exceeded: the
    list is then a prefix of a search no longer comparable with the full enumeration's own prefix,
    and the caller falls back to that enumeration."""
    nbrs = graph.nbrs
    nmask = graph.nbr_mask
    bit2 = 1 << bh2
    n1 = nmask[bh1]
    n2 = nmask[bh2]
    prune = uses_bound(graph, need)
    components_outside = graph.components_outside
    paths: List[List[int]] = []
    frames = 0
    checks = cuts = 0
    capped = False

    def reachable(visited, v):
        """Atoms outside ``visited`` that p's rest or q can still use: the component of bh2 outside
        ``visited`` less bh2, if it touches v or bh1."""
        for comp in components_outside(visited, spend):
            if comp & bit2:
                return comp.bit_count() - 1 if comp & (nmask[v] | n1) & ~bit2 else 0
        return 0

    def partner_room(visited):
        """The most interior atoms a path q disjoint from the complete path ``visited`` can have."""
        room = 0
        for comp in components_outside(visited, spend):
            if comp & n1 and comp & n2:
                size = comp.bit_count()
                if size > room:
                    room = size
        return room

    def dfs(v, visited, path, evaluate):
        nonlocal frames, capped, checks, cuts
        frames += 1
        spend()
        if frames > cap:
            capped = True
            return
        if v == bh2:
            if prune:
                checks += 1
                if len(path) - 2 + partner_room(visited) < need:
                    cuts += 1
                    return
            paths.append(path[:])
            return
        if prune and evaluate:
            checks += 1
            if visited.bit_count() - 1 + reachable(visited, v) < need:
                cuts += 1
                return
        options = [y for y in nbrs[v] if not (visited >> y) & 1]
        evaluate_next = len(options) > 1
        for y in options:
            path.append(y)
            dfs(y, visited | (1 << y), path, evaluate_next)
            path.pop()

    dfs(bh1, 1 << bh1, [bh1], True)
    graph.bound_checks += checks
    graph.bound_cuts += cuts
    return paths, capped


def longest_bridge(graph: RingGraph, bh1: int, bh2: int, free: int, spend: Callable[..., None]) -> int:
    """Interior size of the longest simple bh1->bh2 path whose interior atoms all lie in ``free`` (an
    atom bitmask), or -1 when there is none: the main bridge of a pair whose two paths use the other
    ring atoms, "includes as many of the atoms... not included in the main ring as
    possible"). One perf-budget unit per frame."""
    nbrs = graph.nbrs
    limit = free.bit_count()
    best = -1

    def dfs(v, used, length):
        nonlocal best
        spend()
        for y in nbrs[v]:
            if best == limit:
                return
            if y == bh2:
                if length > best:
                    best = length
            elif (free >> y) & 1 and not (used >> y) & 1:
                dfs(y, used | (1 << y), length + 1)

    dfs(bh1, 0, 0)
    return best
