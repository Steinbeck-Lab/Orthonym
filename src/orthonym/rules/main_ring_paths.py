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

The searches run on an explicit stack and count the perf-budget units themselves (one per frame,
and the units of a flood fill): ``spend`` is called with the total when it would pass what the
budget holds and at the end of the search, not once per frame. The budget ends as the charges one
at a time would have left it, and ``PerfBudgetExceeded`` is raised exactly when they would have
raised it (``fragment_naming.perf_units_headroom``, ``enumerate_paths``'s ``room``).
"""
from typing import Callable, List, Optional, Tuple

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

#: A ``room`` that no search reaches: the budget cannot raise (see ``enumerate_paths``).
UNLIMITED_ROOM = 1 << 62


class RingGraph:
    """Bitmask view of one ring system: its atoms, each atom's neighbours inside it in the
    ``GetNeighbors`` order of the molecule (the order the DFS visits them in), and the components
    of the ring atoms outside a set of atoms (``outside``, kept by the set of those atoms as the DFS
    asks for them)."""

    __slots__ = ("size", "rings", "mask", "nbrs", "nbr_mask", "n_atoms", "tables", "outside",
                 "bound_checks", "bound_cuts")

    def __init__(self, adj_list, ring_atoms, n_atoms):
        self.size = len(ring_atoms)
        self.mask = 0
        self.nbrs = [()] * n_atoms
        self.nbr_mask = [0] * n_atoms
        self.n_atoms = n_atoms
        self.tables = None
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

    def search_tables(self):
        """The tables of ``enumerate_paths``, made on its first use: the bit of each atom, the atom
        of each bit of a ring atom, and, per atom, the neighbours that are free as a list in the
        order the DFS takes them, kept by the bitmask of the free neighbours as the DFS meets
        them."""
        if self.tables is None:
            self.tables = ([1 << a for a in range(self.n_atoms)],
                           {1 << a: a for a in range(self.n_atoms) if (self.mask >> a) & 1},
                           [{} for _ in range(self.n_atoms)])
        return self.tables

    def bound_is_useless(self) -> bool:
        """True once the bound has been evaluated ``BOUND_MIN_CHECKS`` times on this ring system and
        has cut less than one evaluation in ``BOUND_USELESS_PER``."""
        return (self.bound_checks >= BOUND_MIN_CHECKS
                and self.bound_cuts * BOUND_USELESS_PER < self.bound_checks)

    def flood(self, rest: int) -> Tuple[Tuple[int, ...], int]:
        """The connected components (atom bitmasks) of ``rest``, a set of ring atoms, and the
        perf-budget units a set not seen before is charged: one per ``FLOOD_ATOMS_PER_UNIT`` atoms
        (and one more). Nothing is kept or charged here: ``enumerate_paths`` charges the units
        when ``outside`` does not hold the set, and keeps it while ``COMPONENT_CACHE_MAX``
        allows."""
        units = 1 + rest.bit_count() // FLOOD_ATOMS_PER_UNIT
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
        return tuple(comps), units


def uses_bound(graph: RingGraph, need: int) -> bool:
    """True if ``enumerate_paths`` evaluates the bound for ``need`` interior atoms on this ring system:
    a bound is asked for, the main ring leaves out few enough ring atoms (``LEVEL_PRUNE_MAX_FREE``
    and ``LEVEL_PRUNE_MAX_FRACTION``), and the bound has not been found useless here. The list is
    then not the full enumeration's."""
    free = graph.size - 2 - need
    return (need > 0 and free <= LEVEL_PRUNE_MAX_FREE and free <= graph.size * LEVEL_PRUNE_MAX_FRACTION
            and not graph.bound_is_useless())


def enumerate_paths(graph: RingGraph, bh1: int, bh2: int, need: int,
                    spend: Callable[..., None], cap: int,
                    room: Optional[int] = 0) -> Tuple[List[List[int]], bool]:
    """Simple bh1->bh2 paths through the ring system that can take part in a pair of
    interior-disjoint paths covering ``need`` interior atoms, in DFS order; ``(paths, capped)``.

    ``need <= 0`` (or a ring size that ``uses_bound`` does not bound) lists every path:
    the DFS of ``polycyclic._find_all_simple_paths_impl``, frame for frame. One perf-budget unit
    is charged per frame, and ``FLOOD_ATOMS_PER_UNIT`` atoms per unit for each set of atoms
    outside the path whose components are computed (``RingGraph.flood``; a set the cache
    ``RingGraph.outside`` holds is not charged again). ``capped`` is True when ``cap`` frames
    were exceeded: the list is then a prefix of a search no longer comparable with the full
    enumeration's own prefix, and the caller falls back to that enumeration.

    ``room`` is how many units ``spend`` takes in all before it raises (``None``: it never
    does). The units are counted here and handed to ``spend`` in one call when they would exceed
    ``room``, and at the end of the search, instead of one call per frame (a call of
    ``spend_perf_work`` cost as much as the frame it charged): ``spend`` has been charged the same
    total at the end of the search, and it raises when that total reaches what the budget holds,
    as it did one unit at a time. With a ``room`` of 0 (the default) every frame is handed to
    ``spend`` as it is searched, and the units of a flood fill with the next frame.

    The DFS is the recursion of ``polycyclic._find_all_simple_paths_impl`` on an explicit stack:
    a frame with one way on continues in place, and a frame with more leaves its other ways on
    the stack."""
    nbr_mask = graph.nbr_mask
    outside = graph.outside
    outside_get = outside.get
    flood = graph.flood
    bits, bit_atom, ways_on = graph.search_tables()
    cache_max = COMPONENT_CACHE_MAX
    bit2 = 1 << bh2
    n1 = nbr_mask[bh1]
    n2 = nbr_mask[bh2]
    prune = uses_bound(graph, need)
    if room is None:
        room = UNLIMITED_ROOM
    # The units charged so far are ``frames`` and the units of the flood fills: ``spend`` raises
    # when they pass ``room``, that is when ``frames`` passes ``left`` = ``room`` less the units
    # of the floods, and the search stops at ``cap`` frames. ``limit`` is the first of the two.
    left = room
    limit = cap if cap < left else left
    paths: List[List[int]] = []
    frames = checks = cuts = 0
    capped = False
    path = [bh1]
    v = bh1
    rest = graph.mask & ~bits[bh1]           # the ring atoms not on the path
    # the bound is evaluated at the frames that follow a choice, and at the first
    evaluate = True
    # the ways on of the frames with a choice: (atom, path length at the frame, rest at the frame)
    stack = []
    push = stack.append
    pop = stack.pop
    while True:
        frames += 1
        if frames > limit:
            if frames > left:                # the units of the search so far pass ``room``
                spend(frames + room - left)
                # ``spend`` did not raise: it is charged each frame from here
                room = 0
                left = frames
                limit = cap if cap < left else left
                frames -= 1
                continue                     # this frame again
            capped = True
        elif v == bh2:
            if prune:
                checks += 1
                comps = outside_get(rest)
                if comps is None:
                    comps, units = flood(rest)
                    left -= units
                    if left < limit:
                        limit = left
                    if len(outside) < cache_max:
                        outside[rest] = comps
                # the most interior atoms a path q disjoint from the complete path can have
                partner_room = 0
                for comp in comps:
                    if comp & n1 and comp & n2:
                        size = comp.bit_count()
                        if size > partner_room:
                            partner_room = size
                if len(path) - 2 + partner_room < need:
                    cuts += 1
                else:
                    paths.append(path[:])
            else:
                paths.append(path[:])
        else:
            if prune and evaluate:
                checks += 1
                comps = outside_get(rest)
                if comps is None:
                    comps, units = flood(rest)
                    left -= units
                    if left < limit:
                        limit = left
                    if len(outside) < cache_max:
                        outside[rest] = comps
                # atoms outside the path that p's rest or q can still use: the component of bh2
                # less bh2, if it touches v or bh1
                reach = 0
                for comp in comps:
                    if comp & bit2:
                        if (comp ^ bit2) & (nbr_mask[v] | n1):
                            reach = comp.bit_count() - 1
                        break
                if len(path) - 1 + reach < need:
                    cuts += 1
                    free = 0
                else:
                    free = nbr_mask[v] & rest
            else:
                free = nbr_mask[v] & rest
            if free:
                if free & (free - 1):        # two ways on or more: keep the others
                    table = ways_on[v]
                    ways = table.get(free)
                    if ways is None:
                        ways = table[free] = tuple(y for y in graph.nbrs[v] if free & bits[y])
                    plen = len(path)
                    if len(ways) == 2:
                        push((ways[1], plen, rest))
                    else:
                        for y in ways[:0:-1]:
                            push((y, plen, rest))
                    v = ways[0]
                    rest ^= bits[v]
                    evaluate = True
                else:                        # one way on: continue in place
                    v = bit_atom[free]
                    rest ^= free
                    evaluate = False
                path.append(v)
                continue
        # this frame is done: the next way on of the nearest frame that has one
        if not stack:
            break
        v, plen, rest = pop()
        path[plen:] = (v,)
        rest ^= bits[v]
        evaluate = True
    pending = frames + room - left
    if pending:
        spend(pending)
    graph.bound_checks += checks
    graph.bound_cuts += cuts
    return paths, capped


def longest_bridge(graph: RingGraph, bh1: int, bh2: int, free: int, spend: Callable[..., None],
                   room: Optional[int] = 0) -> int:
    """Interior size of the longest simple bh1->bh2 path whose interior atoms all lie in ``free``
    (an atom bitmask), or -1 when there is none: the main bridge of a pair whose two paths use
    the other ring atoms, "includes as many of the atoms... not included in the main
    ring as possible"). One perf-budget unit per frame; ``room`` is as for
    :func:`enumerate_paths`.

    The DFS is on an explicit stack: the neighbours of a frame are kept in the order it takes
    them, the bridgehead bh2 among them (reaching it records the length of the frame), and the
    search ends as soon as the bridge cannot be longer than the atoms in ``free``."""
    nbrs = graph.nbrs
    limit = free.bit_count()
    best = -1
    if room is None:
        room = UNLIMITED_ROOM
    left = room
    frames = 0
    # (atom, atoms used, length), or (-1, 0, length) for bh2 reached from a frame of that length
    stack = [(bh1, 0, 0)]
    push = stack.append
    pop = stack.pop
    while stack:
        v, used, length = pop()
        if v < 0:
            if length > best:
                best = length
                if best == limit:
                    break
            continue
        frames += 1
        if frames > left:
            spend(frames + room - left)
            # ``spend`` did not raise: it is charged each frame from here
            room = 0
            left = frames
        usable = free & ~used
        for y in reversed(nbrs[v]):
            if y == bh2:
                push((-1, 0, length))
            elif (usable >> y) & 1:
                push((y, used | (1 << y), length + 1))
    pending = frames + room - left
    if pending:
        spend(pending)
    return best
