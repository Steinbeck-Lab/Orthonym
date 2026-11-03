"""The main ring of a ring system of many rings, found as a longest cycle.

The main-ring search of ``polycyclic._find_main_ring_impl`` lists, for every pair of bridgeheads,
the simple paths between them and pairs them (``main_ring_paths``, ``main_ring_pairs``). The number
of paths grows exponentially with the rings: a ring system of 21 rings and 65 atoms lists hundreds
of thousands of paths for one pair, runs the perf budget out and is abstained, and a hundred times
the budget does not name it either. Above ``CYCLE_FIRST_MIN_RINGS`` rings the search here takes its
place.

It starts from the rule rather than from the bridgeheads. "Selection of the main ring"
(``the Blue Book Blue Book``): "The main ring of a polycyclic hydrocarbon ring system is
selected so as to include as many skeletal atoms of the structure as possible", so the main ring is a
longest cycle of the ring system. Every longest cycle is listed (:func:`longest_cycles`). For each,
the main bridge is the longest path between two atoms of the cycle through the atoms off it
 "Selection of the main bridge", ``:9603``: "The main bridge is selected so as to be as
large as possible"), and of two equal main bridges the one that divides the main ring more
symmetrically wins "Selection of the main bridge and secondary bridges", rule
 at ``:9661``: "The main ring must be divided as symmetrically as possible by the main
bridge"). The candidates are scored as ``polycyclic._MainRingBest`` scores those of the path search,
(ring size, main bridge, balance), and every candidate at the top is returned, so the numbering
choice among them (``_choose_lowest_locant_numbering``) sees them all.

A longest cycle always has a main bridge: the ring system is biconnected and has more bonds than
atoms, so some path through the atoms off the cycle joins two of its atoms. The two atoms are not
neighbours on the cycle, as the path and the cycle without their bond would be a longer cycle.

The DFS that lists the cycles is a branch and bound. At an atom ``v`` of a path that started at
``s``, the path can still be closed through the atoms not yet used, and only through the blocks
(biconnected components) on the way from ``v`` to ``s`` in the graph of those atoms: a block off that
way can be entered and never left. The atoms of those blocks bound the length of the cycle, and a
path whose bound is below the longest cycle found is cut. On the 21-ring system above the search
visits about 32,000 paths (1.6 s) where a bound on the atoms reachable from ``v`` visits 5.6 million.
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

#: The search here runs for a ring system of at least this many rings (circuit rank,. The
#: best-effort ceiling of the first attempt is 15 rings (``vonbaeyer_universal.
#: BEST_EFFORT_MAX_CAGE_RINGS``), so every system of today's PIN and first-attempt ceilings is
#: searched by the path search as before, with the same result; only the lifted ceilings of the late
#: best-effort offer (``lifted_ring_ceiling_scope``, 16 to 30 rings, never a PIN) reach this search.
CYCLE_FIRST_MIN_RINGS = 16

#: Free atoms per perf-budget unit charged when the DFS visits a path: a visit computes the blocks of
#: the free atoms (about 50 microseconds on 65 atoms, in proportion to the atoms), and a unit is the
#: work of one to two microseconds.
FREE_ATOMS_PER_UNIT = 3

#: Perf-budget units per frame of the main-bridge DFS through the atoms off a cycle.
UNITS_PER_BRIDGE_FRAME = 1

#: Longest cycles kept. Past it the list stops growing: the candidates are a part of the tie, the
#: name is still verified by its read-back, and it is not a PIN at these ring counts.
MAX_CYCLES = 2_000


def _blocks_bound(nbrs: Dict[int, Sequence[int]], free: Set[int], v: int, s: int) -> Optional[int]:
    """The atoms a path from ``v`` to ``s`` through ``free`` (the atoms not yet used, without ``v``
    and ``s``) can visit at most, ``v`` and ``s`` included: the atoms of the blocks on the way from
    ``v`` to ``s`` in the graph of ``free | {v, s}``. None when ``s`` cannot be reached."""
    members = free | {v, s}
    disc = {v: 0}
    low = {v: 0}
    parent = {v: None}
    edges: List[Tuple[int, int]] = []
    block_of: Dict[int, int] = {}       # an atom -> the block of the tree edge into it
    blocks: List[Set[int]] = []
    clock = 1
    stack = [(v, iter(nbrs[v]))]
    while stack:
        x, it = stack[-1]
        for w in it:
            if w not in members:
                continue
            if w not in disc:
                parent[w] = x
                disc[w] = low[w] = clock
                clock += 1
                edges.append((x, w))
                stack.append((w, iter(nbrs[w])))
                break
            if w != parent[x] and disc[w] < disc[x]:
                if disc[w] < low[x]:
                    low[x] = disc[w]
                edges.append((x, w))
        else:
            stack.pop()
            if stack:
                p = stack[-1][0]
                if low[x] < low[p]:
                    low[p] = low[x]
                if low[x] >= disc[p]:
                    block: Set[int] = set()
                    while True:
                        e = edges.pop()
                        block.update(e)
                        if e == (p, x):
                            break
                    index = len(blocks)
                    blocks.append(block)
                    for a in block:
                        if a != p and parent.get(a) is not None and parent[a] in block:
                            block_of.setdefault(a, index)
    if s not in disc:
        return None
    on_way: Set[int] = set()
    y = s
    while y != v:
        on_way.add(block_of[y])
        y = parent[y]
    atoms: Set[int] = set()
    for index in on_way:
        atoms |= blocks[index]
    return len(atoms)


def longest_cycles(nbrs: Dict[int, Sequence[int]], order: Sequence[int],
                   spend: Callable[[int], None]) -> Tuple[int, List[Tuple[int, ...]], bool]:
    """The longest simple cycles of the ring system whose atoms are ``order`` (each atom's ring
    neighbours in ``nbrs``): ``(length, cycles, complete)``. Each cycle is listed once, from its
    atom first in ``order``, in the direction whose second atom comes first in ``order``;
    ``complete`` is False when ``MAX_CYCLES`` cut the list (the first cycles of a DFS that takes the
    atoms and their neighbours in the order of ``order``)."""
    position = {a: k for k, a in enumerate(order)}
    # The neighbours in the order of ``order``: the DFS order, and with it the cycles kept when
    # MAX_CYCLES cuts the list, do not depend on the order the input listed its atoms or bonds in
    nbrs = {a: sorted(nbrs[a], key=position.__getitem__) for a in order}
    best = [3]
    found: List[Tuple[int, ...]] = []
    complete = [True]

    def visit(s, v, path, free):
        spend(1 + len(free) // FREE_ATOMS_PER_UNIT)
        if v != s:
            bound = _blocks_bound(nbrs, free, v, s)
            if bound is None or len(path) - 2 + bound < best[0]:
                return
        if len(path) >= 3 and s in nbrs[v]:
            length = len(path)
            if length > best[0]:
                best[0] = length
                found.clear()
                complete[0] = True
            if length == best[0] and position[path[1]] < position[path[-1]]:
                if len(found) < MAX_CYCLES:
                    found.append(tuple(path))
                else:
                    complete[0] = False
        for w in nbrs[v]:
            if w in free:
                free.discard(w)
                path.append(w)
                visit(s, w, path, free)
                path.pop()
                free.add(w)

    for k, s in enumerate(order):
        if len(order) - k < best[0]:
            break
        visit(s, s, [s], set(order[k + 1:]))
    return best[0], found, complete[0]


def _main_bridges(cycle: Tuple[int, ...], nbrs: Dict[int, Sequence[int]], ring_atoms: Set[int],
                  spend: Callable[[int], None]) -> Dict[Tuple[int, int], Set[int]]:
    """For each pair of atoms of ``cycle`` joined by a path through the ring atoms off it, the
    interior sizes of those paths (0 for a bond between two atoms that are not neighbours on the
    cycle)."""
    on_cycle = set(cycle)
    off = ring_atoms - on_cycle
    length = len(cycle)
    position = {a: k for k, a in enumerate(cycle)}
    sizes: Dict[Tuple[int, int], Set[int]] = {}

    def walk(u, x, depth, seen):
        spend(UNITS_PER_BRIDGE_FRAME)
        for w in nbrs[x]:
            if w in on_cycle:
                if w == u:
                    continue
                if x == u and (position[w] - position[u]) % length in (1, length - 1):
                    continue
                sizes.setdefault((u, w) if u < w else (w, u), set()).add(depth)
            elif w in off and w not in seen:
                seen.add(w)
                walk(u, w, depth + 1, seen)
                seen.discard(w)

    for u in cycle:
        walk(u, u, 0, {u})
    return sizes


def main_bicycles(nbrs: Dict[int, Sequence[int]], ring_atoms: Set[int], rank: Callable[[int], int],
                  spend: Callable[[int], None], allow_multiatom_bridge: bool = True
                  ) -> List[Tuple[List[int], int, int, int, int, int]]:
    """The main bicycles at the top of the -> -> cascade:
    ``(ring, bh1, bh2, branch1, branch2, main_bridge)`` for ``_MainRingBest.consider``, ``ring`` the
    main ring from ``bh1``, ``bh1 < bh2``, in a spelling-invariant order (the canonical ranks of
    the ring's atoms). ``allow_multiatom_bridge`` False keeps main bridges of 0 or 1 atoms (the
    second pass of ``analyze``) when a longest cycle has one."""
    order = sorted(ring_atoms, key=rank)
    length, cycles, _complete = longest_cycles(nbrs, order, spend)
    offers, short = [], []
    for cycle in cycles:
        position = {a: k for k, a in enumerate(cycle)}
        for (u, w), sizes in _main_bridges(cycle, nbrs, ring_atoms, spend).items():
            d = (position[w] - position[u]) % length
            branch1, branch2 = d - 1, length - d - 1
            start = position[u]
            ring = list(cycle[start:] + cycle[:start])
            bridge = max(sizes)
            offers.append(((bridge, min(branch1, branch2)), ring, u, w, branch1, branch2, bridge))
            if sizes & {0, 1}:
                bridge = max(sizes & {0, 1})
                short.append(((bridge, min(branch1, branch2)), ring, u, w, branch1, branch2, bridge))
    if not allow_multiatom_bridge and short:
        offers = short
    if not offers:
        return []
    top = max(o[0] for o in offers)
    tied = [o for o in offers if o[0] == top]
    tied.sort(key=lambda o: (rank(o[2]), rank(o[3]), tuple(rank(a) for a in o[1])))
    return [(ring, u, w, b1, b2, bridge) for _score, ring, u, w, b1, b2, bridge in tied]
