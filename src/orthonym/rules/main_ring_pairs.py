"""Pairs of interior-disjoint bridgehead paths for the von Baeyer main-ring search, without the pair scan.

``polycyclic._find_main_ring_impl`` lists the simple bh1->bh2 paths of a ring system and, for each
bridgehead pair, takes from them the pairs of paths whose interiors (the atoms other than bh1 and bh2)
are disjoint: the two paths and the two bridgeheads are a main ring "Selection of the main
ring", ``the Blue Book Blue Book``: "The main ring of a polycyclic hydrocarbon ring system is
selected so as to include as many skeletal atoms of the structure as possible"), and a third path
disjoint from both is the main bridge "Selection of the main bridge", ``:9603``; the main
ring is divided as symmetrically as possible by it, "Selection of the main bridge and
secondary bridges", rule at ``:9661``). A ring system of 13 or more rings lists thousands
of paths for one bridgehead pair, and the loops that tested every pair of them for disjointness
(``_best_disjoint_pair_loop`` below and the loop of ``polycyclic._consider_multiatom_bridges``, one
perf-budget unit per pair) are quadratic: 13,419 paths are 90 million pairs, more than seven times the
12 million units of the budget, so the molecule was abstained at ``perf_op_budget``.

The answer does not need the pairs. Number the paths by their position in ``order`` (the longest
interior first). With one big integer per ring atom that holds the set of paths through it (bit p set
for the path at position p), the paths disjoint from path p are the complement of the union of those
integers over p's interior atoms, one ``|`` per atom, and the paths of one length are a run of
positions, so the partners of p of a given length are one ``&``.

*:func:`best_disjoint_pair` (Cases 1 and 2): the pair with the largest ring size (the sum of the two
  interior sizes plus two), then the larger balance (the smaller of them). Both are fixed by the two
  lengths, so the class of lengths is found first, longest partner of each path first, and only the pairs
  of that one class are listed and ranked by the canonical-rank tie key, as the loop ranked every pair
  that reached the incumbent's class. The pair returned is the loop's, pair for pair: the largest key
  ``(ring size, balance, canonical-rank tuple of the atoms of the two paths)``, and of two equal keys the
  one with the lower (i, j), i < j, which the loop met first and kept (it replaced the incumbent only on
  a strictly larger key).
*:func:`offered_multiatom_pairs` (Case 3): the pairs the loop offered to ``_MainRingBest.consider``,
  found among the disjoint pairs only, visited in the order of the loop, so the incumbent that bounds
  the ring size evolves as it did and the longest-bridge search (``bridge_len``) is asked for the same
  free atoms in the same order.

The perf budget is charged what the search does: the big-integer operations it makes, each at the
fraction of a unit its size costs (``_UNIT_BASE``, ``_UNIT_BITS``), and never more in all than the loop
it replaces charged for the same list (one unit per pair of the list; in Case 3, per pair of the
ring-size window the loop visits). A list of fewer than ``PAIR_SCAN_MAX_PATHS`` paths is searched by the
loops, as it always was, with the units they charged.
"""
from functools import reduce
from operator import or_
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_SHIFT = (1).__lshift__             # atom index -> its bit


def interior_masks(paths):
    """Per path, its interior atoms (``p[1:-1]``) as one int bitmask (atom index -> bit)
    and the interior size. An ``&`` of two masks is the interior-disjointness test of two
    bh1->bh2 paths, in place of a ``set`` intersection per pair."""
    masks = [reduce(or_, map(_SHIFT, p[1:-1]), 0) for p in paths]
    return masks, [m.bit_count() for m in masks]


#: A list of fewer paths than this is searched by the loops: it is at most
#: ``PAIR_SCAN_MAX_PATHS`` * (``PAIR_SCAN_MAX_PATHS`` - 1) / 2 units, and the set arithmetic below costs
#: more than that to set up.
PAIR_SCAN_MAX_PATHS = 96

#: One perf-budget unit is about a microsecond (a pair of the loops). An ``&`` or ``|`` of two big
#: integers of ``n`` bits takes 0.07 microsecond and 9 nanoseconds per 1,000 bits more (measured: 0.07,
#: 0.21 and 1.0 microsecond at 100, 13,000 and 100,000 bits), so an operation is charged
#: ``(n + _UNIT_BASE) / _UNIT_BITS`` units (rounded up on the total of a run of operations).
_UNIT_BASE = 8192
_UNIT_BITS = 100000

#: Units per pair whose canonical-rank key is built (two sets of ring atoms joined and sorted by rank,
#: about ten microseconds); the loop charged nothing for it.
_RANK_UNITS = 8

#: The sets of paths that share an atom with a path, kept for the Case 3 search of the main bridge, are
#: bounded in memory: at most this many bits in all.
_KEEP_BITS = 1 << 29


class _Meter:
    """The perf-budget charge of a search, in operations on big integers of ``nbits`` bits: ``ops``
    counts them, and:meth:`flush` spends the units of the operations counted so far (and the ``units``
    charged as they are), never more in all than ``cap`` (the units the loop it replaces charged)."""

    __slots__ = ("spend", "nbits", "ops", "units", "spent")

    def __init__(self, spend: Callable[..., None], nbits: int):
        self.spend = spend
        self.nbits = nbits
        self.ops = 0
        self.units = 0          # units charged as they are: work that is not an operation on big integers
        self.spent = 0

    def flush(self, cap: int) -> None:
        due = min(-(-self.ops * (self.nbits + _UNIT_BASE) // _UNIT_BITS) + self.units, cap) - self.spent
        if due > 0:
            self.spent += due
            self.spend(due)


def _positions(bits: int, nbits: int) -> List[int]:
    """The positions of the set bits of ``bits`` (an integer of at most ``nbits`` bits), ascending:
    peeled from the top when few are set, read from the bytes when many are."""
    if bits.bit_count() * 48 <= nbits:
        out = []
        while bits:
            top = bits.bit_length() - 1
            out.append(top)
            bits ^= 1 << top
        out.reverse()
        return out
    out = []
    for at, byte in enumerate(bits.to_bytes((nbits + 7) >> 3, "little")):
        if byte:
            base = at << 3
            out.extend(base + b for b in range(8) if (byte >> b) & 1)
    return out


def best_disjoint_pair(all_paths: Sequence[Sequence[int]], spend: Callable[..., None],
                       rank_key: Callable) -> Optional[Tuple[Sequence[int], Sequence[int]]]:
    """Largest, then most symmetric, pair of interior-disjoint paths of ``all_paths``
    (the bh1->bh2 paths). Deterministic: ties broken by the lowest canon-rank tuple, then by the
    lower position in the list. ``spend`` is charged for the work done (see the module docstring),
    never more than one unit per pair of the list."""
    if len(all_paths) < max(PAIR_SCAN_MAX_PATHS, 2):
        return _best_disjoint_pair_loop(all_paths, spend, rank_key)
    return _best_disjoint_pair_sets(all_paths, [max(len(p) - 2, 0) for p in all_paths], spend, rank_key)


def _best_disjoint_pair_loop(all_paths, spend, rank_key):
    """The pair loop, one perf-budget unit per (i, j) pair (M2.5); the reference of the search below."""
    npaths = len(all_paths)
    masks, plens = interior_masks(all_paths)
    best_local = None
    best_pair = None
    for i in range(npaths - 1):
        mask_i = masks[i]
        len_i = plens[i]
        for j in range(i + 1, npaths):
            spend()
            if mask_i & masks[j]:
                continue
            len_j = plens[j]
            ring_size = len_i + len_j + 2  # shared endpoints
            balance = len_i if len_i < len_j else len_j
            if best_local is not None and (ring_size, balance) < best_local[:2]:
                continue  # the rank key only decides a tie of (ring_size, balance)
            p1 = all_paths[i]
            p2 = all_paths[j]
            key = (ring_size, balance, tuple(-r for r in rank_key(set(p1) | set(p2))))
            if best_local is None or key > best_local:
                best_local = key
                best_pair = (p1, p2)
    return best_pair


#: ``_COLUMN[b]`` maps a byte to the character ``1`` where its bit ``b`` is set and ``0`` where it
#: is not.
_COLUMN = [bytes(0x31 if (value >> bit) & 1 else 0x30 for value in range(256)) for bit in range(8)]


class PathSets:
    """The paths of one bridgehead pair as sets of paths, numbered by position in ``order`` (longest
    interior first, equal lengths in list order). ``through[a]`` has bit p set when the path at position
    p has atom a in its interior; ``runs`` are the (length, first position, last position + 1) of each
    length, longest first; ``atoms[p]`` are the interior atoms of the path at position p.

    ``masks`` are the interior bitmasks of the paths in list order (``interior_masks``), when the
    caller has them. The sets through the atoms are the columns of the table of the interiors, one
    row of bytes a path: the byte of an atom in every row is one slice, and its bit one
    ``translate`` and ``int`` away."""

    __slots__ = ("n", "order", "lens", "atoms", "through", "runs", "by_len", "everything")

    def __init__(self, paths: Sequence[Sequence[int]], plens: Sequence[int],
                 masks: Optional[Sequence[int]] = None):
        n = len(paths)
        self.n = n
        self.order = order = sorted(range(n), key=lambda k: -plens[k])
        self.lens = [plens[k] for k in order]
        self.atoms = [paths[k][1:-1] for k in order]
        if masks is None:
            masks = interior_masks(paths)[0]
        by_position = [masks[k] for k in order]
        used = reduce(or_, by_position, 0)                       # the atoms in any interior
        width = max((used.bit_length() + 7) >> 3, 1)
        table = b"".join([m.to_bytes(width, "little") for m in by_position])
        self.through = through = {}
        while used:
            low = used & -used
            used ^= low
            a = low.bit_length() - 1
            through[a] = int(table[a >> 3::width].translate(_COLUMN[a & 7])[::-1], 2)
        self.runs = runs = []
        for p, length in enumerate(self.lens):
            if runs and runs[-1][0] == length:
                runs[-1][2] = p + 1
            else:
                runs.append([length, p, p + 1])
        self.by_len = {length: ((1 << (stop - start)) - 1) << start for length, start, stop in runs}
        self.everything = (1 << n) - 1

    def blocked(self, p: int) -> int:
        """The paths that share an interior atom with the path at position p (p itself, unless its
        interior is empty)."""
        return reduce(or_, map(self.through.__getitem__, self.atoms[p]), 0)

    def after(self, p: int) -> int:
        """The positions after p."""
        return self.everything ^ ((1 << (p + 1)) - 1)


def _best_disjoint_pair_sets(all_paths, plens, spend, rank_key):
    """:func:`best_disjoint_pair` for a long list: the pair of lengths first, from the sets of paths
    through each atom, then the pairs of that class ranked."""
    npaths = len(all_paths)
    cap = npaths * (npaths - 1) // 2               # what the pair loop charges: the search never charges more
    sets = PathSets(all_paths, plens)
    lens, runs, by_len, order = sets.lens, sets.runs, sets.by_len, sets.order
    meter = _Meter(spend, npaths)
    meter.ops = sum(plens) // 2 + len(sets.through) + npaths // 16      # building the sets
    longest = lens[0]
    best_class = None                              # (ring size, balance) of the best pair so far
    holders = []                                   # (position p, the later positions of the class's other length)
    for p in range(npaths - 1):                    # longest interior first
        len_p = lens[p]
        if best_class is not None and len_p + longest + 2 < best_class[0]:
            break                                  # no partner of p, or of a shorter path, reaches the class
        later = sets.after(p) & ~sets.blocked(p)   # the paths after p that are disjoint from p
        meter.ops += len(sets.atoms[p]) + 3
        for length, start, stop in runs:           # the longest partner of p first
            meter.ops += 2
            if best_class is not None and len_p + length + 2 < best_class[0]:
                break
            partners = later & by_len[length]
            if partners:
                klass = (len_p + length + 2, len_p if len_p < length else length)
                if best_class is None or klass > best_class:
                    best_class = klass
                    holders = [(p, _positions(partners, npaths))]
                elif klass == best_class:
                    holders.append((p, _positions(partners, npaths)))
                meter.ops += 2
                break
        if meter.ops > 4096:
            meter.flush(cap)
    if best_class is None:
        meter.flush(cap)
        return None

    # the pairs of the class, ranked: the largest tie key, then the lowest (i, j)
    best_key = None
    best_at = None
    for p, partners in holders:
        i = order[p]
        set_p = set(all_paths[i])
        for q in partners:
            j = order[q]
            meter.units += _RANK_UNITS
            key = tuple(-r for r in rank_key(set_p | set(all_paths[j])))
            at = (i, j) if i < j else (j, i)
            if best_key is None or key > best_key or (key == best_key and at < best_at):
                best_key = key
                best_at = at
    meter.flush(cap)
    i, j = best_at
    return all_paths[i], all_paths[j]


def offered_multiatom_pairs(all_paths: Sequence[Sequence[int]], masks: Sequence[int], plens: Sequence[int],
                            spend: Callable[..., None], incumbent: int, room: int,
                            bridge_len: Optional[Callable[[int], int]]) -> List[Tuple[int, int, int]]:
    """Case 3 of the main-ring search: the pairs ``(lo, hi, main bridge size)``, lo < hi indexes of the
    list, that ``polycyclic._consider_multiatom_bridges`` offers to ``_MainRingBest.consider``, in the
    order it does (sorted): the pairs of bh1->bh2 paths of the largest ring size at least ``incumbent``
    (and at most ``room``) that are interior-disjoint and have a main bridge of 2 or more atoms. The loop
    also offered the pairs of a smaller ring size that it met before a larger one; ``consider`` drops
    those, and they are not listed.

    The loop visits the pairs of the ring-size window in the order of ``order`` (the longest interior
    first, positions ``a < b``) while the incumbent grows with each valid pair, and so does this search
    on the disjoint pairs, which are the only ones that reach the main bridge: the incumbent and the
    calls of ``bridge_len`` are the loop's. ``bridge_len`` is ``None`` to look for the main bridge among
    the paths of the list (the longest path, of 2 or more atoms, disjoint from both).

    Charged one unit per ``_UNIT_BITS`` bits of operation, and never more in all than the loop charged
    for the visited pairs of the ring-size window (the charge of the longest-bridge search, made by
    ``bridge_len`` itself, is the loop's)."""
    npaths = len(masks)
    if npaths < 2:
        return []
    sets = PathSets(all_paths, plens, masks)
    lens, runs, by_len, order = sets.lens, sets.runs, sets.by_len, sets.order
    through = sets.through
    cand2 = (1 << sum(1 for length in lens if length >= 2)) - 1      # the positions of a path of 2+ atoms
    seen_blocked: Dict[int, int] = {}
    keep = max(1, _KEEP_BITS // max(npaths, 1))
    visits = 0                                     # the pairs of the window the loop visited, so far
    meter = _Meter(spend, npaths)
    meter.ops = sum(plens) // 2 + len(through) + npaths // 16  # building the sets
    offered_at = None                              # the ring size of the pairs in ``valid``
    valid: List[Tuple[int, int, int]] = []
    for p in range(npaths - 1):
        len_p = lens[p]
        if len_p + lens[p + 1] + 2 < incumbent:
            break                                  # the largest ring left is below the incumbent
        # the sets of p are made when a length of the window needs them
        meter.ops += len(sets.atoms[p]) + 3
        block_p = None
        i = order[p]
        for length, start, stop in runs:           # the loop's window: ring size from ``incumbent`` to ``room``
            ring_size = len_p + length + 2
            if ring_size > room or stop <= p + 1:
                continue
            if ring_size < incumbent:
                break
            visits += stop - max(start, p + 1)     # each of these partners was a pair the loop visited
            meter.ops += 2
            if block_p is None:
                block_p = sets.blocked(p)
                later = sets.after(p) & ~block_p    # the paths after p that are disjoint from p
            partners = later & by_len[length]
            for q in (_positions(partners, npaths) if partners else ()):
                j = order[q]
                meter.ops += 4
                if bridge_len is not None:
                    main_bridge = bridge_len(masks[i] | masks[j])
                else:
                    block_q = seen_blocked.get(q)
                    if block_q is None:
                        block_q = sets.blocked(q)
                        meter.ops += len(sets.atoms[q])
                        if len(seen_blocked) < keep:
                            seen_blocked[q] = block_q
                    free = cand2 & ~(block_p | block_q)
                    main_bridge = lens[(free & -free).bit_length() - 1] if free else -1
                if main_bridge < 2:
                    continue
                if offered_at is None or ring_size > offered_at:
                    offered_at = ring_size
                    valid = []
                valid.append((i, j, main_bridge) if i < j else (j, i, main_bridge))
                if ring_size > incumbent:
                    incumbent = ring_size
        if meter.ops > 4096:
            meter.flush(visits)
    meter.flush(visits)
    valid.sort()
    return valid
