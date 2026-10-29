"""The main-ring search counts the perf-budget units it spends and hands them over in blocks.

The searches of ``rules/main_ring_paths.py`` (``enumerate_paths``, ``longest_bridge``) and
``polycyclic._find_all_simple_paths_impl`` used to call ``spend_perf_work`` once per DFS frame, as
a recursion, and a call of ``spend_perf_work`` cost about as much as the frame it charged: on a
ring system that spends the whole 12M budget, a unit cost 1.2 microseconds. They now run on an
explicit stack, count the units themselves and call ``spend_perf_work`` with the total when it
would pass what the budget holds (``fragment_naming.perf_units_headroom``) and at the end. What
must not change is what they compute and what the budget ends as: the same paths in the same
order, the same ``capped``, the same total charged, and ``PerfBudgetExceeded`` raised exactly when
the charges one at a time raise it (a charge that brings the budget to 0 or below), with the
budget left at 0 and one more trip counted.

These tests keep the recursions that charged one unit at a time, verbatim (``_reference_*``), and
run both on random cages and fused polyhexes, with a budget that is about to run out (one unit
less than the search spends, as many, one more, half), and with the real ``spend_perf_work``. The
pair searches' sets (``main_ring_pairs.PathSets``, ``interior_masks``) are compared with the loops
that built them.

No JVM is needed.
"""
import random
from itertools import combinations

import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.rules import main_ring_pairs as mpairs
from orthonym.rules import main_ring_paths as mrp
from orthonym.rules import polycyclic as pc
from tests.unit.rules.test_main_ring_levels import _graph_of, _polyhex, _random_cage

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# the searches as they were: a recursion, one ``spend`` per frame (verbatim)
# ---------------------------------------------------------------------------

def _reference_components_outside(graph, visited, spend):
    """``RingGraph.components_outside`` as it was: kept by ``visited``, charged before the fill."""
    known = graph.outside.get(visited)
    if known is not None:
        return known
    rest = graph.mask & ~visited
    spend(1 + rest.bit_count() // mrp.FLOOD_ATOMS_PER_UNIT)
    nmask = graph.nbr_mask
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
    if len(graph.outside) < mrp.COMPONENT_CACHE_MAX:
        graph.outside[visited] = comps
    return comps


def _reference_enumerate_paths(graph, bh1, bh2, need, spend, cap):
    nbrs = graph.nbrs
    nmask = graph.nbr_mask
    bit2 = 1 << bh2
    n1 = nmask[bh1]
    n2 = nmask[bh2]
    prune = mrp.uses_bound(graph, need)

    def components_outside(visited, spend):
        return _reference_components_outside(graph, visited, spend)

    paths = []
    frames = 0
    checks = cuts = 0
    capped = False

    def reachable(visited, v):
        for comp in components_outside(visited, spend):
            if comp & bit2:
                return comp.bit_count() - 1 if comp & (nmask[v] | n1) & ~bit2 else 0
        return 0

    def partner_room(visited):
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


def _reference_longest_bridge(graph, bh1, bh2, free, spend):
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


def _reference_find_all_simple_paths_impl(mol, start, end, allowed_atoms, adj, spend, cap):
    if start == end:
        return [[start]]

    all_paths = []
    expansions = 0

    def dfs(current, visited, path):
        nonlocal expansions
        expansions += 1
        spend()
        if expansions > cap:
            return
        if current == end:
            all_paths.append(path[:])
            return

        if adj is not None:
            nbrs = adj[current]
        else:
            nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()]
        for nbr_idx in nbrs:
            if nbr_idx in allowed_atoms and nbr_idx not in visited:
                visited.add(nbr_idx)
                path.append(nbr_idx)
                dfs(nbr_idx, visited, path)
                path.pop()
                visited.remove(nbr_idx)

    visited = {start}
    dfs(start, visited, [start])
    return all_paths


def _reference_interior_masks(paths):
    masks = []
    sizes = []
    for p in paths:
        m = 0
        for a in p[1:-1]:
            m |= 1 << a
        masks.append(m)
        sizes.append(m.bit_count())
    return masks, sizes


class _ReferencePathSets:
    """``main_ring_pairs.PathSets`` built one interior atom at a time."""

    def __init__(self, paths, plens):
        n = len(paths)
        self.order = order = sorted(range(n), key=lambda k: -plens[k])
        self.lens = [plens[k] for k in order]
        self.atoms = atoms = []
        rows = {}
        nbytes = (n + 7) >> 3
        for p, k in enumerate(order):
            interior = paths[k][1:-1]
            atoms.append(interior)
            byte, bit = p >> 3, 1 << (p & 7)
            for a in interior:
                row = rows.get(a)
                if row is None:
                    row = rows[a] = bytearray(nbytes)
                row[byte] |= bit
        self.through = {a: int.from_bytes(row, "little") for a, row in rows.items()}

    def blocked(self, p):
        out = 0
        for a in self.atoms[p]:
            out |= self.through[a]
        return out


# ---------------------------------------------------------------------------
# a budget, as ``spend_perf_work`` keeps it
# ---------------------------------------------------------------------------

class _TripError(Exception):
    pass


class _Budget:
    """Raises when a charge brings what is left to 0 or below (``fragment_naming``'s)."""

    def __init__(self, left):
        self.left = left
        self.calls = 0

    def __call__(self, n=1):
        self.calls += 1
        self.left -= n
        if self.left <= 0:
            self.left = 0
            raise _TripError

    def headroom(self):
        return self.left - 1


def _outcome(search, budget):
    """What a search returns or that it tripped, and what the budget has left."""
    try:
        result = search()
    except _TripError:
        result = "trip"
    return result, budget.left


def _budgets(units):
    """The budgets that put the trip at the boundary of a search that spends ``units``."""
    around = (1, 2, 3, units - 1, units, units + 1, units // 2, units * 9 // 10, units * 11 // 10,
              10 ** 9)
    return sorted({b for b in around if b > 0})


@pytest.fixture(autouse=True)
def _restore_perf_budget():
    # Some tests arm the thread's perf budget directly; a failed assertion must not leave it armed
    # for the tests that run after it in the same process.
    saved = getattr(fn._fragment_guard, "perf_budget", None)
    yield
    fn._fragment_guard.perf_budget = saved


@pytest.fixture
def levelled(monkeypatch):
    monkeypatch.setattr(mrp, "LEVEL_MIN_RINGS", 0)
    monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FREE", 100)
    monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FRACTION", 1.0)


def _systems(seeds):
    for seed in seeds:
        rng = random.Random(seed)
        mol = _polyhex(rng, rng.randint(3, 6)) if seed % 2 else _random_cage(rng)
        ring_atoms, adj_list, _ = _graph_of(mol)
        bridgeheads = sorted(
            i for i in ring_atoms if len([n for n in adj_list[i] if n in ring_atoms]) >= 3)
        yield seed, mol, ring_atoms, adj_list, bridgeheads


# ---------------------------------------------------------------------------
# enumerate_paths
# ---------------------------------------------------------------------------

def test_enumerate_paths_equals_the_recursion_charge_for_charge_and_trip_for_trip(levelled):
    """Every budget around the units the search spends: the same outcome and budget left."""
    stats = {"runs": 0, "trips": 0, "capped": 0, "bound": 0, "exact_mode": 0}
    for seed, mol, ring_atoms, adj_list, bridgeheads in _systems(range(14)):
        for bh1, bh2 in list(combinations(bridgeheads, 2))[:5]:
            for need in (0, len(ring_atoms) - 5, len(ring_atoms) - 2):
                for cap in (6, 60, 10 ** 9):
                    def reference(budget):
                        graph = _graph_of(mol)[2]

                        def search():
                            found = _reference_enumerate_paths(graph, bh1, bh2, need, budget, cap)
                            return found + (graph.bound_checks, graph.bound_cuts)
                        return search

                    def new(budget, room):
                        graph = _graph_of(mol)[2]

                        def search():
                            found = mrp.enumerate_paths(graph, bh1, bh2, need, budget, cap,
                                                        room(budget))
                            return found + (graph.bound_checks, graph.bound_cuts)
                        return search

                    probe = _Budget(10 ** 9)
                    expected, _ = _outcome(reference(probe), probe)
                    units = 10 ** 9 - probe.left
                    for budget_size in _budgets(units):
                        ref_budget = _Budget(budget_size)
                        want = _outcome(reference(ref_budget), ref_budget)
                        # the room of a budget: the units it takes before it raises; ``None``
                        # never raises (that budget does not), ``0`` is every unit as it is spent
                        rooms = (("headroom", lambda b: b.headroom()), ("each", lambda b: 0))
                        for name, room in rooms:
                            run_budget = _Budget(budget_size)
                            got = _outcome(new(run_budget, room), run_budget)
                            assert got == want, (seed, bh1, bh2, need, cap, budget_size, name)
                            stats["exact_mode"] += name == "each"
                        stats["runs"] += 1
                        stats["trips"] += want[0] == "trip"
                        stats["capped"] += want[0] != "trip" and want[0][1]
                        stats["bound"] += need > 0 and want[0] != "trip" and want[0][2] > 0
    # not vacuous: trips at the boundary, the cap, and the bound all happened
    assert stats["trips"] > 200 and stats["capped"] > 50 and stats["bound"] > 50, stats


def test_enumerate_paths_charges_the_budget_in_blocks_not_by_the_frame(levelled):
    """The point of the change: far fewer calls of ``spend``, the same total."""
    seed, mol, ring_atoms, adj_list, bridgeheads = next(_systems([2]))
    bh1, bh2 = bridgeheads[0], bridgeheads[-1]
    graph = _graph_of(mol)[2]
    each = _Budget(10 ** 9)
    mrp.enumerate_paths(graph, bh1, bh2, 0, each, 10 ** 9, 0)
    block = _Budget(10 ** 9)
    mrp.enumerate_paths(_graph_of(mol)[2], bh1, bh2, 0, block, 10 ** 9, block.headroom())
    assert 10 ** 9 - block.left == 10 ** 9 - each.left > 50
    assert block.calls == 1 and each.calls > 20


def test_enumerate_paths_with_the_real_spend_perf_work(levelled):
    """``spend_perf_work`` as the namer arms it: the budget left, the trip and the trip count."""
    stats = {"trips": 0, "finished": 0}
    for seed, mol, ring_atoms, adj_list, bridgeheads in _systems(range(10)):
        bh1, bh2 = bridgeheads[0], bridgeheads[1]
        need = len(ring_atoms) - 4
        fn._fragment_guard.perf_budget = 10 ** 9
        _reference_enumerate_paths(_graph_of(mol)[2], bh1, bh2, need, fn.spend_perf_work, 10 ** 9)
        units = 10 ** 9 - fn._fragment_guard.perf_budget
        for size in sorted({units - 1, units, units + 1, max(units // 2, 1)}):
            outcomes = []
            for search in (_reference_enumerate_paths, mrp.enumerate_paths):
                fn._fragment_guard.perf_budget = size
                before = fn.hang_budget_trips()
                graph = _graph_of(mol)[2]
                try:
                    if search is _reference_enumerate_paths:
                        result = search(graph, bh1, bh2, need, fn.spend_perf_work, 10 ** 9)
                    else:
                        result = search(graph, bh1, bh2, need, fn.spend_perf_work, 10 ** 9,
                                        fn.perf_units_headroom())
                except fn.PerfBudgetExceeded:
                    result = "trip"
                outcomes.append((result, fn._fragment_guard.perf_budget,
                                 fn.hang_budget_trips() - before))
            assert outcomes[0] == outcomes[1], (seed, size)
            stats["trips"] += outcomes[0][0] == "trip"
            stats["finished"] += outcomes[0][0] != "trip"
    fn._fragment_guard.perf_budget = None
    assert stats["trips"] > 10 and stats["finished"] > 5, stats


def test_a_budget_that_cannot_raise_is_charged_the_total_at_the_end(levelled, monkeypatch):
    """No budget armed, or measurement mode: the room is ``None``, the total is handed over once."""
    seed, mol, ring_atoms, adj_list, bridgeheads = next(_systems([4]))
    bh1, bh2 = bridgeheads[0], bridgeheads[1]
    each = _Budget(10 ** 9)
    _reference_enumerate_paths(_graph_of(mol)[2], bh1, bh2, 0, each, 10 ** 9)
    block = _Budget(10 ** 9)
    mrp.enumerate_paths(_graph_of(mol)[2], bh1, bh2, 0, block, 10 ** 9, None)
    assert block.calls == 1 and block.left == each.left
    # measurement mode counts every unit and never raises: the total is the recursion's
    monkeypatch.setattr(fn, "_PERF_MEASURE", True)
    fn._PERF_SPENT_TOTAL[0] = 0
    fn._fragment_guard.perf_budget = None
    assert fn.perf_units_headroom() is None
    mrp.enumerate_paths(_graph_of(mol)[2], bh1, bh2, 0, fn.spend_perf_work, 10 ** 9,
                        fn.perf_units_headroom())
    assert fn._PERF_SPENT_TOTAL[0] == 10 ** 9 - each.left


def test_perf_units_headroom():
    """One less than the budget left (a charge that brings it to 0 raises); ``None`` when it
    cannot raise."""
    fn._fragment_guard.perf_budget = None
    assert fn.perf_units_headroom() is None                     # no budget armed
    for left in (12_000_000, 1, 0):
        fn._fragment_guard.perf_budget = left
        assert fn.perf_units_headroom() == left - 1
    # the headroom is what ``spend_perf_work`` takes without raising, and one more raises
    fn._fragment_guard.perf_budget = 5
    fn.spend_perf_work(fn.perf_units_headroom())
    assert fn._fragment_guard.perf_budget == 1
    with pytest.raises(fn.PerfBudgetExceeded):
        fn.spend_perf_work(1)
    assert fn._fragment_guard.perf_budget == 0
    fn._fragment_guard.perf_budget = None


# ---------------------------------------------------------------------------
# the flood fill
# ---------------------------------------------------------------------------

def test_flood_is_the_components_of_a_set_and_its_charge():
    mol = Chem.MolFromSmiles("C1CCC(CC1)C1CCCCC1")                 # two rings joined by a bond
    ring_atoms, adj_list, graph = _graph_of(mol)
    rest = graph.mask & ~((1 << 0) | (1 << 3))
    comps, units = graph.flood(rest)
    assert sorted(c.bit_count() for c in comps) == [2, 2, 6]
    assert units == 1 + rest.bit_count() // mrp.FLOOD_ATOMS_PER_UNIT
    # every component is connected and they partition the rest
    assert sum(c.bit_count() for c in comps) == rest.bit_count()
    assert all(a & b == 0 for a, b in combinations(comps, 2))
    assert graph.outside == {}                       # nothing is kept or charged by the fill
    # the reference kept by ``visited`` makes the same components, in the same order
    visited = (1 << 0) | (1 << 3)
    assert _reference_components_outside(_graph_of(mol)[2], visited, lambda n=1: None) == comps
    assert graph.flood(0) == ((), 1)


@pytest.mark.parametrize("cap", [0, 3, 40])
def test_a_set_the_component_cache_does_not_keep_is_charged_again(cap, levelled, monkeypatch):
    """With ``COMPONENT_CACHE_MAX`` small, a set past the cap is flooded and charged each time it
    is met: the units and the result of the search are the recursion's."""
    monkeypatch.setattr(mrp, "COMPONENT_CACHE_MAX", cap)
    stats = {"runs": 0, "recharged": 0}
    for seed, mol, ring_atoms, adj_list, bridgeheads in _systems(range(10)):
        for bh1, bh2 in list(combinations(bridgeheads, 2))[:4]:
            need = len(ring_atoms) - 4
            each, block = _Budget(10 ** 9), _Budget(10 ** 9)
            graph = _graph_of(mol)[2]
            want = _reference_enumerate_paths(graph, bh1, bh2, need, each, 10 ** 9)
            kept = len(graph.outside)
            graph = _graph_of(mol)[2]
            got = mrp.enumerate_paths(graph, bh1, bh2, need, block, 10 ** 9, block.headroom())
            assert got == want and block.left == each.left, (seed, bh1, bh2)
            assert len(graph.outside) == kept <= cap
            stats["runs"] += 1
            stats["recharged"] += kept == cap
    assert stats["runs"] >= 20 and (cap == 0 or stats["recharged"] > 5), stats


# ---------------------------------------------------------------------------
# longest_bridge
# ---------------------------------------------------------------------------

def test_longest_bridge_equals_the_recursion_unit_for_unit_and_trip_for_trip():
    stats = {"runs": 0, "trips": 0, "bridges": set()}
    for seed, mol, ring_atoms, adj_list, bridgeheads in _systems(range(24)):
        graph = _graph_of(mol)[2]
        for bh1, bh2 in list(combinations(bridgeheads, 2))[:6]:
            ends = (1 << bh1) | (1 << bh2)
            rng = random.Random(seed * 100 + bh1 * 10 + bh2)
            frees = (graph.mask & ~ends, graph.mask & ~ends & rng.getrandbits(graph.n_atoms), 0)
            for free in frees:
                probe = _Budget(10 ** 9)
                _outcome(lambda: _reference_longest_bridge(graph, bh1, bh2, free, probe), probe)
                units = 10 ** 9 - probe.left
                for size in _budgets(units):
                    ref_budget = _Budget(size)
                    want = _outcome(
                        lambda: _reference_longest_bridge(graph, bh1, bh2, free, ref_budget),
                        ref_budget)
                    for room in (lambda b: b.headroom(), lambda b: 0):
                        budget = _Budget(size)
                        got = _outcome(
                            lambda: mrp.longest_bridge(graph, bh1, bh2, free, budget, room(budget)),
                            budget)
                        assert got == want, (seed, bh1, bh2, free, size)
                    stats["runs"] += 1
                    stats["trips"] += want[0] == "trip"
                    stats["bridges"].add(want[0])
    assert stats["trips"] > 100 and len(stats["bridges"]) > 4, stats


# ---------------------------------------------------------------------------
# _find_all_simple_paths_impl
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_adj", [True, False], ids=["adj", "mol"])
def test_find_all_simple_paths_equals_the_recursion_charge_for_charge(with_adj, monkeypatch):
    stats = {"runs": 0, "trips": 0, "capped": 0}
    for seed, mol, ring_atoms, adj_list, bridgeheads in _systems(range(14)):
        adj = adj_list if with_adj else None
        allowed_sets = [ring_atoms, ring_atoms - {sorted(ring_atoms)[len(ring_atoms) // 2]}]
        for bh1, bh2 in list(combinations(bridgeheads, 2))[:5]:
            for allowed in allowed_sets:
                for cap in (5, 50, 200_000):
                    monkeypatch.setattr(pc, "_MAX_DFS_EXPANSIONS", cap)
                    probe = _Budget(10 ** 9)
                    _reference_find_all_simple_paths_impl(mol, bh1, bh2, allowed, adj, probe, cap)
                    units = 10 ** 9 - probe.left
                    for size in _budgets(units):
                        ref_budget = _Budget(size)
                        want = _outcome(lambda: _reference_find_all_simple_paths_impl(
                            mol, bh1, bh2, allowed, adj, ref_budget, cap), ref_budget)
                        budget = _Budget(size)
                        monkeypatch.setattr(pc, "spend_perf_work", budget)
                        monkeypatch.setattr(pc, "perf_units_headroom", budget.headroom)
                        got = _outcome(
                            lambda: pc._find_all_simple_paths_impl(mol, bh1, bh2, allowed, adj),
                            budget)
                        assert got == want, (seed, bh1, bh2, cap, size)
                        stats["runs"] += 1
                        stats["trips"] += want[0] == "trip"
                        stats["capped"] += want[0] != "trip" and units > cap
    assert stats["trips"] > 200 and stats["capped"] > 30, stats


def test_find_all_simple_paths_of_one_atom_and_of_a_start_outside_the_allowed_atoms():
    mol = Chem.MolFromSmiles("C1CC2CCC1CC2")
    ring_atoms, adj_list, _ = _graph_of(mol)
    assert pc._find_all_simple_paths_impl(mol, 2, 2, ring_atoms, adj_list) == [[2]]
    allowed = ring_atoms - {0}
    # the start, 0, is not in the allowed atoms: it is the start all the same
    paths = pc._find_all_simple_paths_impl(mol, 0, 5, allowed, adj_list)
    assert paths == _reference_find_all_simple_paths_impl(
        mol, 0, 5, allowed, adj_list, lambda n=1: None, 10 ** 9)
    assert paths and all(p[0] == 0 and p[-1] == 5 for p in paths)


def test_find_all_simple_paths_fails_on_an_atom_the_adjacency_does_not_hold_when_it_reaches_it():
    mol = Chem.MolFromSmiles("C1CCCCC1")
    adj = {0: [1, 5], 1: [0, 2]}                                   # 2 is missing
    with pytest.raises(KeyError):
        _reference_find_all_simple_paths_impl(
            mol, 0, 3, {0, 1, 2, 3}, adj, lambda n=1: None, 10 ** 9)
    with pytest.raises(KeyError):
        pc._find_all_simple_paths_impl(mol, 0, 3, {0, 1, 2, 3}, adj)


# ---------------------------------------------------------------------------
# the sets of the pair searches
# ---------------------------------------------------------------------------

def _random_paths(rng):
    natoms = rng.randint(4, 150)
    paths = []
    for _ in range(rng.choice([1, 2, 3, 5, 17, 60, 200, 700])):
        interior = rng.sample(range(2, natoms), min(rng.randint(0, 30), natoms - 2))
        paths.append([0] + interior + [1])
    return paths


def test_interior_masks_equal_the_loop():
    for seed in range(120):
        paths = _random_paths(random.Random(seed))
        assert mpairs.interior_masks(paths) == _reference_interior_masks(paths)
    assert mpairs.interior_masks([]) == ([], [])
    assert mpairs.interior_masks([[0, 1], [0, 5, 1]]) == ([0, 1 << 5], [0, 1])


@pytest.mark.parametrize("lengths", ["len", "bits"])
def test_path_sets_equal_the_ones_built_an_atom_at_a_time(lengths):
    for seed in range(150):
        rng = random.Random(seed)
        paths = _random_paths(rng)
        masks, bits = mpairs.interior_masks(paths)
        plens = [max(len(p) - 2, 0) for p in paths] if lengths == "len" else bits
        reference = _ReferencePathSets(paths, plens)
        for made in (mpairs.PathSets(paths, plens), mpairs.PathSets(paths, plens, masks)):
            assert made.order == reference.order and made.lens == reference.lens
            assert made.atoms == reference.atoms
            assert made.through == reference.through
            assert made.everything == (1 << len(paths)) - 1
            for p in rng.sample(range(len(paths)), min(len(paths), 25)):
                assert made.blocked(p) == reference.blocked(p)
                assert made.after(p) == made.everything ^ ((1 << (p + 1)) - 1)


def test_path_sets_of_paths_without_an_interior():
    made = mpairs.PathSets([[0, 1], [0, 1]], [0, 0])
    assert made.through == {} and made.blocked(0) == 0 and made.runs == [[0, 0, 2]]


# ---------------------------------------------------------------------------
# the search as a whole, on the real budget
# ---------------------------------------------------------------------------

def test_the_whole_search_trips_at_exactly_the_units_it_spends(levelled, monkeypatch):
    """``_find_main_ring_impl`` with the real ``spend_perf_work`` armed at the units one search
    spends, one less and one more: a budget of ``units`` or less trips (a charge that brings it to
    0 raises), one more finishes with the same result and exactly ``units`` gone."""
    from orthonym.rules.polycyclic import VonBaeyerAnalyzer
    from tests.unit.breadth.test_unnamed_l7_vonbaeyer import _ring_systems
    analyzer = VonBaeyerAnalyzer()
    stats = {"trips": 0, "finished": 0}
    for seed in range(12):
        rng = random.Random(seed)
        mol = _polyhex(rng, rng.randint(4, 7)) if seed % 2 else _random_cage(rng)
        for ring_atoms, bridgeheads in _ring_systems(analyzer, mol):
            fn._fragment_guard.perf_budget = 10 ** 9
            expected = analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads,
                                                     return_candidates=True)
            units = 10 ** 9 - fn._fragment_guard.perf_budget
            for size in (units - 1, units, units + 1):
                if size <= 0:
                    continue
                fn._fragment_guard.perf_budget = size
                try:
                    got = analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads,
                                                        return_candidates=True)
                except fn.PerfBudgetExceeded:
                    assert size <= units and fn._fragment_guard.perf_budget == 0
                    stats["trips"] += 1
                else:
                    assert size > units and got == expected
                    assert size - fn._fragment_guard.perf_budget == units
                    stats["finished"] += 1
    fn._fragment_guard.perf_budget = None
    assert stats["trips"] > 20 and stats["finished"] > 10, stats
