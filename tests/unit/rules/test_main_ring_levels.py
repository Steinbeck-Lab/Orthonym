"""UN lane B2: the levelled main-ring search of a ring system of 9 or more rings is exact.

``VonBaeyerAnalyzer._find_main_ring_impl`` used to list every simple path between the two
bridgeheads of every pair and pair them up; the cost of a ring system of 12 or more rings is that
listing (``rules/main_ring_paths.py``). It now lists, per bridgehead pair, only the paths that can
take part in a main ring at least as large as the incumbent's "Selection of the main
ring", ``the Blue Book Blue Book``: the main ring includes as many skeletal atoms as
possible; the ring size is the first criterion of the score the search keeps). What the change must
not do is move any result, so these tests pin:

* the whole result (main ring, bridgehead pair, every oriented tied candidate in order) equals the
  search as it was, ``_reference_find_main_ring_impl`` of the UN L7b tests (verbatim, tests only),
  on random cages, fused polyhexes (the shape of the 12-ring PubChem rows) and the Blue Book cages,
  with the levelled search forced on every system and with the bound's reach set to 0, 2, 8 and
  more atoms, and with the bound switched off by ``BOUND_USELESS_PER`` part of the way through;
* ``enumerate_paths`` lists every path of a pair of the asked ring size, in the order of the full
  enumeration, and every path when it is not asked to prune (and then in the same number of frames);
* a pair whose enumeration exceeds ``_MAX_DFS_EXPANSIONS`` is searched as before, cut off where it
  always was;
* a system of fewer than 9 rings is searched exactly as before, in the same units;
* the structure-keyed memo holds the main ring of an RWMol.

No JVM is needed.
"""
import math
import random
from itertools import combinations

import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from orthonym.rules import main_ring_paths as mrp
from orthonym.rules import polycyclic as pc
from orthonym.rules.polycyclic import VonBaeyerAnalyzer
from tests.unit.breadth.test_unnamed_l7_vonbaeyer import (
    GOLD_VON_BAEYER,
    _reference_find_main_ring_impl,
    _ring_systems,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ring systems
# ---------------------------------------------------------------------------

def _mol_from_graph(rng, n, edges):
    """A molecule of the graph; degree-2 atoms are sometimes N, O or S, so that canonical-rank ties
    and non-ties both occur."""
    degree = [0] * n
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    rw = Chem.RWMol()
    for v in range(n):
        symbol = rng.choice("NOS") if degree[v] == 2 and rng.random() < 0.25 else "C"
        rw.AddAtom(Chem.Atom(symbol))
    for a, b in edges:
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _random_cage(rng):
    """A random connected skeleton of 9-16 atoms (spanning tree plus 3-8 edges, degree <= 4)."""
    n = rng.randint(9, 16)
    edges, degree = set(), [0] * n
    for v in range(1, n):
        u = rng.choice([u for u in range(v) if degree[u] < 4])
        edges.add((u, v))
        degree[u] += 1
        degree[v] += 1
    extra = rng.randint(3, 8)
    for _ in range(500):
        if not extra:
            break
        u, v = sorted(rng.sample(range(n), 2))
        if (u, v) in edges or degree[u] >= 4 or degree[v] >= 4:
            continue
        edges.add((u, v))
        degree[u] += 1
        degree[v] += 1
        extra -= 1
    return _mol_from_graph(rng, n, sorted(edges))


def _polyhex(rng, hexagons):
    """A connected set of fused hexagons (a fused polycyclic aromatic skeleton): the shape of the
    12-ring PubChem rows. Atoms are the hexagon corners, bonds the hexagon sides."""
    cells = {(0, 0)}
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    while len(cells) < hexagons:
        q, r = rng.choice(sorted(cells))
        dq, dr = rng.choice(directions)
        cells.add((q + dq, r + dr))
    corner_ids, edges = {}, set()
    for q, r in sorted(cells):
        cx, cy = math.sqrt(3) * (q + r / 2.0), 1.5 * r
        ids = []
        for k in range(6):
            angle = math.radians(60 * k + 30)
            corner = (round((cx + math.cos(angle)) * 1000), round((cy + math.sin(angle)) * 1000))
            ids.append(corner_ids.setdefault(corner, len(corner_ids)))
        for k in range(6):
            a, b = ids[k], ids[(k + 1) % 6]
            edges.add((min(a, b), max(a, b)))
    return _mol_from_graph(rng, len(corner_ids), sorted(edges))


@pytest.fixture
def levelled(monkeypatch):
    """Force the levelled search on every ring system, with a given reach of the bound."""
    def setting(free):
        monkeypatch.setattr(mrp, "LEVEL_MIN_RINGS", 0)
        monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FREE", free)
        monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FRACTION", 1.0)
    return setting


def _both(mol, allow_multiatom_bridge, units):
    """``(reference result, new result, reference units, new units)`` of the main-ring search of every
    ring system of ``mol`` with two or more bridgeheads."""
    analyzer = VonBaeyerAnalyzer()
    analyzer._allow_multiatom_bridge = allow_multiatom_bridge
    out = []
    for ring_atoms, bridgeheads in _ring_systems(analyzer, mol):
        units[0] = 0
        ref = _reference_find_main_ring_impl(analyzer, mol, ring_atoms, bridgeheads, return_candidates=True)
        ref_units = units[0]
        units[0] = 0
        new = analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
        out.append((ref, new, ref_units, units[0]))
    return out


@pytest.fixture
def units(monkeypatch):
    """Counts the perf-budget units charged inside ``polycyclic`` (its ``spend_perf_work`` is what the
    search is handed). It never touches the real budget."""
    spent = [0]
    monkeypatch.setattr(pc, "spend_perf_work", lambda n=1: spent.__setitem__(0, spent[0] + n))
    return spent


# ---------------------------------------------------------------------------
# (a) the result equals the search as it was
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("free", [0, 2, 8, 100])
def test_levelled_search_equals_the_full_search_on_random_cages(free, levelled, units):
    """Every field of the result, ties and their order included, with the preference for a 2+-atom main
    bridge on and off, for a bound that reaches 0, 2, 8 and all atoms (no bound)."""
    levelled(free)
    stats = {"systems": 0, "tied": 0, "case3": 0}
    for seed in range(70):
        rng = random.Random(seed * 7 + free)
        mol = _random_cage(rng)
        for allow in (True, False):
            for ref, new, _, _ in _both(mol, allow, units):
                assert new == ref, (seed, allow)
                stats["systems"] += 1
                stats["tied"] += len(ref[2]) > 1
                stats["case3"] += allow and ref[0] is not None
    # not vacuous: ties were built, and the main bridge preference ran
    assert stats["systems"] > 100 and stats["tied"] > 30 and stats["case3"] > 30, stats


@pytest.mark.parametrize("free", [0, 4, 8, 100])
def test_levelled_search_equals_the_full_search_on_fused_polyhexes(free, levelled, units):
    levelled(free)
    stats = {"systems": 0, "tied": 0}
    for seed in range(25):
        rng = random.Random(seed * 11 + free)
        mol = _polyhex(rng, rng.randint(3, 7))
        for ref, new, _, _ in _both(mol, True, units):
            assert new == ref, seed
            stats["systems"] += 1
            stats["tied"] += len(ref[2]) > 1
    assert stats["systems"] >= 25 and stats["tied"] > 5, stats


@pytest.mark.parametrize("smiles", GOLD_VON_BAEYER)
def test_levelled_search_equals_the_full_search_on_the_blue_book_cages(smiles, levelled, units):
    levelled(8)
    mol = Chem.MolFromSmiles(smiles)
    for allow in (True, False):
        for ref, new, _, _ in _both(mol, allow, units):
            assert new == ref


def test_levelled_search_charges_far_fewer_units_on_a_large_system(levelled, units):
    """A fused system of 10 hexagons (28 atoms, 10 rings): the reference lists every path of 231 pairs;
    the levelled search charges a tenth of its units or less (measured 1.7M against 0.15M)."""
    levelled(8)
    mol = _polyhex(random.Random(1000 + 1), 10)
    (ref, new, ref_units, new_units), = _both(mol, True, units)
    assert new == ref
    assert ref_units > 1_000_000 and new_units * 10 < ref_units, (ref_units, new_units)


# ---------------------------------------------------------------------------
# (b) the path listing
# ---------------------------------------------------------------------------

def _graph_of(mol):
    ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
    adj_list = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()] for i in ring_atoms}
    return ring_atoms, adj_list, mrp.RingGraph(adj_list, ring_atoms, mol.GetNumAtoms())


def _pairs_ring_size(paths):
    """For each path, the largest ring of a pair it forms with a path interior-disjoint from it."""
    interiors = [set(p[1:-1]) for p in paths]
    best = [0] * len(paths)
    for i, j in combinations(range(len(paths)), 2):
        if not interiors[i] & interiors[j]:
            size = len(paths[i]) + len(paths[j]) - 2
            best[i] = max(best[i], size)
            best[j] = max(best[j], size)
    return best


def _is_subsequence(part, whole):
    it = iter(whole)
    return all(any(item == other for other in it) for item in part)


def test_enumerate_paths_lists_every_path_of_a_pair_of_the_asked_size_in_order(monkeypatch):
    monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FREE", 12)
    monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FRACTION", 1.0)
    stats = {"calls": 0, "cut": 0, "kept_unpaired": 0}
    for seed in range(30):
        rng = random.Random(500 + seed)
        mol = _polyhex(rng, rng.randint(4, 7)) if seed % 2 else _random_cage(rng)
        ring_atoms, adj_list, graph = _graph_of(mol)
        bridgeheads = sorted(i for i in ring_atoms if len([n for n in adj_list[i] if n in ring_atoms]) >= 3)
        for bh1, bh2 in list(combinations(bridgeheads, 2))[:12]:
            full = pc._find_all_simple_paths_impl(mol, bh1, bh2, ring_atoms, adj_list)
            pair_size = _pairs_ring_size(full)
            for threshold in (len(ring_atoms), len(ring_atoms) - 1, len(ring_atoms) - 3, len(ring_atoms) - 7):
                paths, capped = mrp.enumerate_paths(graph, bh1, bh2, threshold - 2, lambda n=1: None, 10 ** 9)
                assert not capped
                assert _is_subsequence(paths, full)
                needed = [p for p, size in zip(full, pair_size) if size >= threshold]
                assert all(p in paths for p in needed), (seed, bh1, bh2, threshold)
                stats["calls"] += 1
                stats["cut"] += len(paths) < len(full)
                stats["kept_unpaired"] += len(paths) > len(needed)
    # not vacuous: paths were cut, and the bound is a necessary condition only (it keeps some paths of no pair)
    assert stats["cut"] > 100 and stats["kept_unpaired"] > 20, stats


def test_enumerate_paths_without_a_bound_is_the_full_enumeration_frame_for_frame(monkeypatch):
    """No bound (``need`` 0, or a reach below the atoms free): the same paths, the same frames."""
    for seed in range(20):
        rng = random.Random(900 + seed)
        mol = _random_cage(rng)
        ring_atoms, adj_list, graph = _graph_of(mol)
        bridgeheads = sorted(i for i in ring_atoms if len([n for n in adj_list[i] if n in ring_atoms]) >= 3)
        for bh1, bh2 in list(combinations(bridgeheads, 2))[:8]:
            charged = [0, 0]
            monkeypatch.setattr(pc, "spend_perf_work", lambda n=1: charged.__setitem__(0, charged[0] + n))
            full = pc._find_all_simple_paths_impl(mol, bh1, bh2, ring_atoms, adj_list)
            for need, reach in ((0, 8), (3, 0)):
                monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FREE", reach)
                charged[1] = 0
                paths, capped = mrp.enumerate_paths(
                    graph, bh1, bh2, need, lambda n=1: charged.__setitem__(1, charged[1] + n), 10 ** 9)
                assert paths == full and not capped
                assert charged[1] == charged[0]
            charged[0] = 0


def test_enumerate_paths_reports_the_cap(monkeypatch):
    mol = Chem.MolFromSmiles("C12C3C4C1C5C2C3C45")           # cubane
    ring_atoms, adj_list, graph = _graph_of(mol)
    paths, capped = mrp.enumerate_paths(graph, 0, 5, 0, lambda n=1: None, 7)
    assert capped
    paths, capped = mrp.enumerate_paths(graph, 0, 5, 0, lambda n=1: None, 10 ** 9)
    assert not capped and paths == pc._find_all_simple_paths_impl(mol, 0, 5, ring_atoms, adj_list)


def test_uses_bound_asks_for_a_bound_only_where_it_can_cut(monkeypatch):
    graph = mrp.RingGraph({i: [(i - 1) % 50, (i + 1) % 50] for i in range(50)}, set(range(50)), 50)   # 50 atoms
    assert mrp.LEVEL_PRUNE_MAX_FREE == 48 and mrp.LEVEL_PRUNE_MAX_FRACTION == 0.6
    assert not mrp.uses_bound(graph, 0)                  # no bound asked for
    assert mrp.uses_bound(graph, 48 - 2)                 # 2 atoms left out
    assert mrp.uses_bound(graph, 50 - 2 - 30)            # 30 left out: the share limit 0.6
    assert not mrp.uses_bound(graph, 50 - 2 - 31)        # 31 left out: above it
    assert not mrp.uses_bound(graph, 1)                  # no incumbent (a ring of 3): nothing to cut
    monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FRACTION", 1.0)
    monkeypatch.setattr(mrp, "LEVEL_PRUNE_MAX_FREE", 40)
    assert mrp.uses_bound(graph, 50 - 2 - 40) and not mrp.uses_bound(graph, 50 - 2 - 41)
    # a bound that cuts less than one evaluation in BOUND_USELESS_PER, after BOUND_MIN_CHECKS, is off
    monkeypatch.setattr(mrp, "BOUND_MIN_CHECKS", 100)
    graph.bound_checks, graph.bound_cuts = 99, 0
    assert mrp.uses_bound(graph, 20)
    graph.bound_checks, graph.bound_cuts = 100, 0
    assert not mrp.uses_bound(graph, 20)
    graph.bound_checks, graph.bound_cuts = 100, 1
    assert mrp.uses_bound(graph, 20)                    # 1 cut in 100 checks is not less than 1 in 100
    graph.bound_checks, graph.bound_cuts = 100, 2
    assert mrp.uses_bound(graph, 20)


def test_a_bound_switched_off_part_of_the_way_does_not_change_the_result(levelled, units, monkeypatch):
    """The first evaluations of every ring system count; from the second one on the bound is found
    useless (``BOUND_USELESS_PER`` huge, the cuts of the first few aside): the later pairs list every
    path. The result is the reference's."""
    levelled(100)
    monkeypatch.setattr(mrp, "BOUND_MIN_CHECKS", 30)
    monkeypatch.setattr(mrp, "BOUND_USELESS_PER", 10 ** 9)
    stats = {"systems": 0}
    for seed in range(20):
        mol = _polyhex(random.Random(seed), random.Random(seed).randint(4, 7))
        for ref, new, _, _ in _both(mol, True, units):
            assert new == ref, seed
            stats["systems"] += 1
    assert stats["systems"] >= 20


def test_longest_bridge_is_the_longest_path_through_the_free_atoms():
    mol = Chem.MolFromSmiles("C1CC2CCC1CC2")                  # bicyclo[2.2.2]octane
    ring_atoms, adj_list, graph = _graph_of(mol)
    bridgeheads = sorted(i for i in ring_atoms if len(adj_list[i]) >= 3)
    bh1, bh2 = bridgeheads
    everything = graph.mask & ~(1 << bh1) & ~(1 << bh2)
    assert mrp.longest_bridge(graph, bh1, bh2, everything, lambda n=1: None) == 2
    assert mrp.longest_bridge(graph, bh1, bh2, 0, lambda n=1: None) == -1     # no atoms free, no direct bond


# ---------------------------------------------------------------------------
# (c) the cap, and the systems the levels do not touch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cap", [40, 400, 200_000])
def test_a_pair_reported_capped_is_searched_by_the_full_enumerations(cap, levelled, units, monkeypatch):
    """When the levelled listing of a pair exceeds the cap, the pair is searched as before: the full
    enumerations, each cut off where it always was (the reference's own, at the same cap)."""
    levelled(8)
    monkeypatch.setattr(pc, "_MAX_DFS_EXPANSIONS", cap)
    monkeypatch.setattr(pc, "enumerate_paths", lambda *a, **k: ([], True))
    stats = {"systems": 0, "cut_off": 0}
    for seed in range(25):
        rng = random.Random(3000 + seed)
        mol = _polyhex(rng, rng.randint(3, 6)) if seed % 2 else _random_cage(rng)
        for ref, new, _, _ in _both(mol, True, units):
            assert new == ref, (cap, seed)
            stats["systems"] += 1
    assert stats["systems"] >= 25


def test_the_cap_of_the_listing_is_the_caps_of_the_enumerations(monkeypatch):
    """``enumerate_paths`` stops at ``cap`` frames as ``_find_all_simple_paths_impl`` does, on the same
    frame: a list never longer than the full enumeration's own prefix and ``capped`` set."""
    mol = Chem.MolFromSmiles("C12C3C4C1C5C2C3C45")           # cubane
    ring_atoms, adj_list, graph = _graph_of(mol)
    for cap in (3, 7, 20):
        monkeypatch.setattr(pc, "_MAX_DFS_EXPANSIONS", cap)
        full = pc._find_all_simple_paths_impl(mol, 0, 5, ring_atoms, adj_list)
        paths, capped = mrp.enumerate_paths(graph, 0, 5, 0, lambda n=1: None, cap)
        assert paths == full and capped


def test_a_system_of_fewer_than_nine_rings_is_searched_exactly_as_before(units):
    """Under the shipped threshold: the same result in the same units (the preference off: no Case 3, so
    no enumeration is shared; the DFS and the pair loops are the reference's)."""
    assert mrp.LEVEL_MIN_RINGS == 9
    stats = {"systems": 0}
    for seed in range(40):
        mol = _random_cage(random.Random(seed))
        for ref, new, ref_units, new_units in _both(mol, False, units):
            assert new == ref
            assert new_units == ref_units
            stats["systems"] += 1
    assert stats["systems"] >= 40


def test_the_levelled_search_is_not_entered_below_the_threshold(monkeypatch, units):
    calls = []
    real = mrp.enumerate_paths
    monkeypatch.setattr(pc, "enumerate_paths", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    mol = Chem.MolFromSmiles("C1C2CC3CC1CC(C2)C3")            # adamantane, 3 rings
    analyzer = VonBaeyerAnalyzer()
    for ring_atoms, bridgeheads in _ring_systems(analyzer, mol):
        analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads)
    assert calls == []
    monkeypatch.setattr(mrp, "LEVEL_MIN_RINGS", 3)
    for ring_atoms, bridgeheads in _ring_systems(analyzer, mol):
        analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads)
    assert calls


# ---------------------------------------------------------------------------
# (d) the structure-keyed memo of an RWMol
# ---------------------------------------------------------------------------

def _rwmol_system(smiles):
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    analyzer = VonBaeyerAnalyzer()
    (ring_atoms, bridgeheads), = _ring_systems(analyzer, mol)
    return mol, analyzer, ring_atoms, bridgeheads


def test_an_rwmol_of_the_same_structure_is_searched_once_per_scope(monkeypatch):
    """The universal cage analysis hands every ``analyze`` a fresh RWMol of the same cage: one search,
    then hits, whichever RWMol object asks; a hit replays the units the search charged."""
    runs = []
    real = VonBaeyerAnalyzer._find_main_ring_impl
    monkeypatch.setattr(VonBaeyerAnalyzer, "_find_main_ring_impl",
                        lambda self, *a, **k: (runs.append(1), real(self, *a, **k))[1])
    smiles = "C1C2CC3CC1CC(C2)C3"
    mol_a, analyzer, ring_atoms, bridgeheads = _rwmol_system(smiles)
    mol_b = _rwmol_system(smiles)[0]
    assert mol_a is not mol_b
    fn._fragment_guard.perf_budget = 10 ** 9
    token = memo.push_scope()
    try:
        first = analyzer._find_main_ring(mol_a, ring_atoms, bridgeheads, return_candidates=True)
        spent_first = 10 ** 9 - fn._fragment_guard.perf_budget
        fn._fragment_guard.perf_budget = 10 ** 9
        again = analyzer._find_main_ring(mol_a, ring_atoms, bridgeheads, return_candidates=True)
        other = analyzer._find_main_ring(mol_b, ring_atoms, bridgeheads, return_candidates=True)
        spent_hits = 10 ** 9 - fn._fragment_guard.perf_budget
    finally:
        memo.pop_scope(token)
        fn._fragment_guard.perf_budget = None
    assert len(runs) == 1
    assert first == again == other == real(analyzer, mol_a, ring_atoms, bridgeheads, return_candidates=True)
    assert spent_first > 0 and spent_hits == 2 * spent_first     # a hit charges the units it replaces


def test_an_rwmol_is_not_served_the_value_of_the_other_main_bridge_preference():
    """``analyze`` runs the search with the preference for a 2+-atom main bridge on, and again with it off
    on a cage it cannot number; the value of one is not the value of the other, and the RWMol memo keys
    them apart. (The key of a plain Mol is as it was and does not: its first value serves both.)"""
    found = 0
    for seed in range(300):
        mol = Chem.RWMol(_random_cage(random.Random(seed)))
        analyzer = VonBaeyerAnalyzer()
        for ring_atoms, bridgeheads in _ring_systems(analyzer, mol):
            analyzer._allow_multiatom_bridge = True
            on = real_on = analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
            analyzer._allow_multiatom_bridge = False
            off = real_off = analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
            if real_on == real_off:
                continue
            found += 1
            token = memo.push_scope()
            try:
                analyzer._allow_multiatom_bridge = True
                on = analyzer._find_main_ring(mol, ring_atoms, bridgeheads, return_candidates=True)
                analyzer._allow_multiatom_bridge = False
                off = analyzer._find_main_ring(mol, ring_atoms, bridgeheads, return_candidates=True)
            finally:
                memo.pop_scope(token)
            assert on == real_on and off == real_off
    assert found >= 1


def test_an_edited_rwmol_is_not_served_the_value_of_the_structure_it_had():
    smiles = "C1CC2CCC1C2"
    mol, analyzer, ring_atoms, bridgeheads = _rwmol_system(smiles)
    token = memo.push_scope()
    try:
        before = analyzer._find_main_ring(mol, ring_atoms, bridgeheads, return_candidates=True)
        mol.GetAtomWithIdx(0).SetAtomicNum(7)                     # an edit in place changes the ranks
        after = analyzer._find_main_ring(mol, ring_atoms, bridgeheads, return_candidates=True)
    finally:
        memo.pop_scope(token)
    assert after == analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
    assert before == analyzer._find_main_ring_impl(Chem.RWMol(Chem.MolFromSmiles(smiles)), ring_atoms,
                                                    bridgeheads, return_candidates=True)
