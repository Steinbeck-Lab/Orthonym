"""The set-arithmetic pair searches of the von Baeyer main-ring search are exact.

 "Selection of the main ring" (``the Blue Book Blue Book``), "Selection of the
main bridge" (``:9603``) and "Selection of the main bridge and secondary bridges", rule
 (``:9661``) fix which pair of interior-disjoint bridgehead paths is the main ring. The
two loops that chose it by testing every pair of the paths of a bridgehead pair
(``polycyclic._best_disjoint_path_pair``, Cases 1 and 2, and ``polycyclic._consider_multiatom_bridges``,
Case 3) are quadratic in the number of paths, which is in the thousands for a ring system of 13 or more
rings. ``rules/main_ring_pairs.py`` finds the same pair from the sets of paths through each atom.
Nothing it returns may differ from the loop's, so these tests run both:

* on random path sets (interiors from a small pool, so that overlaps, equal lengths and exact ties of
  every kind occur), the returned pair (the same list entries, not an equal pair), and for Case 3 the
  final state of ``_MainRingBest`` (score, ring, bridgehead pair and the tied list in its order) and the
  sequence of the calls of ``bridge_len``;
* on the main-ring search of real cages with the set searches forced on every list
  (``PAIR_SCAN_MAX_PATHS = 0``), against the loops (``10 ** 9``): the main ring, the bridgehead pair
  and every oriented tied candidate in order;
* on what the perf budget is charged: never more than the loop charged, and fewer on a long list, so a
  search that ran out of the 12M budget at one unit per pair can finish.

No JVM is needed.
"""
import random

import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.rules import main_ring_pairs as mrp
from orthonym.rules import polycyclic as pc

pytestmark = pytest.mark.unit


def _paths(rng, big):
    """bh1 (0) -> bh2 (1) paths with interiors drawn from a small pool, and a random total rank order
    over the atoms."""
    universe = rng.randint(6, 44 if big else 14)
    paths = []
    for _ in range(rng.randint(2, 300 if big else 16)):
        interior = rng.sample(range(2, universe), rng.randint(0, min(14 if big else 6, universe - 2)))
        paths.append([0] + interior + [1])
    ranks = list(range(max(universe, 20)))
    rng.shuffle(ranks)
    return paths, (lambda atoms: tuple(sorted(ranks[a] for a in atoms))), universe


class _Spend:
    def __init__(self):
        self.units = 0

    def __call__(self, n=1):
        self.units += n


# ---------------------------------------------------------------------------
# Cases 1 and 2: the best pair
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("big", [False, True], ids=["short", "long"])
def test_best_pair_by_sets_is_the_pair_of_the_loop(big):
    ties = fewer = 0
    for seed in range(300):
        paths, rank_key, _ = _paths(random.Random(seed), big)
        plens = mrp.interior_masks(paths)[1]
        loop_spend, set_spend = _Spend(), _Spend()
        expected = mrp._best_disjoint_pair_loop(paths, loop_spend, rank_key)
        found = mrp._best_disjoint_pair_sets(paths, plens, set_spend, rank_key)
        if expected is None:
            assert found is None
        else:
            assert found is not None
            assert found[0] is expected[0] and found[1] is expected[1], seed   # the same list entries
            ties += len(set(map(tuple, paths))) < len(paths)
        assert set_spend.units <= loop_spend.units
        fewer += set_spend.units < loop_spend.units
    assert fewer > 100                 # a long list is where the sets are cheaper
    assert ties > 0 or not big         # equal paths occur in the long lists


def test_best_pair_routes_by_the_length_of_the_list(monkeypatch):
    paths, rank_key, _ = _paths(random.Random(7), True)
    monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", len(paths) + 1)
    by_loop = _Spend()
    expected = pc._best_disjoint_path_pair(paths, by_loop, rank_key)
    assert by_loop.units == len(paths) * (len(paths) - 1) // 2          # the loop: one unit per pair
    monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", len(paths))
    by_sets = _Spend()
    found = pc._best_disjoint_path_pair(paths, by_sets, rank_key)
    assert found == expected and by_sets.units < by_loop.units


def test_best_pair_of_fewer_than_two_paths():
    for paths in ([], [[0, 2, 1]]):
        assert mrp.best_disjoint_pair(paths, _Spend(), lambda atoms: tuple(sorted(atoms))) is None


# ---------------------------------------------------------------------------
# Case 3: the pairs offered to the main bridge search
# ---------------------------------------------------------------------------

def _case3(paths, rank_key, room, bridge_len_for, seed, threshold, monkeypatch):
    monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", threshold)
    best = pc._MainRingBest(rank_key)
    r = random.Random(seed + 1)
    if r.random() < 0.4:                                    # an incumbent from Cases 1 and 2
        best.consider(list(range(r.randint(4, 14))), 0, 1, r.randint(0, 5), r.randint(0, 5), r.randint(0, 3))
    calls = []
    spend = _Spend()
    pc._consider_multiatom_bridges([list(p) for p in paths], spend, best, 0, 1, room,
                                   bridge_len_for(calls))
    return (best.score, best.ring, best.bh_pair, best.tied), calls, spend.units


@pytest.mark.parametrize("mode", ["scan", "bridge_len"])
@pytest.mark.parametrize("big", [False, True], ids=["short", "long"])
def test_case3_by_sets_offers_what_the_loop_offers(big, mode, monkeypatch):
    tied_more = fewer = 0
    for seed in range(250):
        paths, rank_key, _ = _paths(random.Random(seed), big)
        room = len({a for p in paths for a in p[1:-1]}) + random.Random(seed).randint(0, 2)

        def bridge_len_for(calls, salt=seed):
            if mode == "scan":
                return None

            def bridge_len(used):                           # any function of the free atoms will do
                calls.append(used)
                return hash((used, salt)) % 7 - 1
            return bridge_len

        expected, calls_loop, units_loop = _case3(paths, rank_key, room, bridge_len_for, seed, 10 ** 9, monkeypatch)
        found, calls_sets, units_sets = _case3(paths, rank_key, room, bridge_len_for, seed, 0, monkeypatch)
        assert found == expected, seed
        assert calls_sets == calls_loop, seed               # the longest-bridge search is asked the same, in order
        assert units_sets <= units_loop, seed
        if big and units_loop > 100:
            assert units_sets > 0, seed                      # the search is charged what it does
        tied_more += len(expected[3]) > 1
        fewer += units_sets < units_loop
    assert tied_more > 20
    assert fewer > (100 if big else 50)


def test_case3_routes_by_the_length_of_the_list(monkeypatch):
    paths, rank_key, _ = _paths(random.Random(11), True)
    room = len({a for p in paths for a in p[1:-1]})

    def no_bridge_len(calls):
        return None

    by_loop = _case3(paths, rank_key, room, no_bridge_len, 11, len(paths) + 1, monkeypatch)
    reference = _case3(paths, rank_key, room, no_bridge_len, 11, 10 ** 9, monkeypatch)
    assert by_loop == reference                                   # one path shorter than the threshold: the loop
    by_sets = _case3(paths, rank_key, room, no_bridge_len, 11, len(paths), monkeypatch)
    assert by_sets[0] == reference[0] and by_sets[2] < reference[2]   # at the threshold: the sets, fewer units


# ---------------------------------------------------------------------------
# the main-ring search of real cages
# ---------------------------------------------------------------------------

CAGES = [
    "C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3",                      # tetracyclo[6.3.3.2^3,6.1^2,6]heptadecane
    "C12CC34CCCC(CC(CCCC1)CCC2)(CC3)C4",
    "C12C3C4C1C5C2C3C45",                                   # cubane
    "C12C3C4C1C1C2C3C41",                                   # prismane-like cage of 8 atoms
    "C1C2C3CC4C1C2C34",
    "C12C3C4C5C1C1C2C3C4C51",                               # a 10-atom cage with 6 rings
    "C1CC2CC3CC2CC13",
    "C12C3CC4CC(C1)CC(C2)C34",
    "O1[SiH2]O[SiH]2O[SiH2]O[SiH]1O2",
]


def _main_ring(mol, threshold, monkeypatch):
    monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", threshold)
    analyzer = pc.VonBaeyerAnalyzer()
    spent = _Spend()
    monkeypatch.setattr(pc, "spend_perf_work", spent)
    out = []
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for system in [s for s in systems if s & merged]:
            systems.remove(system)
            merged |= system
        systems.append(merged)
    for ring_atoms in sorted(systems, key=lambda s: -len(s)):
        bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
        if len(bridgeheads) >= 2:
            out.append(analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True))
    return out, spent.units


@pytest.mark.parametrize("smiles", CAGES)
def test_main_ring_search_of_a_cage_is_the_same_with_the_set_searches(smiles, monkeypatch):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    with_loops, units_loops = _main_ring(mol, 10 ** 9, monkeypatch)
    with_sets, units_sets = _main_ring(mol, 0, monkeypatch)
    assert with_loops and with_sets == with_loops
    assert units_sets <= units_loops


def _random_cage(rng):
    """A random connected skeleton (spanning tree plus 2-5 extra edges, degree <= 4), with heteroatoms
    on low-degree atoms so that canonical-rank ties and non-ties both occur."""
    n = rng.randint(8, 13)
    edges = set()
    degree = [0] * n
    for v in range(1, n):
        u = rng.choice([u for u in range(v) if degree[u] < 4])
        edges.add((u, v))
        degree[u] += 1
        degree[v] += 1
    extra = rng.randint(2, 5)
    for _ in range(60):
        if extra == 0:
            break
        u, v = sorted(rng.sample(range(n), 2))
        if (u, v) in edges or degree[u] >= 4 or degree[v] >= 4:
            continue
        edges.add((u, v))
        degree[u] += 1
        degree[v] += 1
        extra -= 1
    rw = Chem.RWMol()
    for v in range(n):
        symbols = ["C", "N", "O", "S"] if degree[v] <= 2 else (["C", "N"] if degree[v] == 3 else ["C"])
        rw.AddAtom(Chem.Atom(rng.choice(symbols)))
    for u, v in sorted(edges):
        rw.AddBond(u, v, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("seed", range(60))
def test_main_ring_search_of_a_random_cage_is_the_same_with_the_set_searches(seed, monkeypatch):
    """Random polycyclic skeletons, both settings of the main-bridge preference (``analyze`` runs the
    cascade again with it off when the preferred main bicycle cannot be numbered)."""
    mol = _random_cage(random.Random(seed))
    for allow in (True, False):
        found = {}
        for threshold in (10 ** 9, 0):
            monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", threshold)
            analyzer = pc.VonBaeyerAnalyzer()
            analyzer._allow_multiatom_bridge = allow
            spent = _Spend()
            monkeypatch.setattr(pc, "spend_perf_work", spent)
            systems = []
            for ring in mol.GetRingInfo().AtomRings():
                merged = set(ring)
                for system in [x for x in systems if x & merged]:
                    systems.remove(system)
                    merged |= system
                systems.append(merged)
            out = []
            for ring_atoms in sorted(systems, key=lambda x: -len(x)):
                bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
                if len(bridgeheads) >= 2:
                    out.append(analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True))
            found[threshold] = (out, spent.units)
            monkeypatch.undo()
        assert found[0][0] == found[10 ** 9][0], (seed, allow)
        assert found[0][1] <= found[10 ** 9][1]


def test_a_search_over_the_budget_of_the_loops_finishes_with_the_sets(monkeypatch):
    """A budget between the two counts: the loops run out of it (the abstention at ``perf_op_budget``),
    the set searches finish, with the result the loops give when they are not limited."""
    mol = Chem.MolFromSmiles("C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3")
    analyzer = pc.VonBaeyerAnalyzer()
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
    monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", 10 ** 9)
    spent = _Spend()
    monkeypatch.setattr(pc, "spend_perf_work", spent)
    expected = analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
    units_loops = spent.units
    monkeypatch.undo()
    monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", 0)
    spent = _Spend()
    monkeypatch.setattr(pc, "spend_perf_work", spent)
    analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
    units_sets = spent.units
    monkeypatch.undo()
    assert units_sets < units_loops

    def run(threshold, budget):
        monkeypatch.setattr(mrp, "PAIR_SCAN_MAX_PATHS", threshold)
        fn._fragment_guard.perf_budget = budget
        try:
            return analyzer._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
        except fn.PerfBudgetExceeded:
            return "exceeded"
        finally:
            fn._fragment_guard.perf_budget = None
            monkeypatch.undo()

    between = units_sets + 1
    assert run(10 ** 9, between) == "exceeded"
    assert run(0, between) == expected
    assert run(0, units_sets) == "exceeded"


# ---------------------------------------------------------------------------
# a cage above the PIN ceilings is not labelled a PIN on the path of name_polycyclic_complete
# ---------------------------------------------------------------------------

def _recorded_non_pin(smiles, monkeypatch, max_rings):
    from orthonym.metrics import provenance
    from orthonym.rules import vonbaeyer_universal as vu

    monkeypatch.setattr(vu, "MAX_CAGE_RINGS", max_rings)
    provenance.clear_provenance()
    name = pc.name_polycyclic_complete(Chem.MolFromSmiles(smiles))[0]
    return name, provenance.get_provenance()["non_pin_fragments"]


def test_a_cage_above_the_pin_ring_ceiling_is_recorded_as_not_a_pin(monkeypatch):
    """ (``the Blue Book``) is not applied to every numbering of a cage that only the
    lifted ceilings analyse, so its descriptor is recorded as a non-PIN fragment (the name is unchanged)."""
    smiles = "C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3"                 # tetracyclo[6.3.3.2^3,6.1^2,6]heptadecane, 4 rings
    name, within = _recorded_non_pin(smiles, monkeypatch, max_rings=8)
    name_above, above = _recorded_non_pin(smiles, monkeypatch, max_rings=3)
    assert name_above == name and name.startswith("tetracyclo[")
    descriptor = name[:name.index("]") + 1]
    assert not any(descriptor in fragment for fragment in within)
    assert any(descriptor in fragment for fragment in above)
