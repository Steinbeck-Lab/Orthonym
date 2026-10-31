"""A main-ring search that ran the perf budget out is not run again by the rescues of the same naming (SPD2).

``VonBaeyerAnalyzer._find_main_ring`` stores a search that finishes (and replays its units on a hit), but not one
that raises ``PerfBudgetExceeded``. The outermost ``name`` then offers a rescue and a run under the PIN-tier
ceilings (``budget_rescue.pin_ceiling_name``): a whole naming that reaches the same search with the same structure and
the same budget, and runs it out again (measured: 35% of the wall time of the search-bound rows of PubChem, which spend
the whole 12M budget twice). The search is a function of the structure and of the budget it starts with: a trip is the
first charge that brings the budget to 0 or below, and the charges do not depend on the budget. So a search that starts
with no more budget than one that ran out ends as that one did (budget 0, one more trip counted, the same exception), and
the search is not run. Counts of calls and units, no clock.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from orthonym.rules.polycyclic import VonBaeyerAnalyzer

pytestmark = pytest.mark.unit

#: adamantane: a cage that is searched (tricyclic, four bridgeheads) and spends a few thousand units
CAGE = "C1C2CC3CC1CC(C2)C3"


def _setup(smiles=CAGE):
    mol = Chem.MolFromSmiles(smiles)
    analyzer = VonBaeyerAnalyzer()
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    return mol, analyzer, ring_atoms, analyzer._find_all_bridgeheads(mol, ring_atoms)


@pytest.fixture(autouse=True)
def _clean_state():
    fn._fragment_guard.main_ring_trips = None
    yield
    fn._fragment_guard.main_ring_trips = None
    fn._fragment_guard.perf_budget = None


@pytest.fixture
def impl_calls(monkeypatch):
    calls = []
    real = VonBaeyerAnalyzer._find_main_ring_impl
    monkeypatch.setattr(VonBaeyerAnalyzer, "_find_main_ring_impl",
                        lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1])
    return calls


def _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads):
    fn._fragment_guard.perf_budget = 10 ** 9
    token = memo.push_scope()
    try:
        analyzer._find_main_ring(mol, ring_atoms, bridgeheads)
    finally:
        memo.pop_scope(token)
    spent = 10 ** 9 - fn._fragment_guard.perf_budget
    fn._fragment_guard.perf_budget = None
    return spent


def _ask(analyzer, mol, ring_atoms, bridgeheads, budget):
    """One search with ``budget``: (outcome, budget left, trips counted)."""
    fn._fragment_guard.perf_budget = budget
    trips = fn.hang_budget_trips()
    token = memo.push_scope()
    try:
        try:
            value = analyzer._find_main_ring(mol, ring_atoms, bridgeheads, return_candidates=True)
        except fn.PerfBudgetExceeded:
            return "trip", fn._fragment_guard.perf_budget, fn.hang_budget_trips() - trips
        return value, fn._fragment_guard.perf_budget, fn.hang_budget_trips() - trips
    finally:
        memo.pop_scope(token)


def test_a_second_search_with_no_more_budget_ends_as_the_first_did_without_running(impl_calls):
    mol, analyzer, ring_atoms, bridgeheads = _setup()
    units = _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads)
    assert units > 10
    impl_calls.clear()
    budget = units // 2
    first = _ask(analyzer, mol, ring_atoms, bridgeheads, budget)
    assert first == ("trip", 0, 1) and len(impl_calls) == 1
    # the rescue of the same naming: a fresh memo scope, the same structure (a new mol object), the same budget
    again = _ask(analyzer, Chem.MolFromSmiles(CAGE), ring_atoms, bridgeheads, budget)
    smaller = _ask(analyzer, mol, ring_atoms, bridgeheads, budget - 3)
    assert again == ("trip", 0, 1) and smaller == ("trip", 0, 1)
    assert len(impl_calls) == 1, "the search that ran the budget out was run again"


def test_a_search_with_more_budget_than_the_one_that_ran_out_is_run(impl_calls):
    """The trip says what a budget of that size cannot do, not what a larger one can."""
    mol, analyzer, ring_atoms, bridgeheads = _setup()
    units = _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads)
    impl_calls.clear()
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, units // 2)[0] == "trip"
    value, left, trips = _ask(analyzer, mol, ring_atoms, bridgeheads, units + 5)
    assert len(impl_calls) == 2 and value != "trip" and trips == 0 and left == 5
    # and the finished search is stored as before: a hit replays its units and runs nothing
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, units + 5)[1:] == (5, 0)


def test_the_trip_of_one_structure_is_not_the_trip_of_another(impl_calls):
    mol, analyzer, ring_atoms, bridgeheads = _setup()
    units = _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads)
    other, other_analyzer, other_atoms, other_heads = _setup("C1C2CC3CC1C2C3")      # nortricyclane-like cage
    impl_calls.clear()
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, units // 2)[0] == "trip"
    _ask(other_analyzer, other, other_atoms, other_heads, 10 ** 9)
    assert len(impl_calls) == 2


def test_the_main_bridge_preference_is_part_of_the_key(impl_calls):
    mol, analyzer, ring_atoms, bridgeheads = _setup()
    units = _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads)
    impl_calls.clear()
    analyzer._allow_multiatom_bridge = True
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, units // 2)[0] == "trip"
    analyzer._allow_multiatom_bridge = False
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, units // 2)[0] == "trip"
    assert len(impl_calls) == 2


def test_the_memory_of_a_trip_ends_with_the_outermost_name_scope():
    mol, analyzer, ring_atoms, bridgeheads = _setup()
    units = _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads)
    fn.enter_name_scope()
    try:
        fn._fragment_guard.perf_budget = units // 2
        with pytest.raises(fn.PerfBudgetExceeded):
            analyzer._find_main_ring(mol, ring_atoms, bridgeheads)
        assert fn.main_ring_trips()
    finally:
        fn.exit_name_scope()
    assert fn.main_ring_trips() == {}


# ---------------------------------------------------------------------------
# a ring system of LEVEL_MIN_RINGS rings or more: the pairs still to come are units still to be charged
# ---------------------------------------------------------------------------

def _polyhex_system(seed=1, hexagons=10):
    import random
    from tests.unit.rules.test_main_ring_levels import _polyhex
    mol = _polyhex(random.Random(seed), hexagons)
    analyzer = VonBaeyerAnalyzer()
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    return mol, analyzer, ring_atoms, analyzer._find_all_bridgeheads(mol, ring_atoms)


def test_a_search_of_many_rings_that_ran_out_early_certifies_the_pairs_to_come(impl_calls):
    """The rescue starts the same search with a few dozen units more than the main path did. The units that the pairs
    still to come charge (one at least each) are charged by that search too, so it ends as the first did."""
    mol, analyzer, ring_atoms, bridgeheads = _polyhex_system()
    assert len(bridgeheads) >= 8
    units = _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads)
    assert units > 4000
    impl_calls.clear()
    budget = 3000                      # runs out in the first pairs of the search
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, budget) == ("trip", 0, 1)
    pairs = len(bridgeheads) * (len(bridgeheads) - 1) // 2
    reach = fn.main_ring_trips()
    assert len(reach) == 1 and budget < next(iter(reach.values())) <= budget + pairs
    # a budget above the first one and within the pairs to come: it trips, and the search is not run
    for more in (1, 48, 50, min(pairs - 1, 120)):
        assert _ask(analyzer, mol, ring_atoms, bridgeheads, budget + more) == ("trip", 0, 1), more
    assert len(impl_calls) == 1
    # a budget beyond what the pairs to come certify: the search is run, as before
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, budget + pairs + 5000)[0] != "trip" or len(impl_calls) == 2
    assert len(impl_calls) == 2


def test_every_budget_up_to_the_certified_reach_ends_as_the_search_does(monkeypatch):
    """No budget in the certified range gives another outcome than the search itself (the memo off): for every budget
    from the first trip's to its reach, the result, the budget left and the trips counted are those of a search."""
    mol, analyzer, ring_atoms, bridgeheads = _polyhex_system(seed=2, hexagons=10)
    units = _units_of_the_search(mol, analyzer, ring_atoms, bridgeheads)
    budget = max(units // 3, 500)
    assert _ask(analyzer, mol, ring_atoms, bridgeheads, budget)[0] == "trip"
    reach = next(iter(fn.main_ring_trips().values()))
    assert reach > budget
    for probe in sorted({budget, budget + 1, (budget + reach) // 2, reach - 1, reach}):
        with_memo = _ask(analyzer, mol, ring_atoms, bridgeheads, probe)
        fn._fragment_guard.main_ring_trips = {}              # the memo off: the search itself
        without_memo = _ask(analyzer, mol, ring_atoms, bridgeheads, probe)
        fn._fragment_guard.main_ring_trips = None
        assert with_memo == without_memo, (probe, with_memo, without_memo)
        assert _ask(analyzer, mol, ring_atoms, bridgeheads, budget)[0] == "trip"      # restore the memo
