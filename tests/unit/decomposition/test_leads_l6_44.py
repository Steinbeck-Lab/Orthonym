"""Leads program L6, item 44: the cut-off of the unproven-split search belongs to its budget.

Once a molecule has met an unproven split, its searches spend only a share of the analysis-call
budget on other bonds (``engine._exploration_exhausted``): the cut-off is a LEVEL of that
budget (62 of 500 calls below where the first unproven split was begun). It used to be kept as
a side value of the naming scope (``assembly.memo``), in absolute budget units, while the
budgets are armed afresh inside one scope: ``own_hang_budgets`` (the component of an adduct, a
compound of its own, the Blue Book) and ``rearm_hang_budgets`` (the last-resort
 rescue and the exit re-arm of the outermost ``name``). A fresh budget of 500 then
inherited the old level (433) and read as spent after 70 calls of it, and a level noted inside
the body reached the enclosing budget it was put back to. The level now lives on
``fragment_naming._fragment_guard`` beside ``analysis_budget``, is reset wherever the budget is
armed, and is saved, cleared and restored by ``own_hang_budgets`` and
``speculative_fragment_naming``.

Part (1): the share is counted from the START of the first unproven split, so a split that
cost 7 calls left 55 calls of exploration (62 - 7) although what an exploration that finds a name
needs is a few calls; a 129-atom molecule at best-effort kept opening other bonds for about 15 s.
``engine._UNPROVEN_EXPLORATION_AFTER_SPLIT`` caps the exploration at that many calls after the END
of the split; the share from the start stays as the other bound (an expensive first split
opens nothing more, as before). The constant is the measured one: 91 molecules of the eval sets
that meet an unproven split x 3 tiers keep every name and label at 16, 8, 4 and 0 (recorded in
internal notes).

Speed and breadth only, no naming rule; the budgets count operations, never time.

Mutation checks (run once, recorded in internal notes): keeping the level in
the memo scope again (``engine._note_unproven_split`` / ``_exploration_exhausted`` on
``memo.side_put`` / ``side_get``) fails the part (2) tests; setting
``engine._UNPROVEN_EXPLORATION_AFTER_SPLIT`` out of reach fails the part (1) tests.
"""
import pytest

from orthonym.assembly import fragment_naming as fn
from orthonym.decomposition import engine
from tests.unit.decomposition.test_unproven_split_search import _GOOD, _search

CEILING = 500


@pytest.fixture
def scope():
    """An outermost name scope with its budgets armed (what ``Orthonym.name`` opens)."""
    assert fn.name_scope_depth() == 0
    fn.enter_name_scope()
    try:
        yield
    finally:
        fn.exit_name_scope()
        fn._fragment_guard.unproven_floor = None


def _spend(calls):
    fn.spend_analysis_call(calls)


def _first_unproven_split(cost):
    """Begin an unproven split at the current level of the budget and let it cost ``cost``
    analysis calls (what ``try_decompose`` does around an assembler that proves nothing)."""
    before = fn._fragment_guard.analysis_budget
    _spend(cost)
    engine._note_unproven_split(before)


def test_the_share_of_a_budget_is_counted_from_its_own_first_unproven_split(scope):
    """An expensive first split (50 of the 62 calls of the share): the share is counted from
    the start of the split, as before, and is the tighter of the two bounds."""
    assert engine._exploration_exhausted() is False       # none met yet: no share
    _first_unproven_split(50)                             # 500 -> 450, floor 500 - 62 = 438
    assert fn._fragment_guard.unproven_floor == CEILING - CEILING // 8
    assert engine._exploration_exhausted() is False       # 450 > 438
    _spend(12)                                            # 438 <= 438
    assert engine._exploration_exhausted() is True


def test_a_first_split_that_costs_more_than_the_share_opens_nothing_more(scope):
    """The split's own cost is part of the share: 104 calls (the first split of a 126-atom
    alcohol) leave nothing, however small the cap after the split is."""
    _first_unproven_split(104)
    assert fn._fragment_guard.unproven_floor == CEILING - CEILING // 8
    assert engine._exploration_exhausted() is True


def test_a_cheap_first_split_leaves_only_the_cap_after_it(scope):
    """A first split that costs 7 calls used to leave 55 calls of exploration (62 - 7), the
    whole share, although what an exploration that finds a name needs is a few calls: the
    cap counts from the END of the split."""
    cap = engine._UNPROVEN_EXPLORATION_AFTER_SPLIT
    assert cap < CEILING // engine._UNPROVEN_EXPLORATION_SHARE - 7      # the cap bites here
    _first_unproven_split(7)                              # 500 -> 493
    assert fn._fragment_guard.unproven_floor == CEILING - 7 - cap
    _spend(cap - 1)
    assert engine._exploration_exhausted() is False
    _spend(1)
    assert engine._exploration_exhausted() is True


def test_a_rearmed_budget_has_met_no_unproven_split(scope):
    """``rearm_hang_budgets`` (the rescue after a budget trip, the exit re-arm): a fresh
    ceiling of 500 used to inherit the level noted for the old one and read as spent after
    70 calls (500 -> 430 <= 433)."""
    _first_unproven_split(5)                              # floor 500 - 62 = 438... of the old budget
    old_floor = fn._fragment_guard.unproven_floor
    assert old_floor is not None
    fn.rearm_hang_budgets()
    assert fn._fragment_guard.analysis_budget == CEILING
    assert fn._fragment_guard.unproven_floor is None
    _spend(CEILING - old_floor + 5)                       # more than the old level's share
    assert engine._exploration_exhausted() is False       # the new budget met no unproven split
    # and the new budget gets a share of its own at its own first unproven split
    _first_unproven_split(3)
    assert fn._fragment_guard.unproven_floor == fn._fragment_guard.analysis_budget - (
        engine._UNPROVEN_EXPLORATION_AFTER_SPLIT)


def test_a_component_budget_neither_inherits_nor_leaks_a_level(scope):
    """``own_hang_budgets`` (each component of an adduct): inside it the budget is fresh and
    has met no unproven split; the level it notes is its own and is gone when the enclosing
    budget is put back."""
    _first_unproven_split(5)
    outer_floor = fn._fragment_guard.unproven_floor
    outer_budget = fn._fragment_guard.analysis_budget
    with fn.own_hang_budgets():
        assert fn._fragment_guard.analysis_budget == CEILING
        assert fn._fragment_guard.unproven_floor is None
        _spend(CEILING - outer_floor + 5)                 # past the enclosing level
        assert engine._exploration_exhausted() is False   # inherited it would read as spent
        _first_unproven_split(4)
        inner_floor = fn._fragment_guard.unproven_floor
        assert inner_floor is not None
        assert inner_floor != outer_floor
    assert fn._fragment_guard.analysis_budget == outer_budget
    assert fn._fragment_guard.unproven_floor == outer_floor     # the body's level did not leak
    assert engine._exploration_exhausted() is False             # outer: 495 > 438


def test_a_disarmed_budget_stays_without_a_level(scope):
    fn.disarm_hang_budgets()
    engine._note_unproven_split(None)                     # no armed budget: no level
    assert fn._fragment_guard.unproven_floor is None
    assert engine._exploration_exhausted() is False


def test_the_outermost_scope_starts_and_ends_without_a_level():
    fn._fragment_guard.unproven_floor = 123               # left over from a molecule before
    fn.enter_name_scope()
    try:
        assert fn._fragment_guard.unproven_floor is None
        _first_unproven_split(5)
        assert fn._fragment_guard.unproven_floor is not None
        fn.enter_name_scope()                             # a nested name shares the budget
        try:
            assert fn._fragment_guard.unproven_floor is not None
        finally:
            fn.exit_name_scope()
    finally:
        fn.exit_name_scope()
    assert fn._fragment_guard.unproven_floor is None


def test_a_speculative_naming_leaves_no_level_behind(scope):
    """The body of ``speculative_fragment_naming`` works on copies of the budgets; a level it
    notes is a level of those copies."""
    with fn.speculative_fragment_naming():
        _first_unproven_split(5)
        assert fn._fragment_guard.unproven_floor is not None
    assert fn._fragment_guard.unproven_floor is None
    assert engine._exploration_exhausted() is False


def test_the_cap_ends_a_search_the_share_would_have_left_open():
    """``try_decompose`` on three bonds, the first two unproven (``_search`` of the unproven-split
    tests, whose share is 31 calls): 3 + 17 calls spend 20 of the 31, but the second split
    ended 17 calls after the first, past the cap of 16: the third bond is not opened."""
    cap = engine._UNPROVEN_EXPLORATION_AFTER_SPLIT
    result, attempted = _search(spend_on_first=3, spend_on_second=cap + 1)
    assert attempted == [0, 1] and result is None
    # the same search with the second split inside the cap opens the third bond
    result, attempted = _search(spend_on_first=3, spend_on_second=cap - 1)
    assert attempted == [0, 1, 2] and result == _GOOD
