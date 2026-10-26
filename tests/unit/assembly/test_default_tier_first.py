"""A producer of the best-effort tier that has a default-tier route asks that route first (lane TI2).

``assembly.default_tier_scope.default_tier_scope`` is the context a producer is asked under when it is asked the
default tier's way, ``assembly.default_tier_offer.default_tier_first`` the order (the default tier's reading first,
the wider reading where the default tier declines), and ``deferred_handler_row`` the safety net: a handler that
declined the molecule the default tier's way and has a wider reading is asked again with the wider reading
admitted when the other routes leave the molecule without a name. The cases of the invariant are in
tests/unit/test_tier_inversions.py; these are the mechanisms, on stubs, so that a mutation of one of them is
seen without a JVM.
"""
import pytest

from orthonym.assembly import default_tier_offer as offer
from orthonym.assembly.default_tier_scope import default_tier_scope, wider_tier_running
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.unit

_BREADTH = (pv.general_fallback_ctx, pv.best_effort_ctx, pv.allow_aromatic_general_ctx, pv.full_coverage_ctx)


@pytest.fixture(autouse=True)
def _no_held_state():
    """The deferred handlers of an earlier naming of the process are not this test's."""
    offer.clear_held()
    yield
    offer.clear_held()


@pytest.fixture
def best_effort():
    """The context of a best-effort call: the four breadth variables on."""
    tokens = [(var, var.set(True)) for var in _BREADTH]
    offer.clear_held()
    try:
        yield
    finally:
        offer.clear_held()
        for var, token in reversed(tokens):
            var.reset(token)


def test_the_scope_puts_the_four_breadth_variables_off_and_back(best_effort):
    assert wider_tier_running()
    with default_tier_scope():
        assert [var.get() for var in _BREADTH] == [False] * 4
        assert not wider_tier_running()
    assert [var.get() for var in _BREADTH] == [True] * 4


def test_the_default_tier_is_not_a_wider_tier():
    assert not wider_tier_running()


def test_outside_a_wider_tier_the_producer_is_asked_as_it_is():
    seen = []
    assert offer.default_tier_first(lambda: seen.append(wider_tier_running()) or "name", "organometallic") == "name"
    assert seen == [False] and offer._DEFERRED_HANDLERS.get() == ()


def test_the_default_tier_reading_is_asked_first_and_wins(best_effort):
    seen = []

    def produce():
        seen.append(wider_tier_running())
        return "default-tier name" if not wider_tier_running() else "wider name"
    assert offer.default_tier_first(produce, "poly_anion") == "default-tier name"
    assert seen == [False]                  # the wider reading is not asked once the default tier names it
    assert offer._DEFERRED_HANDLERS.get() == ()


def test_where_the_default_tier_declines_the_handler_is_deferred(best_effort):
    seen = []

    def produce():
        seen.append(wider_tier_running())
        return None if not wider_tier_running() else "wider name"
    assert not offer.default_tier_first(produce, "organometallic")
    assert seen == [False] and offer._DEFERRED_HANDLERS.get() == ("organometallic",)


def test_the_handler_is_deferred_only_where_a_part_was_declined(best_effort):
    seen = []

    def produce():
        seen.append(wider_tier_running())
        if not wider_tier_running():
            offer.note_declined_part("ligand")
            return None
        return "wider name"
    assert not offer.default_tier_first(produce, "organometallic", "ligand")
    assert offer._DEFERRED_HANDLERS.get() == ("organometallic",) and seen == [False]
    offer.clear_held()
    # another part declined: the wider reading is asked now, as it always was
    seen.clear()
    assert offer.default_tier_first(produce, "organometallic", "owner") == "wider name"
    assert offer._DEFERRED_HANDLERS.get() == () and seen == [False, True]


def test_the_deferral_condition_is_read_on_a_decline(best_effort):
    def produce():
        if not wider_tier_running():
            offer.note_declined_part("ligand")
            return None
        return "wider name"
    # no condition met: the wider reading now
    assert offer.default_tier_first(produce, "organometallic", "ligand", lambda: False) == "wider name"
    assert offer._DEFERRED_HANDLERS.get() == ()
    assert not offer.default_tier_first(produce, "organometallic", "ligand", lambda: True)
    assert offer._DEFERRED_HANDLERS.get() == ("organometallic",)


def test_the_admitted_handler_is_asked_with_the_wider_reading(best_effort):
    token = offer._ADMITTED_HANDLERS.set(("organometallic",))
    try:
        assert offer.handler_admitted("organometallic") and not offer.handler_admitted("poly_anion")
        assert offer.default_tier_first(lambda: wider_tier_running(), "organometallic") is True
        # a handler that is not admitted is still asked the default tier's way
        assert offer.default_tier_first(lambda: wider_tier_running(), "poly_anion") is False
    finally:
        offer._ADMITTED_HANDLERS.reset(token)


def test_a_deferred_handler_is_forgotten_at_the_start_of_the_next_naming(best_effort):
    offer.defer_handler("organometallic")
    offer.defer_handler("organometallic")
    assert offer._DEFERRED_HANDLERS.get() == ("organometallic",)
    offer.clear_held()
    assert offer._DEFERRED_HANDLERS.get() == ()


def test_no_deferred_handler_no_second_pass():
    class _Namer:
        _general_fallback_unverified = True
        _forced_parent_rank = 0
    assert offer.deferred_handler_row(_Namer(), "CCO") is None
