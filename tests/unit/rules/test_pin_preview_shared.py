"""The preview of the promotion calls (``rules.pin_vocabulary.promotion_could_change``) names what the calls of one family
ask for once.

The first run of a molecule reaches ``promote_at_pin_tier`` again and again with the same fragment (42 recorded calls of
one fluorescein phosphate ask for 8 distinct memo entries). The preview of each call used to open a copy of the memo scope
of its own, so every call named its fragments again. The calls that ran in the same memo scope, at the same fragment-
recursion depth and in the same ambient naming state (``assembly.nested_memo.ambient_naming_state``: the visited set, the
tier flags and the context variables a naming path branches on) now share one sandbox; the answer of every call is the one
it gave with a copy of its own (checked row by row over the dev2000 panel at both tiers).
"""
import pytest

from orthonym.assembly import memo
from orthonym.assembly.fragment_naming import _fragment_guard
from orthonym.rules import pin_vocabulary as pv

pytestmark = pytest.mark.unit


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.setattr(memo, "_MODE", "on")
    token = memo.push_scope()
    try:
        yield
    finally:
        memo.pop_scope(token)


@pytest.fixture
def clean_guard():
    """The fragment guard is thread-local state: put back exactly what was there (an absent attribute stays absent)."""
    absent = object()
    saved = {a: getattr(_fragment_guard, a, absent) for a in ("visited", "session_depth")}
    try:
        yield _fragment_guard
    finally:
        for attr, value in saved.items():
            if value is absent:
                if hasattr(_fragment_guard, attr):
                    delattr(_fragment_guard, attr)
            else:
                setattr(_fragment_guard, attr, value)


def _producer(computed, result=None):
    """A promoted producer that names one fragment through the memo, as the ring-substituent producers do."""
    def produce():
        memo.cache_or_compute("t_preview_shared", "fragment", lambda: computed.append(1) or "named")
        return result
    return produce


def _record(produce):
    assert pv.promote_at_pin_tier(produce) is None           # outside the re-run: no promotion, one recorded call


def test_calls_of_one_family_name_their_fragment_once(scope):
    computed = []
    produce = _producer(computed)
    with pv.promotion_site_probe() as probe:
        for _ in range(5):
            _record(produce)
    assert len(probe.sites) == 5
    assert pv.promotion_could_change(probe) is False
    assert len(computed) == 1


def test_a_preview_without_sharing_names_it_for_every_call(scope):
    computed = []
    produce = _producer(computed)
    with pv.promotion_site_probe() as probe:
        for _ in range(5):
            _record(produce)
    assert [pv._preview_site(site) for site in probe.sites] == [False] * 5
    assert len(computed) == 5


def test_a_kept_name_is_still_found_among_repeated_calls(scope):
    computed = []
    with pv.promotion_site_probe() as probe:
        _record(_producer(computed))
        _record(_producer(computed))
        _record(_producer(computed, result="phenyl"))          # the promoted producer keeps a PIN-vocabulary name
        _record(_producer(computed))
    assert pv.promotion_could_change(probe) is True


def test_calls_in_another_memo_scope_do_not_share(scope):
    computed = []
    produce = _producer(computed)
    with pv.promotion_site_probe() as probe:
        _record(produce)
        box = memo.fresh_scope()                 # an empty copy of the scope: its entries are not the enclosing scope's
        try:
            _record(produce)
        finally:
            memo.pop_sandbox(box)
    assert pv.promotion_could_change(probe) is False
    assert len(computed) == 2


def test_calls_with_another_visited_set_do_not_share(scope, clean_guard):
    computed = []
    produce = _producer(computed)
    with pv.promotion_site_probe() as probe:
        clean_guard.visited = {"CCO"}
        _record(produce)
        clean_guard.visited = {"CCO", "CCN"}
        _record(produce)
        _record(produce)
    assert pv.promotion_could_change(probe) is False
    assert len(computed) == 2                    # one per visited set; the third call joined the second's family


def test_the_preview_leaves_the_scope_of_the_caller_alone(scope):
    computed = []
    with pv.promotion_site_probe() as probe:
        _record(_producer(computed))
        _record(_producer(computed))
    held = dict(memo.current_scope())
    assert pv.promotion_could_change(probe) is False
    assert dict(memo.current_scope()) == held
