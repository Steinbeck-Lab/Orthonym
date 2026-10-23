"""``append_missing`` is the merge ``restore_provenance`` and a nested-memo hit make of an
accumulating provenance record: the base, then what the base lacks, in order. It must give
what the scanning form ``base + tuple(f for f in extra if f not in base)`` gave."""
import random

import pytest

from orthonym.assembly import memo
from orthonym.assembly import nested_memo
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.unit


def _reference(base, extra):
    return base + tuple(f for f in extra if f not in base)


def test_append_missing_keeps_order_and_drops_what_base_holds():
    assert pv.append_missing(("a", "b"), ("c", "a", "d")) == ("a", "b", "c", "d")
    assert pv.append_missing((), ("x", "y")) == ("x", "y")
    assert pv.append_missing(("x",), ()) == ("x",)


def test_append_missing_keeps_repeats_inside_extra_that_base_lacks():
    # the scanning form tested each item against the base only, so a repeat in ``extra`` stays
    assert pv.append_missing(("a",), ("b", "b", "a")) == ("a", "b", "b")


def test_append_missing_equals_the_scanning_form_on_random_records():
    rng = random.Random(20261023)
    pool = [f"frag-{i}" for i in range(40)]
    for _ in range(200):
        base = tuple(rng.choice(pool) for _ in range(rng.randint(0, 25)))
        extra = tuple(rng.choice(pool) for _ in range(rng.randint(0, 25)))
        assert pv.append_missing(base, extra) == _reference(base, extra)


def test_append_missing_with_an_unhashable_item_falls_back_to_the_scan():
    base = (["a"], "b")
    extra = ("b", ["a"], "c")
    assert pv.append_missing(base, extra) == _reference(base, extra) == (["a"], "b", "c")


def test_append_missing_with_an_unhashable_item_only_in_extra_falls_back_to_the_scan():
    base = ("a",)
    extra = (["x"], "a", "b")
    assert pv.append_missing(base, extra) == _reference(base, extra) == ("a", ["x"], "b")


def test_restore_provenance_merges_the_snapshot_record_first():
    pv.clear_provenance()
    try:
        pv.record_non_pin_fragment("alpha")
        pv.record_non_pin_fragment("beta")
        snap = pv.get_provenance()
        pv.record_non_pin_fragment("gamma")
        pv.record_non_pin_fragment("alpha")      # already held: not recorded twice
        live = pv.get_provenance()["non_pin_fragments"]
        pv.restore_provenance(snap)
        assert pv.get_provenance()["non_pin_fragments"] == _reference(
            snap["non_pin_fragments"], live) == ("alpha", "beta", "gamma")
    finally:
        pv.clear_provenance()


def test_nested_memo_hit_merges_the_recorded_fragments_after_the_current_ones():
    tok = memo.push_scope()
    pv.clear_provenance()
    try:
        def fresh():
            pv.record_non_pin_fragment("one")
            pv.record_non_pin_fragment("two")
            return "name"

        assert nested_memo.cached_nested_call("t-append", ("k",), fresh) == "name"
        pv.clear_provenance()
        pv.record_non_pin_fragment("two")          # held already when the hit merges
        pv.record_non_pin_fragment("zero")
        assert nested_memo.cached_nested_call("t-append", ("k",), lambda: "never") == "name"
        assert pv.get_provenance()["non_pin_fragments"] == ("two", "zero", "one")
    finally:
        pv.clear_provenance()
        memo.pop_scope(tok)
