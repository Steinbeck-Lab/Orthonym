"""The record-call log of a memoised substituent-fragment naming: typed items, compaction, replay.

``record_derived_non_pin_fragment(source, derived)`` reads the record of non-PIN fragments only to decide
whether it records ``derived``; nothing it returns depends on it. It logs the call
(``('D', source, derived)``) in place of a provenance read, and a hit makes the call again. The compacted
log (``compact_non_pin_log``) is an equivalent tuple with the no-op repeats removed: its replay leaves every
record as the replay of the raw log leaves it, which the brute-force test below checks over every starting record
of a small alphabet, derived-only forms included.
"""
import itertools
import random

import pytest

from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean():
    pv.clear_provenance()
    yield
    pv.clear_provenance()


def D(source, derived):
    return ('D', source, derived)


def _replayed(items, start=()):
    """The record that the replay of ``items`` leaves, from the record ``start``."""
    pv.clear_provenance()
    for fragment in start:
        pv.record_non_pin_fragment(fragment)
    pv.replay_non_pin_log(items)
    return pv.get_provenance()["non_pin_fragments"]


# ---- the typed log item -------------------------------------------------------------------------

def test_a_derived_call_logs_itself_and_not_a_provenance_read():
    log = pv.push_non_pin_log()
    try:
        pv.record_derived_non_pin_fragment("acid x", "x-yl")
    finally:
        pv.pop_non_pin_log()
    assert log == [D("acid x", "x-yl")]
    assert not any(item is pv.PROVENANCE_READ for item in log)


def test_a_firing_derived_call_records_the_derived_form_and_logs_only_the_call():
    pv.record_non_pin_fragment("acid")
    log = pv.push_non_pin_log()
    try:
        pv.record_derived_non_pin_fragment("acid x", "x-yl")
    finally:
        pv.pop_non_pin_log()
    assert pv.get_provenance()["non_pin_fragments"] == ("acid", "x-yl")
    # no plain item for 'x-yl': the replay decides again on the record it finds
    assert log == [D("acid x", "x-yl")]


def test_a_derived_call_that_finds_no_fragment_records_nothing():
    pv.record_non_pin_fragment("other")
    pv.record_derived_non_pin_fragment("acid x", "x-yl")
    assert pv.get_provenance()["non_pin_fragments"] == ("other",)


def test_a_derived_call_without_a_source_or_a_form_is_not_logged():
    log = pv.push_non_pin_log()
    try:
        pv.record_derived_non_pin_fragment("", "x-yl")
        pv.record_derived_non_pin_fragment("acid x", "")
    finally:
        pv.pop_non_pin_log()
    assert log == []


def test_every_other_read_still_logs_a_provenance_read():
    for read in (pv.get_provenance, lambda: pv.name_carries_non_pin_part({}, "x"),
                 lambda: pv.non_pin_label_rule_failures({}, "x"),
                 lambda: pv.verified_field("opsin", {}, "x"),
                 lambda: pv.record_derived_uncertified_pin_name("a", "b")):
        log = pv.push_non_pin_log()
        try:
            read()
        finally:
            pv.pop_non_pin_log()
        assert any(item is pv.PROVENANCE_READ for item in log), read


def test_the_isolated_body_drops_a_derived_call_from_the_log_and_the_record():
    pv.record_non_pin_fragment("acid")
    log = pv.push_non_pin_log()
    try:
        with pv.isolated_provenance():
            pv.record_derived_non_pin_fragment("acid x", "x-yl")
            assert pv.get_provenance()["non_pin_fragments"] == ("acid", "x-yl")
    finally:
        pv.pop_non_pin_log()
    assert pv.get_provenance()["non_pin_fragments"] == ("acid",)
    # the call is gone from the log; the read of the record in the body (get_provenance) stays
    assert log == [pv.PROVENANCE_READ]


# ---- the compaction rule -------------------------------------------------------------------------

def test_a_repeated_plain_fragment_is_dropped():
    assert pv.compact_non_pin_log(["a", "b", "a", "b", "a"]) == ("a", "b")


def test_the_order_of_first_record_is_kept():
    assert pv.compact_non_pin_log(["c", "a", "c", "b"]) == ("c", "a", "b")


def test_a_repeat_of_a_derived_call_with_nothing_between_is_dropped():
    assert pv.compact_non_pin_log([D("abc", "x"), D("abc", "x")]) == (D("abc", "x"),)


def test_the_reviewers_counterexample_keeps_the_repeat():
    # D(s, d), record f, D(s, d) with f a substring of s: the first call finds nothing in an empty record,
    # the repeat finds f and records d. A first-occurrence dedup drops the repeat and loses d.
    log = [D("abc", "x"), "b", D("abc", "x")]
    assert pv.compact_non_pin_log(log) == tuple(log)
    assert _replayed(log) == ("b", "x")
    naive = tuple(dict.fromkeys(log))
    assert _replayed(naive) == ("b",)          # what the naive dedup would leave
    assert _replayed(pv.compact_non_pin_log(log)) == _replayed(log)


def test_a_plain_item_between_that_is_not_a_substring_of_the_source_does_not_keep_the_repeat():
    assert pv.compact_non_pin_log([D("abc", "x"), "q", D("abc", "x")]) == (D("abc", "x"), "q")


def test_a_kept_derived_call_between_whose_form_is_a_substring_keeps_the_repeat():
    log = [D("abc", "x"), D("zzz", "b"), D("abc", "x")]
    assert pv.compact_non_pin_log(log) == tuple(log)
    # started from a record that fires the middle call, the repeat records x
    assert _replayed(log, ("z",)) == ("z", "b", "x")
    assert _replayed(log) == ()


def test_a_kept_derived_call_between_whose_form_is_not_a_substring_does_not_keep_the_repeat():
    assert pv.compact_non_pin_log([D("abc", "x"), D("zzz", "q"), D("abc", "x")]) == (
        D("abc", "x"), D("zzz", "q"))


def test_a_plain_item_between_that_was_recorded_before_the_first_call_does_not_keep_the_repeat():
    assert pv.compact_non_pin_log(["b", D("abc", "x"), "b", D("abc", "x")]) == ("b", D("abc", "x"))


def test_the_repeat_is_judged_against_its_last_kept_occurrence():
    # the second call is kept (b was recorded in between); a third with nothing since the second is dropped
    log = [D("abc", "x"), "b", D("abc", "x"), D("abc", "x")]
    assert pv.compact_non_pin_log(log) == (D("abc", "x"), "b", D("abc", "x"))
    # an item that adds a substring of the source after the second call keeps the third
    log = [D("abc", "x"), "b", D("abc", "x"), "c", D("abc", "x")]
    assert pv.compact_non_pin_log(log) == tuple(log)


def test_a_derived_call_whose_form_was_recorded_as_a_plain_fragment_is_dropped():
    assert pv.compact_non_pin_log(["x", D("abc", "x")]) == ("x",)
    # a plain record after the call does not drop it: the call may be what the middle item reads
    assert pv.compact_non_pin_log([D("abc", "x"), "x"]) == (D("abc", "x"), "x")


def test_a_log_with_a_provenance_read_is_not_stored():
    assert pv.compact_non_pin_log(["a", pv.PROVENANCE_READ]) is None
    assert pv.compact_non_pin_log([D("abc", "x"), pv.PROVENANCE_READ, "a"]) is None


def test_the_empty_log_compacts_to_the_empty_tuple():
    assert pv.compact_non_pin_log([]) == ()


# ---- the cap -------------------------------------------------------------------------------------

def test_a_compacted_log_with_a_derived_call_over_the_cap_is_not_stored():
    log = [D("source %d" % k, "form %d" % k) for k in range(5)]
    assert pv.compact_non_pin_log(log, cap=5) == tuple(log)
    assert pv.compact_non_pin_log(log, cap=4) is None
    # the derived call counts when it comes after the plain items too
    assert pv.compact_non_pin_log(["f%d" % k for k in range(4)] + [D("source", "form")], cap=4) is None


def test_a_log_of_plain_items_is_stored_whatever_its_length():
    log = ["f%d" % k for k in range(1000)]
    assert pv.compact_non_pin_log(log, cap=3) == tuple(log)
    # a derived call that adds nothing (its form is recorded already) is not in the stored log
    assert pv.compact_non_pin_log(log + [D("source", "f1")], cap=3) == tuple(log)


def test_the_cap_counts_the_compacted_log_not_the_raw_one():
    log = ["a", "b"] * 100 + [D("ab", "x")] * 100
    assert pv.compact_non_pin_log(log, cap=3) == ("a", "b", D("ab", "x"))


def test_the_default_cap_is_the_module_constant(monkeypatch):
    log = [D("source %d" % k, "form %d" % k) for k in range(4)]
    monkeypatch.setattr(pv, "NON_PIN_LOG_CAP", 3)
    assert pv.compact_non_pin_log(log) is None
    monkeypatch.setattr(pv, "NON_PIN_LOG_CAP", 4)
    assert pv.compact_non_pin_log(log) == tuple(log)


# ---- replay --------------------------------------------------------------------------------------

def test_the_replay_runs_the_items_in_order(monkeypatch):
    log = ["a", D("ab", "x"), "b", D("xb", "y")]
    assert _replayed(log) == ("a", "x", "b", "y")
    seen = []
    real_plain, real_derived = pv.record_non_pin_fragment, pv.record_derived_non_pin_fragment
    monkeypatch.setattr(pv, "record_non_pin_fragment", lambda f: (seen.append(("plain", f)), real_plain(f))[1])
    monkeypatch.setattr(pv, "record_derived_non_pin_fragment",
                        lambda s, d, item=None: (seen.append(("derived", s, d)), real_derived(s, d, item))[1])
    pv.replay_non_pin_log(log)
    assert seen == [("plain", "a"), ("derived", "ab", "x"), ("plain", "b"), ("derived", "xb", "y")]


def test_a_derived_call_decides_again_on_the_record_of_the_replay():
    log = [D("acid x", "x-yl")]
    assert _replayed(log, ("acid",)) == ("acid", "x-yl")
    assert _replayed(log, ("other",)) == ("other",)
    assert _replayed(log) == ()


def test_the_replay_logs_into_the_enclosing_logs_as_a_fresh_call_does():
    pv.record_non_pin_fragment("acid")
    log = pv.push_non_pin_log()
    try:
        pv.replay_non_pin_log(("p", D("acid x", "x-yl")))
    finally:
        pv.pop_non_pin_log()
    assert log == ["p", D("acid x", "x-yl")]


def test_a_nested_replay_compacts_to_the_replay_of_its_parts():
    inner = pv.compact_non_pin_log(["acid", D("acid x", "x-yl"), "acid"])
    outer_log = pv.push_non_pin_log()
    try:
        for _ in range(3):
            pv.replay_non_pin_log(inner)
            pv.record_non_pin_fragment("sibling")
    finally:
        pv.pop_non_pin_log()
    assert len(outer_log) == 3 * (len(inner) + 1)
    assert pv.compact_non_pin_log(outer_log) == ("acid", D("acid x", "x-yl"), "sibling")


# ---- exactness, checked by brute force ------------------------------------------------------------

_FRAGMENTS = ("a", "b", "c", "ab", "bc", "d")
_SOURCES = ("ab", "abc", "bcd", "zz", "d", "abcd")
_DERIVED = ("a", "b", "bc", "x", "y", "zz")


def _items(rng, n):
    out = []
    for _ in range(n):
        if rng.random() < 0.45:
            out.append(rng.choice(_FRAGMENTS + _DERIVED))
        else:
            out.append(D(rng.choice(_SOURCES), rng.choice(_DERIVED)))
    return out


def _starting_records():
    # every record of plain fragments, and every record of up to three items that also holds derived-only forms
    for size in range(len(_FRAGMENTS) + 1):
        yield from itertools.combinations(_FRAGMENTS, size)
    pool = _FRAGMENTS + tuple(d for d in _DERIVED if d not in _FRAGMENTS)
    for size in range(1, 4):
        for subset in itertools.combinations(pool, size):
            if not set(subset) <= set(_FRAGMENTS):
                yield subset


def test_the_compacted_log_leaves_every_record_the_raw_log_leaves():
    rng = random.Random(20261023)
    starts = list(_starting_records())
    shortened = 0
    for _ in range(600):
        raw = _items(rng, rng.randint(1, 10))
        compacted = pv.compact_non_pin_log(raw, cap=10 ** 6)
        assert compacted is not None
        shortened += len(compacted) < len(raw)
        assert len(compacted) <= len(raw)
        for start in starts:
            assert set(_replayed(compacted, start)) == set(_replayed(raw, start)), (raw, compacted, start)
    assert shortened > 100           # the property is not checked on logs that never compact


def test_the_compacted_log_leaves_the_record_in_the_order_the_raw_log_does():
    rng = random.Random(7)
    starts = list(_starting_records())[::7]
    for _ in range(300):
        raw = _items(rng, rng.randint(1, 10))
        compacted = pv.compact_non_pin_log(raw, cap=10 ** 6)
        for start in starts:
            assert _replayed(compacted, start) == _replayed(raw, start), (raw, compacted, start)


def test_a_replay_logs_the_stored_item_itself():
    item = D("abc", "x")
    pv.record_non_pin_fragment("a")
    log = pv.push_non_pin_log()
    try:
        pv.replay_non_pin_log((item,))
    finally:
        pv.pop_non_pin_log()
    assert log == [item] and log[0] is item
    assert "x" in pv.get_provenance()["non_pin_fragments"]
