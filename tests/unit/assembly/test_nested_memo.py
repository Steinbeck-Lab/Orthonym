"""R3: a memo hit of a nested naming call must replay (1) the provenance ContextVars the
call changed and (2) the perf/analysis/work budget units it charged, so the OUTER
molecule's tier and abstention decisions are byte-identical to a fresh run."""
import pytest
from orthonym.assembly import memo, fragment_naming as fn
from orthonym.assembly.nested_memo import cached_nested_call
from orthonym.metrics import provenance as pv


@pytest.fixture
def scope():
    tok = memo.push_scope(); pv.clear_provenance()
    fn._fragment_guard.perf_budget = 10_000; fn._fragment_guard.analysis_budget = 100; fn._fragment_guard.work_budget = 100
    yield
    fn._fragment_guard.perf_budget = None; fn._fragment_guard.analysis_budget = None; fn._fragment_guard.work_budget = None
    pv.clear_provenance(); memo.pop_scope(tok)


def _work():
    pv.record_source("general_engine", "verified"); pv.record_general_ring_prefix()
    fn.spend_perf_work(250); fn.spend_analysis_call(3)
    return "some acid"


def test_hit_replays_changed_provenance_and_budgets(scope):
    a = cached_nested_call("t", ("k",), _work)
    prov_after_fresh = pv.get_provenance(); b1 = (fn._fragment_guard.perf_budget, fn._fragment_guard.analysis_budget)
    pv.clear_provenance(); fn._fragment_guard.perf_budget = 10_000; fn._fragment_guard.analysis_budget = 100
    calls = []
    b = cached_nested_call("t", ("k",), lambda: calls.append(1) or "never")
    assert a == b == "some acid" and calls == []
    assert pv.get_provenance() == prov_after_fresh
    assert (fn._fragment_guard.perf_budget, fn._fragment_guard.analysis_budget) == b1


def test_hit_leaves_unchanged_vars_alone(scope):
    pv.record_source("pin_path")                       # pre-existing state the nested call did not touch
    cached_nested_call("t2", ("k",), lambda: (fn.spend_perf_work(1), "x")[1])
    pv.record_source("other")                          # outer state moves on
    cached_nested_call("t2", ("k",), lambda: "never")  # hit: must NOT reset source to "pin_path"
    assert pv.get_provenance()["source"] == "other"


def test_hit_replays_touched_var_even_when_net_unchanged(scope):
    """a performance pass (exact replay): a nested call that writes a var A->B->A
    (net unchanged) still TOUCHED it, so a later hit in a different ambient context
    must replay it to the value the fresh call left — a before/after delta would
    drop it and leave the stale ambient value."""
    pv.record_source("original")                       # ambient before the nested call

    def work():
        pv.record_source("x")
        pv.record_source("original")                   # A->B->A: net unchanged, but touched
        return "acid"

    a = cached_nested_call("t_exact", ("k",), work)
    pv.record_source("other")                          # ambient moves on
    pv.record_gate_outcome("SOMEGATE", "somename")     # a var the nested call did NOT touch

    def _boom():
        raise AssertionError("fn ran on a hit")

    b = cached_nested_call("t_exact", ("k",), _boom)   # hit: fn must not run
    assert a == b == "acid"
    # EXACT: the touched var is replayed to the nested call's end value, not left "other".
    assert pv.get_provenance()["source"] == "original"
    # A var the nested call never touched keeps the outer's value.
    assert pv.get_provenance()["gate_outcome"] == "SOMEGATE"


def test_exception_propagates_uncached(scope):
    """A nested fn that raises must propagate the exception uncached (nothing
    stored), so a later call re-runs rather than replaying a phantom entry."""
    calls = []

    class Boom(Exception):
        pass

    def _raiser():
        calls.append(1)
        raise Boom()

    with pytest.raises(Boom):
        cached_nested_call("t_exc", ("k",), _raiser)
    # Not cached: a second call re-runs fn (and raises again).
    with pytest.raises(Boom):
        cached_nested_call("t_exc", ("k",), _raiser)
    assert calls == [1, 1]
    # The touched-log stack is balanced after the raises (no leak): a normal call works.
    ok = cached_nested_call("t_exc2", ("k",), lambda: "fine")
    assert ok == "fine"


def test_none_results_are_cached(scope):
    calls = []
    cached_nested_call("t3", ("k",), lambda: calls.append(1))   # returns None
    cached_nested_call("t3", ("k",), lambda: calls.append(1))
    assert calls == [1]


def test_no_scope_means_no_cache():
    memo._cache_var.set(None)
    calls = []
    cached_nested_call("t4", ("k",), lambda: calls.append(1) or "v")
    cached_nested_call("t4", ("k",), lambda: calls.append(1) or "v")
    assert calls == [1, 1]


def test_nested_name_compound_is_memoised_only_when_nested(monkeypatch):
    """a lever: a NESTED name_compound(plain string) is served from the replay-memo;
    a TOP-LEVEL name_compound is NEVER memoised. Counting Orthonym.name calls for
    the exact input SMILES tells whether the engine re-ran."""
    import orthonym.namer as nm
    from orthonym.assembly import fragment_naming as fnm
    n = []
    real = nm.Orthonym.name
    monkeypatch.setattr(nm.Orthonym, "name",
                        lambda self, s, **k: n.append(s) or real(self, s, **k))

    # Top level: every call re-runs the engine (never memoised).
    nm.name_compound("CC(=O)O"); nm.name_compound("CC(=O)O")
    assert n.count("CC(=O)O") == 2

    n.clear()
    # Simulate "inside an outer name": is_top_level_naming is len(visited)==0,
    # and a real outer name adds the parent fragment's SMILES to visited before
    # recursing (fragment_naming.py:723). start_naming_session resets visited, so
    # add the marker AFTER it; open a memo scope for the nested memo to cache in.
    fnm.start_naming_session()
    fnm._get_visited().add("__outer_marker__")
    tok = memo.push_scope()
    try:
        assert not fnm.is_top_level_naming()
        nm.name_compound("CC(=O)O"); nm.name_compound("CC(=O)O")
        assert n.count("CC(=O)O") == 1
    finally:
        memo.pop_scope(tok); fnm.end_naming_session()
