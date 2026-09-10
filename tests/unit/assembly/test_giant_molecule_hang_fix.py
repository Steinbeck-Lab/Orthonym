""" giant-molecule hang fix — regression locks.

The best-effort namer used to HANG on >60-heavy-atom molecules (acyl-CoA,
peptide-glycan bioconjugates): its recursive per-fragment naming re-explored the
SAME fragment dozens of times because (a) the runtime fragment cache was RESET
across ``isolated_naming_session`` and (b) a previous work-budget attempt was
re-initialised by that same isolation. Each re-exploration paid an OPSIN
subprocess (up to 30 s).

These tests lock the three coordinated fixes at the unit level (no JVM):
  A. the fragment cache PERSISTS across ``isolated_naming_session``;
  B. the per-top-level work budget SURVIVES ``isolated_naming_session`` and is
     armed once at the true outermost ``name`` scope;
  C. OPSIN parse results are memoized (successes AND failures).
"""
import pytest

from orthonym.assembly import fragment_naming as fn


@pytest.fixture(autouse=True)
def _clean_guard():
    """Each test starts and ends with a pristine fragment thread-local."""
    g = fn._fragment_guard
    for attr in ('session_depth', 'cache', 'visited', 'name_call_depth', 'work_budget'):
        if hasattr(g, attr):
            delattr(g, attr)
    yield
    for attr in ('session_depth', 'cache', 'visited', 'name_call_depth', 'work_budget'):
        if hasattr(g, attr):
            delattr(g, attr)


# ---------------------------------------------------------------- B: work budget

def test_work_budget_armed_once_at_outermost_scope():
    fn.enter_name_scope()
    assert fn._fragment_guard.work_budget == fn._WORK_BUDGET
    assert fn._fragment_guard.name_call_depth == 1
    fn.exit_name_scope()
    assert getattr(fn._fragment_guard, 'work_budget', None) is None
    assert fn._fragment_guard.name_call_depth == 0


def test_nested_name_scopes_share_one_budget():
    fn.enter_name_scope()          # true outermost arms
    b0 = fn._fragment_guard.work_budget
    fn.enter_name_scope()          # nested re-entry does NOT re-arm
    assert fn._fragment_guard.work_budget == b0
    assert fn.spend_fragment_work() is True
    fn.exit_name_scope()           # nested exit keeps the budget
    assert fn._fragment_guard.work_budget == b0 - 1
    assert fn._fragment_guard.name_call_depth == 1
    fn.exit_name_scope()           # outermost exit disarms
    assert getattr(fn._fragment_guard, 'work_budget', None) is None


def test_work_budget_survives_isolated_naming_session():
    """THE core regression: isolation must NOT reset the budget (the previous
    attempt failed exactly here)."""
    fn.enter_name_scope()
    try:
        assert fn.spend_fragment_work() is True
        remaining = fn._fragment_guard.work_budget
        with fn.isolated_naming_session():
            # isolation reset session state but must leave the budget intact
            assert fn._fragment_guard.work_budget == remaining
            assert fn.spend_fragment_work() is True
        # the spend inside isolation persisted (budget was never restored/reset)
        assert fn._fragment_guard.work_budget == remaining - 1
    finally:
        fn.exit_name_scope()


def test_spend_returns_false_when_exhausted():
    fn.enter_name_scope()
    try:
        fn._fragment_guard.work_budget = 1
        assert fn.spend_fragment_work() is True    # spends the last unit -> 0
        assert fn._fragment_guard.work_budget == 0
        assert fn.spend_fragment_work() is False   # exhausted -> abstain
        assert fn.spend_fragment_work() is False   # stays exhausted
    finally:
        fn.exit_name_scope()


def test_spend_is_noop_without_a_scope():
    # A direct producer call outside any name scope must behave exactly as
    # before: no budget armed -> spend always allows.
    assert getattr(fn._fragment_guard, 'work_budget', None) is None
    assert fn.spend_fragment_work() is True
    assert fn.spend_fragment_work() is True


# ------------------------------------------------------------- A: cache persist

def test_fragment_cache_persists_across_isolation():
    fn.start_naming_session()
    try:
        cache = fn._fragment_guard.cache
        assert cache is not None
        cache['CCO'] = 'ethanol-marker'
        with fn.isolated_naming_session():
            # same live cache object, NOT a fresh empty one
            assert fn._fragment_guard.cache is cache
            assert fn._fragment_guard.cache.get('CCO') == 'ethanol-marker'
            fn._fragment_guard.cache['CCC'] = 'propane-marker'
        # an entry added inside isolation survives afterwards (memo is not thrown away)
        assert cache.get('CCC') == 'propane-marker'
    finally:
        fn.end_naming_session()


def test_isolation_still_resets_depth_budget():
    """Isolation must still give a fresh depth budget (its reason to exist):
    session_depth and the visited set reset, only the cache is kept."""
    fn.start_naming_session()
    try:
        fn._get_visited().add('SOMEFRAG')
        assert fn.get_naming_depth() >= 1
        with fn.isolated_naming_session():
            assert fn.get_naming_depth() == 0          # fresh depth budget
            assert fn._session_depth() == 0
        assert 'SOMEFRAG' in fn._get_visited()          # restored after
    finally:
        fn.end_naming_session()


def test_genuine_top_level_starts_a_fresh_cache():
    fn.start_naming_session()
    fn._fragment_guard.cache['X'] = 'y'
    fn.end_naming_session()                              # outermost clears cache
    assert getattr(fn._fragment_guard, 'cache', None) is None
    fn.start_naming_session()
    try:
        assert fn._fragment_guard.cache == {}           # fresh, not the old one
    finally:
        fn.end_naming_session()


# ------------------------------------------------------------------ C: OPSIN memo

def test_opsin_parse_is_memoized(monkeypatch):
    from orthonym.validation import atom_coverage as ac
    ac._OPSIN_PARSE_CACHE.clear()
    calls = {'n': 0}

    def fake_uncached(name, jar):
        calls['n'] += 1
        return 'C' if name == 'methane' else None

    monkeypatch.setattr(ac, '_parse_name_with_opsin_uncached', fake_uncached)
    try:
        assert ac._parse_name_with_opsin('methane', '/fake.jar') == 'C'
        assert ac._parse_name_with_opsin('methane', '/fake.jar') == 'C'
        assert calls['n'] == 1                          # 2nd served from cache
        # failures (None) are cached too, so a pathological name is parsed once
        assert ac._parse_name_with_opsin('bogus-name', '/fake.jar') is None
        assert ac._parse_name_with_opsin('bogus-name', '/fake.jar') is None
        assert calls['n'] == 2
    finally:
        ac._OPSIN_PARSE_CACHE.clear()


def test_opsin_memo_rejects_empty_inputs():
    from orthonym.validation import atom_coverage as ac
    assert ac._parse_name_with_opsin('', '/fake.jar') is None
    assert ac._parse_name_with_opsin('methane', '') is None
