"""A leaked fragment-recursion guard must NOT disable the tier flags forever.

The defect, found while landing the enrichment-vocabulary change:

`namer.name()` publishes `general_fallback_ctx` and `best_effort_ctx` only inside
`if is_top_level_naming():`, and `is_top_level_naming()` is
`len(_fragment_guard.visited) == 0`. But `end_naming_session`
(`fragment_naming.py:232-241`) clears `visited` **only when it is already empty**:

    visited = getattr(_fragment_guard, 'visited', None)
    if visited is None or len(visited) == 0:
        _fragment_guard.cache = None
        _fragment_guard.visited = set()

So if an exception ever escapes a fragment naming without discarding its SMILES, the set
stays non-empty **for the rest of the thread**, `is_top_level_naming()` never returns True
again, and **the tier flags are never published again — best-effort silently reverts to the
PIN vocabulary, with no error anywhere.**

⚠ **`general_fallback_ctx` has shipped under that guard since v25**, so the exposure is five
milestones wide and is not specific to the flag added this session.

Not theoretical: it first appeared as a test that passed alone and failed inside a 17-file
run that took 54 s instead of 25 s under load — i.e. it needs a real interruption, which is
exactly what makes it rare and long-lived rather than harmless.

The emptiness guard itself is CORRECT and must stay: it stops a nested call from clearing
its parent's cycle-detection state. The fix is that the OUTERMOST caller — which already
knows it is outermost, because it holds a ctx token only in that case — resets
unconditionally instead of asking `visited` whether it may.
"""

import pytest


def _visited():
    from orthonym.assembly import fragment_naming as fn
    return getattr(fn._fragment_guard, "visited", None)


@pytest.fixture(autouse=True)
def _clean_guard():
    """Leave the thread's guard clean whichever way the test exits."""
    from orthonym.assembly import fragment_naming as fn
    fn._fragment_guard.visited = set()
    fn._fragment_guard.cache = None
    yield
    fn._fragment_guard.visited = set()
    fn._fragment_guard.cache = None


def test_the_premise_is_still_live():
    """Pin the SYMPTOM the fix has to defeat: a polluted visited set still makes
    `is_top_level_naming()` False, so the flag-publication guard is still the thing at
    risk. Deliberately does NOT assert on source text — the first draft of this test
    grepped `end_naming_session` for "len(visited) == 0", which the fix removes, so it
    would have inverted into a failure the moment the defect was repaired."""
    from orthonym.assembly import fragment_naming as fn

    assert fn.is_top_level_naming() is True
    fn._fragment_guard.visited = {"CCO"}
    assert fn.is_top_level_naming() is False, (
        "a non-empty visited set no longer makes is_top_level_naming False; the leak "
        "mechanism has changed and this file must be re-derived"
    )


def test_the_outermost_teardown_now_clears_unconditionally():
    """The fix, at the unit level: at depth 0 the teardown must clear a polluted set
    rather than asking it for permission."""
    from orthonym.assembly import fragment_naming as fn

    fn._fragment_guard.visited = {"CCO"}
    fn._fragment_guard.session_depth = 0
    fn.end_naming_session()
    assert fn._fragment_guard.visited == set(), (
        "a leaked visited set survived the outermost teardown"
    )
    assert fn.is_top_level_naming() is True


def test_a_top_level_naming_heals_a_leaked_guard():
    """THE regression. After a polluted guard, one top-level naming must leave the
    thread usable: `visited` empty and `is_top_level_naming()` True again.

    Before the fix this failed — the naming ran with the guard already non-empty, so it
    never took the top-level branch, never published a ctx token, and never reset.
    """
    from orthonym import Orthonym
    from orthonym.assembly import fragment_naming as fn

    fn._fragment_guard.visited = {"a-leaked-fragment-smiles"}
    assert fn.is_top_level_naming() is False, "precondition: the guard is polluted"

    Orthonym().name("CCO")

    assert fn.is_top_level_naming() is True, (
        "a leaked guard survived a top-level naming: every later best-effort naming on "
        "this thread would silently use the PIN vocabulary"
    )
    assert not _visited(), f"visited should be empty, got {_visited()!r}"


def test_the_tier_flag_reaches_its_consumer_after_a_leak():
    """The consequence that actually costs correctness: after a leak, a best-effort
    naming must still be given the best-effort substituent vocabulary.

    ⚠ Asserted through the CONSUMER, not by spying on `best_effort_ctx.set` — that is a
    C-level `ContextVar` method and is **read-only**, so the first draft's monkeypatch
    raised `AttributeError` and the test failed for its own reasons rather than the
    code's. Verified directly:
    `'_contextvars.ContextVar' object attribute 'set' is read-only`.
    """
    from orthonym.assembly import composer, fragment_naming as fn
    from orthonym.cli import _emit_tier_flags
    from orthonym import Orthonym

    seen = []
    orig = composer._integrate_universal_prefixes

    def spy(*a, allow_mancude=None, **kw):
        seen.append(allow_mancude)
        return orig(*a, allow_mancude=allow_mancude, **kw)

    fn._fragment_guard.visited = {"a-leaked-fragment-smiles"}
    composer._integrate_universal_prefixes = spy
    try:
        be = _emit_tier_flags("best-effort")
        for smi in ("CC(=O)Cl", "CCC(=O)Cl"):      # acid halides call the helper
            Orthonym(style="pin", **be).name(smi)
    finally:
        composer._integrate_universal_prefixes = orig

    assert seen, "the helper was never reached; pick a probe molecule that calls it"
    assert any(v is True or v is None for v in seen), (
        f"after a leaked guard the helper was handed {seen!r} -- an explicit False on "
        f"every call means the tier flag is dead on this thread and best-effort has "
        f"silently reverted to the PIN vocabulary"
    )


def test_nested_naming_still_shares_the_parent_session():
    """The guard's PURPOSE must survive the fix: a nested call must not clear its
    parent's cycle-detection state, or cycle detection breaks.

    Exercised through the DEPTH COUNTER, because that is now what "nested" means. The
    first draft simulated nesting by pre-polluting `visited`, which conflated the
    symptom of the bug with the feature — and would have made the fix look like a
    regression.
    """
    from orthonym.assembly import fragment_naming as fn

    fn.start_naming_session()                     # outermost
    fn._fragment_guard.visited.add("parent-smiles")
    fn.start_naming_session()                     # nested
    assert fn._fragment_guard.visited == {"parent-smiles"}, (
        "a nested start wiped the live parent session"
    )
    fn.end_naming_session()                       # nested returns
    assert fn._fragment_guard.visited == {"parent-smiles"}, (
        "a nested end wiped the live parent session"
    )
    fn.end_naming_session()                       # outermost returns
    assert fn._fragment_guard.visited == set(), (
        "the outermost end failed to clear the session"
    )
