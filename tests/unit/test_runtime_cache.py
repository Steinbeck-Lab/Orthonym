"""Tests for the runtime dynamic fragment cache (a phase).

The runtime cache stores (canonical SMILES -> name) pairs during a single
naming session, avoiding redundant re-computation of the same fragment.

Session nesting is an explicit counter (``session_depth``, cdb8db132), not the visited
set's emptiness, and the fragment cache is owned by the name scope (``name_call_depth``,
60cb5f8f6 / be5449980); the tests that need a nested or open session set those counters.
"""

import threading

import pytest
from rdkit import Chem

from orthonym.assembly.fragment_naming import (
    _fragment_guard,
    _get_visited,
    start_naming_session,
    end_naming_session,
    get_naming_depth,
    name_fragment_recursively,
    FRAGMENT_NAME_CACHE,
)


_DEPTHS = ("session_depth", "name_call_depth")


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset thread-local state before and after each test.

    The two depth counters too (TRIAGE g8 C2, 2026-09-27): since the session
    became an explicit counter, ``test_start_creates_cache`` (a start with no
    end) left ``session_depth`` at 1, so ``test_end_clears_cache`` -- run after
    it in the same process -- saw a nested session and kept the cache. It
    passed alone and failed wherever xdist put the two on one worker.
    """
    _fragment_guard.visited = set()
    _fragment_guard.cache = None
    for attr in _DEPTHS:
        setattr(_fragment_guard, attr, 0)
    yield
    _fragment_guard.visited = set()
    _fragment_guard.cache = None
    for attr in _DEPTHS:
        setattr(_fragment_guard, attr, 0)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_start_creates_cache():
    """start_naming_session creates a cache dict when visited set is empty."""
    start_naming_session()
    assert isinstance(getattr(_fragment_guard, 'cache', None), dict)


def test_start_noop_when_nested():
    """start_naming_session allocates no cache inside an outer session (nested call).

    "Nested" is an explicit counter (``start_naming_session``: "'Outermost' is an
    EXPLICIT COUNTER, not `len(visited) == 0`"), so the outer session is set up as
    the counter, not only as a non-empty visited set. This test used to pass only
    on the ``session_depth`` the previous test leaked (TRIAGE g8 C2).
    """
    _get_visited().add("PARENT_SMILES")
    _fragment_guard.session_depth = 1  # an outer session is open
    _fragment_guard.cache = None
    start_naming_session()
    assert getattr(_fragment_guard, 'cache', None) is None


def test_end_clears_cache():
    """end_naming_session sets cache to None when visited set is empty."""
    _fragment_guard.cache = {"CCO": "ethanol"}
    end_naming_session()
    assert getattr(_fragment_guard, 'cache', None) is None


def test_end_noop_when_nested():
    """end_naming_session closes only the INNER session when sessions are nested: the outer
    one keeps its cache and visited set.

    "Nested" is the explicit ``session_depth`` counter (see ``start_naming_session``), not a
    non-empty visited set; this test used to pass only on the depth the previous test leaked
    (TRIAGE g8 C2, d093d2cb7: the autouse fixture now resets the counters)."""
    _get_visited().add("PARENT_SMILES")
    _fragment_guard.session_depth = 2  # an outer session and one nested session are open
    _fragment_guard.cache = {"CCO": "ethanol"}
    end_naming_session()
    assert getattr(_fragment_guard, 'cache', None) == {"CCO": "ethanol"}
    assert _fragment_guard.session_depth == 1
    assert "PARENT_SMILES" in _get_visited()


# ---------------------------------------------------------------------------
# Cache populates and serves
# ---------------------------------------------------------------------------

def test_cache_populates_on_success():
    """Successful naming stores result in runtime cache.

    The cache is owned by the name scope (60cb5f8f6, be5449980) and the session is an explicit
    counter; the autouse fixture resets both counters to 0, so the test
    opens the scope itself instead of relying on a depth leaked by an earlier test."""
    _fragment_guard.name_call_depth = 1   # inside a name scope
    _fragment_guard.session_depth = 1     # inside its naming session
    _fragment_guard.cache = {}

    # Name something not in the static cache
    result = name_fragment_recursively("CCCCCCCCCCCCC")  # tridecane
    assert result is not None

    # Verify it's in the runtime cache
    canonical = Chem.CanonSmiles("CCCCCCCCCCCCC")
    assert _fragment_guard.cache.get(canonical) == result


def test_cache_serves_on_repeat_call():
    """Second call for same SMILES returns cached result."""
    canonical = Chem.CanonSmiles("CCCCCCCCCCCCC")
    _fragment_guard.cache = {canonical: "tridecane"}

    # Should get cache hit
    result = name_fragment_recursively("CCCCCCCCCCCCC")
    assert result == "tridecane"


def test_cache_does_not_store_none():
    """Failed naming does NOT store None in the cache."""
    _fragment_guard.cache = {}

    # An invalid SMILES returns None — should not be cached
    result = name_fragment_recursively("INVALID_SMILES")
    assert result is None
    assert len(_fragment_guard.cache) == 0


def test_cache_bypasses_cycle_detection():
    """Cached results are returned even when SMILES is in visited set."""
    canonical = Chem.CanonSmiles("CCCCCCCCCCCCC")
    _get_visited().add(canonical)  # Simulate being in the naming stack
    _fragment_guard.cache = {canonical: "tridecane"}

    result = name_fragment_recursively("CCCCCCCCCCCCC")
    assert result == "tridecane"  # Cache hit, bypasses cycle detection


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_thread_safety_independent_caches():
    """Two threads have independent runtime caches."""
    results = {}

    def name_in_thread(smiles, thread_id):
        start_naming_session()
        try:
            result = name_fragment_recursively(smiles)
            cache = getattr(_fragment_guard, 'cache', None)
            results[thread_id] = {
                'result': result,
                'cache_size': len(cache) if cache else 0,
            }
        finally:
            end_naming_session()

    t1 = threading.Thread(target=name_in_thread, args=("CCO", "t1"))
    t2 = threading.Thread(target=name_in_thread, args=("CCC", "t2"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results["t1"]["result"] == "ethanol"
    assert results["t2"]["result"] == "propane"


# ---------------------------------------------------------------------------
# Integration: repeated fragments benefit from cache
# ---------------------------------------------------------------------------

def test_repeated_fragment_naming():
    """The same fragment named multiple times within a session uses cache."""
    from orthonym import name_compound

    # Ethyl docosanoate — the decomposition names both acid and alkyl fragments
    # with the cache active, repeated fragments should be cached
    result = name_compound("CCCCCCCCCCCCCCCCCCCCCC(=O)OCC")
    assert result == "ethyl docosanoate"


def test_session_cleanup_no_leak():
    """After end_naming_session, cache is None — no memory leak."""
    start_naming_session()
    name_fragment_recursively("CCO")
    end_naming_session()
    assert getattr(_fragment_guard, 'cache', None) is None
