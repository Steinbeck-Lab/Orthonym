"""Tests for the runtime dynamic fragment cache (a phase).

The runtime cache stores (canonical SMILES -> name) pairs during a single
naming session, avoiding redundant re-computation of the same fragment.

Updated for cycle-detection architecture: session nesting is determined
by visited-set emptiness, not a depth counter.
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


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset thread-local state before and after each test."""
    _fragment_guard.visited = set()
    _fragment_guard.cache = None
    yield
    _fragment_guard.visited = set()
    _fragment_guard.cache = None


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_start_creates_cache():
    """start_naming_session creates a cache dict when visited set is empty."""
    start_naming_session()
    assert isinstance(getattr(_fragment_guard, 'cache', None), dict)


def test_start_noop_when_nested():
    """start_naming_session is a no-op when visited set is non-empty (nested call)."""
    _get_visited().add("PARENT_SMILES")
    _fragment_guard.cache = None
    start_naming_session()
    assert getattr(_fragment_guard, 'cache', None) is None


def test_end_clears_cache():
    """end_naming_session sets cache to None when visited set is empty."""
    _fragment_guard.cache = {"CCO": "ethanol"}
    end_naming_session()
    assert getattr(_fragment_guard, 'cache', None) is None


def test_end_noop_when_nested():
    """end_naming_session is a no-op when visited set is non-empty."""
    _get_visited().add("PARENT_SMILES")
    _fragment_guard.cache = {"CCO": "ethanol"}
    end_naming_session()
    assert getattr(_fragment_guard, 'cache', None) == {"CCO": "ethanol"}


# ---------------------------------------------------------------------------
# Cache populates and serves
# ---------------------------------------------------------------------------

def test_cache_populates_on_success():
    """Successful naming stores result in runtime cache."""
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
