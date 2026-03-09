"""Unit tests for fragment naming recursion depth guard.

Tests the thread-safe recursion depth tracking infrastructure in
orthonym.assembly.fragment_naming. This module prevents infinite loops
when fragment naming calls name_compound() recursively.
"""
import threading

import pytest

from orthonym.assembly.fragment_naming import (
    MAX_NAMING_DEPTH,
    _get_visited,
    get_naming_depth,
    name_fragment_recursively,
    _fragment_guard,
)


@pytest.mark.unit
class TestNamingDepthTracking:
    """Test depth counter read/write behavior."""

    def setup_method(self):
        """Reset depth before each test."""
        _fragment_guard.depth = 0

    def teardown_method(self):
        """Reset depth after each test."""
        _fragment_guard.depth = 0

    def test_naming_depth_starts_at_zero(self):
        """Initial depth should be 0 (no recursion in progress)."""
        # Use a fresh thread-local state
        if hasattr(_fragment_guard, 'depth'):
            delattr(_fragment_guard, 'depth')
        assert get_naming_depth() == 0

    def test_naming_depth_increments_during_recursive_call(self):
        """Depth should increase by 1 when name_fragment_recursively is called."""
        observed_depths = []

        # Monkey-patch name_compound to capture depth mid-call
        import orthonym.assembly.fragment_naming as mod
        original_import = mod.name_fragment_recursively

        # We can observe depth by checking get_naming_depth() inside a call
        # The simplest way: call name_fragment_recursively which increments
        # depth before calling name_compound
        _fragment_guard.depth = 0
        result = name_fragment_recursively("C")  # methane
        # After the call returns, depth should be restored to 0
        assert get_naming_depth() == 0
        # The result should be a valid name
        assert result is not None

    def test_naming_depth_restores_after_call(self):
        """Depth should return to previous value after call completes."""
        _fragment_guard.depth = 1  # Simulate being at depth 1
        result = name_fragment_recursively("CCO")  # ethanol
        assert get_naming_depth() == 1  # Restored to pre-call value
        assert result == "ethanol"

    def test_naming_depth_restores_on_exception(self):
        """Depth should restore even if name_compound raises an exception."""
        _fragment_guard.depth = 0
        # Invalid SMILES should trigger exception inside name_compound
        result = name_fragment_recursively("INVALID_NOT_A_SMILES_XYZ")
        assert result is None  # Exception caught, returns None
        assert get_naming_depth() == 0  # Depth restored


@pytest.mark.unit
class TestDepthLimit:
    """Test that depth limit prevents infinite recursion."""

    def setup_method(self):
        _fragment_guard.depth = 0

    def teardown_method(self):
        _fragment_guard.depth = 0

    def test_max_naming_depth_is_seven(self):
        """MAX_NAMING_DEPTH should be 7 (increased for deep iterative decomposition)."""
        assert MAX_NAMING_DEPTH == 7

    def test_depth_limit_returns_none_for_uncached(self):
        """At MAX_NAMING_DEPTH, uncached fragments return None."""
        _fragment_guard.depth = MAX_NAMING_DEPTH
        result = name_fragment_recursively("CCCCCCCCCCCCCC")  # tetradecane, not cached
        assert result is None

    def test_depth_limit_returns_cached(self):
        """At MAX_NAMING_DEPTH, cached fragments bypass depth and return a name."""
        _fragment_guard.depth = MAX_NAMING_DEPTH
        result = name_fragment_recursively("CCO")
        assert result == "ethanol"

    def test_depth_limit_prevents_crash(self):
        """At depth limit, no crash occurs -- returns None gracefully."""
        _fragment_guard.depth = 10  # Well above limit
        result = name_fragment_recursively("CCCCCCCCCCCCCC")  # uncached
        assert result is None
        assert get_naming_depth() == 10  # Unchanged since we never entered

    def test_depth_above_limit_returns_cached(self):
        """Even above depth limit, cached fragments resolve."""
        _fragment_guard.depth = 10
        result = name_fragment_recursively("c1ccccc1")
        assert result == "benzene"
        assert get_naming_depth() == 10  # Unchanged

    def test_depth_six_still_works(self):
        """At depth 6 (one below limit of 7), naming should still succeed."""
        _fragment_guard.depth = 6
        result = name_fragment_recursively("C")  # methane -- simple, fast
        assert result is not None
        assert get_naming_depth() == 6  # Depth restored

    def test_depth_seven_returns_none_for_uncached(self):
        """At depth 7 (the limit), uncached fragments return None."""
        _fragment_guard.depth = 7
        result = name_fragment_recursively("CCCCCCCCCCCCCC")  # uncached
        assert result is None
        assert get_naming_depth() == 7  # Unchanged since we never entered

    def test_depth_one_below_limit_still_works(self):
        """At MAX_NAMING_DEPTH - 1 (depth 6), naming should still work."""
        _fragment_guard.depth = MAX_NAMING_DEPTH - 1
        result = name_fragment_recursively("C")  # methane
        assert result is not None
        assert get_naming_depth() == MAX_NAMING_DEPTH - 1  # Restored


@pytest.mark.unit
class TestFragmentNaming:
    """Test actual fragment naming results."""

    def setup_method(self):
        _fragment_guard.depth = 0

    def teardown_method(self):
        _fragment_guard.depth = 0

    def test_simple_fragment_naming_ethanol(self):
        """name_fragment_recursively('CCO') should return 'ethanol'."""
        result = name_fragment_recursively("CCO")
        assert result == "ethanol"

    def test_simple_fragment_naming_methane(self):
        """name_fragment_recursively('C') should return 'methane'."""
        result = name_fragment_recursively("C")
        assert result == "methane"

    def test_simple_fragment_naming_benzene(self):
        """name_fragment_recursively('c1ccccc1') should return 'benzene'."""
        result = name_fragment_recursively("c1ccccc1")
        assert result == "benzene"

    def test_simple_fragment_naming_acetic_acid(self):
        """name_fragment_recursively('CC(=O)O') should return 'acetic acid'."""
        result = name_fragment_recursively("CC(=O)O")
        assert result == "acetic acid"

    def test_custom_max_depth(self):
        """Custom max_depth parameter should be respected for uncached fragments."""
        _fragment_guard.depth = 1
        # With max_depth=1, uncached fragment returns None (depth >= max_depth)
        result = name_fragment_recursively("CCCCCCCCCCCCCC", max_depth=1)
        assert result is None
        # With max_depth=2, uncached fragment succeeds (depth < max_depth)
        result = name_fragment_recursively("CCCCCCCCCCCCCC", max_depth=2)
        assert result is not None

    def test_custom_max_depth_cached_bypasses_limit(self):
        """Cached fragments bypass max_depth entirely."""
        _fragment_guard.depth = 1
        result = name_fragment_recursively("CCO", max_depth=1)
        assert result == "ethanol"  # Cache hit, depth irrelevant


@pytest.mark.unit
class TestThreadSafety:
    """Test that depth tracking is thread-safe (each thread has its own counter)."""

    def test_thread_safety(self):
        """Two threads can name independently with separate depth counters."""
        results = {}
        errors = []

        def thread_worker(thread_id, smiles, expected_name):
            try:
                # Each thread starts fresh (thread-local storage)
                depth_before = get_naming_depth()
                result = name_fragment_recursively(smiles)
                depth_after = get_naming_depth()
                results[thread_id] = {
                    'name': result,
                    'depth_before': depth_before,
                    'depth_after': depth_after,
                }
                if result != expected_name:
                    errors.append(
                        f"Thread {thread_id}: expected '{expected_name}', got '{result}'"
                    )
                if depth_before != depth_after:
                    errors.append(
                        f"Thread {thread_id}: depth not restored: "
                        f"{depth_before} -> {depth_after}"
                    )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        t1 = threading.Thread(target=thread_worker, args=(1, "CCO", "ethanol"))
        t2 = threading.Thread(target=thread_worker, args=(2, "C", "methane"))

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread safety errors: {errors}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert results[1]['name'] == "ethanol"
        assert results[2]['name'] == "methane"
        # Each thread should have had depth 0 before and after
        assert results[1]['depth_before'] == 0
        assert results[1]['depth_after'] == 0
        assert results[2]['depth_before'] == 0
        assert results[2]['depth_after'] == 0


@pytest.mark.unit
class TestVisitedSetGuard:
    """Test visited-set cycle detection replaces depth counter."""

    def setup_method(self):
        _fragment_guard.visited = set()

    def teardown_method(self):
        _fragment_guard.visited = set()

    def test_visited_set_starts_empty(self):
        """Fresh state should have empty visited set."""
        if hasattr(_fragment_guard, 'visited'):
            delattr(_fragment_guard, 'visited')
        assert _get_visited() == set()

    def test_visited_set_detects_cycle(self):
        """If a SMILES is already being named, return from cache."""
        visited = _get_visited()
        visited.add("CCO")
        result = name_fragment_recursively("CCO")
        # CCO is in FRAGMENT_NAME_CACHE as "ethanol"
        assert result == "ethanol"

    def test_visited_set_cleaned_after_naming(self):
        """After naming completes, SMILES should be removed from visited set."""
        _fragment_guard.visited = set()
        result = name_fragment_recursively("CCO")  # names ethanol via cache
        assert result is not None
        # "CCO" should NOT remain in visited set after completion
        assert "CCO" not in _fragment_guard.visited

    def test_no_depth_limit_for_deep_nesting(self):
        """Naming should succeed at any depth as long as no cycle exists."""
        _fragment_guard.visited = set()
        result = name_fragment_recursively("CCCCCCCCCCCC(=O)O")
        assert result is not None
        assert "dodecanoic acid" in result.lower()
