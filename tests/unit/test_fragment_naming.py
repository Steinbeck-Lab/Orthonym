"""Unit tests for fragment naming with cycle-detection guard.

Tests the thread-safe visited-set cycle detection infrastructure in
orthonym.assembly.fragment_naming. This module prevents infinite loops
when fragment naming calls name_compound() recursively.
"""
import threading

import pytest

from orthonym.assembly.fragment_naming import (
    FRAGMENT_NAME_CACHE,
    MAX_NAMING_DEPTH,
    _MAX_VISITED_SIZE,
    _get_visited,
    get_naming_depth,
    is_top_level_naming,
    name_fragment_recursively,
    start_naming_session,
    end_naming_session,
    _fragment_guard,
)


@pytest.fixture(autouse=True)
def _reset_fragment_guard():
    """Reset thread-local state before and after each test."""
    _fragment_guard.visited = set()
    _fragment_guard.cache = None
    yield
    _fragment_guard.visited = set()
    _fragment_guard.cache = None


@pytest.mark.unit
class TestVisitedSetTracking:
    """Test visited-set depth proxy behavior."""

    def test_naming_depth_starts_at_zero(self):
        """Initial depth (visited set size) should be 0."""
        if hasattr(_fragment_guard, 'visited'):
            delattr(_fragment_guard, 'visited')
        assert get_naming_depth() == 0

    def test_naming_depth_returns_visited_size(self):
        """get_naming_depth() returns the size of the visited set."""
        visited = _get_visited()
        visited.add("AAA")
        visited.add("BBB")
        assert get_naming_depth() == 2

    def test_is_top_level_when_visited_empty(self):
        """is_top_level_naming() returns True when visited set is empty."""
        assert is_top_level_naming()

    def test_not_top_level_when_visited_nonempty(self):
        """is_top_level_naming() returns False when visited set has entries."""
        _get_visited().add("CCO")
        assert not is_top_level_naming()

    def test_naming_depth_zero_after_call(self):
        """After a successful call, visited set should be empty again."""
        result = name_fragment_recursively("CCO")  # ethanol (cached)
        assert result == "ethanol"
        assert get_naming_depth() == 0

    def test_visited_set_preserved_for_parent(self):
        """Parent visited entries should remain after a nested call completes."""
        visited = _get_visited()
        visited.add("FAKE_PARENT_SMILES")
        result = name_fragment_recursively("CCO")  # ethanol (cached)
        assert result == "ethanol"
        # Parent's entry is still there
        assert "FAKE_PARENT_SMILES" in visited
        assert get_naming_depth() == 1  # Just the parent entry

    def test_naming_depth_restores_on_invalid_smiles(self):
        """Visited set should restore even for invalid SMILES."""
        result = name_fragment_recursively("INVALID_NOT_A_SMILES_XYZ")
        assert result is None
        assert get_naming_depth() == 0


@pytest.mark.unit
class TestCycleDetection:
    """Test that cycle detection prevents infinite recursion."""

    def test_cycle_detected_returns_none_for_uncached(self):
        """If a SMILES is already being named, return None for uncached fragments."""
        visited = _get_visited()
        canonical = "CCCCCCCCCCCCCC"  # tetradecane (not in static cache)
        visited.add(canonical)
        result = name_fragment_recursively(canonical)
        assert result is None

    def test_cached_fragments_bypass_cycle_detection(self):
        """Static-cached fragments resolve even when in the visited set."""
        visited = _get_visited()
        visited.add("CCO")  # ethanol is in FRAGMENT_NAME_CACHE
        result = name_fragment_recursively("CCO")
        assert result == "ethanol"

    def test_safety_net_returns_none(self):
        """When visited set reaches _MAX_VISITED_SIZE, return None."""
        visited = _get_visited()
        for i in range(_MAX_VISITED_SIZE):
            visited.add(f"FAKE_SMILES_{i}")
        # Now at the limit — new uncached SMILES should return None
        result = name_fragment_recursively("CCCCCCCCCCCCCC")
        assert result is None

    def test_safety_net_allows_cached(self):
        """Even at safety net limit, cached fragments still resolve."""
        visited = _get_visited()
        for i in range(_MAX_VISITED_SIZE):
            visited.add(f"FAKE_SMILES_{i}")
        result = name_fragment_recursively("CCO")
        assert result == "ethanol"

    def test_visited_set_cleaned_after_naming(self):
        """After naming completes, SMILES should be removed from visited set."""
        result = name_fragment_recursively("CCO")
        assert result is not None
        assert "CCO" not in _get_visited()

    def test_no_depth_limit_for_deep_but_finite_nesting(self):
        """Naming should succeed beyond old MAX_NAMING_DEPTH if no cycle."""
        result = name_fragment_recursively("CCCCCCCCCCCC(=O)O")
        assert result is not None
        assert "dodecanoic acid" in result.lower()


@pytest.mark.unit
class TestLegacyConstants:
    """Test backward-compatibility constants are preserved."""

    def test_max_naming_depth_is_seven(self):
        """MAX_NAMING_DEPTH legacy constant should be 7."""
        assert MAX_NAMING_DEPTH == 7

    def test_max_visited_size_is_twenty(self):
        """Safety-net limit should be 20."""
        assert _MAX_VISITED_SIZE == 20


@pytest.mark.unit
class TestFragmentNaming:
    """Test actual fragment naming results."""

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

    def test_max_depth_param_ignored(self):
        """max_depth parameter is legacy and should not affect behavior."""
        # Even with max_depth=1, naming should work (cycle detection, not depth)
        result = name_fragment_recursively("CCO", max_depth=1)
        assert result == "ethanol"  # Cached, always works

    def test_static_cache_has_common_fragments(self):
        """FRAGMENT_NAME_CACHE should contain common fragments."""
        assert FRAGMENT_NAME_CACHE.get("CCO") == "ethanol"
        assert FRAGMENT_NAME_CACHE.get("c1ccccc1") == "benzene"
        assert FRAGMENT_NAME_CACHE.get("CC(=O)O") == "acetic acid"


@pytest.mark.unit
class TestSessionManagement:
    """Test naming session start/end lifecycle."""

    def test_start_session_initializes_cache(self):
        """start_naming_session() should create a runtime cache."""
        assert getattr(_fragment_guard, 'cache', None) is None
        start_naming_session()
        assert _fragment_guard.cache == {}

    def test_end_session_clears_cache(self):
        """end_naming_session() should clear the runtime cache."""
        start_naming_session()
        _fragment_guard.cache["test"] = "value"
        end_naming_session()
        assert _fragment_guard.cache is None

    def test_nested_session_preserves_parent(self):
        """Starting a session while visited set is non-empty should not reset."""
        start_naming_session()
        _fragment_guard.cache["parent"] = "value"
        _get_visited().add("PARENT_SMILES")
        # Nested start should NOT clear parent cache
        start_naming_session()
        assert "parent" in _fragment_guard.cache

    def test_nested_end_preserves_parent(self):
        """Ending while visited set is non-empty should not clear."""
        start_naming_session()
        _fragment_guard.cache["parent"] = "value"
        _get_visited().add("PARENT_SMILES")
        end_naming_session()
        # Cache should still exist because visited set is non-empty
        assert _fragment_guard.cache is not None


@pytest.mark.unit
class TestThreadSafety:
    """Test that visited-set tracking is thread-safe."""

    def test_thread_safety(self):
        """Two threads can name independently with separate visited sets."""
        results = {}
        errors = []

        def thread_worker(thread_id, smiles, expected_name):
            try:
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
        assert len(results) == 2
        assert results[1]['name'] == "ethanol"
        assert results[2]['name'] == "methane"
        assert results[1]['depth_before'] == 0
        assert results[1]['depth_after'] == 0
        assert results[2]['depth_before'] == 0
        assert results[2]['depth_after'] == 0
