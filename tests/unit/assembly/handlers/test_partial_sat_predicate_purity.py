"""WR-02 regression: predicates do not re-run _is_complex_ring_system per call.

Per 160-REVIEW.md WR-02: the partial_sat, polycyclic, and ring_ester
predicates each call ``composer._is_complex_ring_system(mol)`` directly.
Three predicates running the same SMARTS check per dispatch is wasteful.
Plan-05 introduces a shared memoization helper
``handlers._handler_shared.cached_is_complex_ring_system`` that caches
the result on the features object via a private attribute.

This test asserts the memoization is in place and the cache is shared
across predicates (one SMARTS call per features instance).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_cached_is_complex_ring_system_memoizes_per_features():
    """Multiple predicate calls on the same features → single SMARTS call."""
    from orthonym.assembly.handlers._handler_shared import (
        cached_is_complex_ring_system,
    )

    call_count = [0]

    def fake_check(mol):
        call_count[0] += 1
        return False

    class FakeFeatures:
        mol = MagicMock()
        is_cyclic = True
        chain_is_parent = False

    feats = FakeFeatures()
    with patch(
        "orthonym.assembly.composer._is_complex_ring_system",
        side_effect=fake_check,
    ):
        for _ in range(10):
            cached_is_complex_ring_system(feats)
        assert call_count[0] == 1  # only the first call ran the SMARTS check


def test_predicates_share_the_cache_across_handlers():
    """partial_sat, polycyclic, ring_ester all read the same cache."""
    from orthonym.assembly.handlers._handler_shared import (
        cached_is_complex_ring_system,
    )

    call_count = [0]

    def fake_check(mol):
        call_count[0] += 1
        return False

    class FakeFeatures:
        mol = MagicMock()
        is_cyclic = True
        chain_is_parent = False
        principal_group = None
        ring_systems = None

    feats = FakeFeatures()
    with patch(
        "orthonym.assembly.composer._is_complex_ring_system",
        side_effect=fake_check,
    ):
        cached_is_complex_ring_system(feats)
        cached_is_complex_ring_system(feats)
        cached_is_complex_ring_system(feats)
        assert call_count[0] == 1


def test_predicates_use_cached_helper():
    """All three predicates import cached_is_complex_ring_system instead of
    calling composer._is_complex_ring_system directly."""
    import inspect
    from orthonym.assembly.handlers import partial_sat, polycyclic, ring_ester
    for mod in (partial_sat, polycyclic, ring_ester):
        src = inspect.getsource(mod)
        assert "cached_is_complex_ring_system" in src, (
            f"{mod.__name__} must use cached_is_complex_ring_system"
        )
