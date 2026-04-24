"""Phase 146 Plan 01 Task 3: verify thread-local auto-clear fixture works.

The autouse fixture ``_phase146_clear_thread_locals`` in ``tests/conftest.py``
runs after every test, clearing the three thread-local stores. These three
sanity tests confirm that a fresh test sees a clean state (no leakage from
the previous test's molecule-naming work).

Reference: 146-RESEARCH.md §8.4.
"""
import pytest


def test_pool_store_cleared_between_tests():
    """If conftest fixture is active, _pool_store.stack contains at most
    a single fresh pool at test start (no leaked candidates from prior runs).

    Note: clear_pool() replaces the top of the stack with a fresh pool
    (or pushes one if empty), so the stack should hold 0 or 1 pools at
    test start — never more.
    """
    from orthonym.assembly.candidate_pool import _ensure_stack
    stack = _ensure_stack()
    assert len(stack) <= 1, (
        f"Stack contains {len(stack)} pools; cleanup failed "
        f"(prior test leaked nested pools)"
    )


def test_pc_context_cleared_between_tests():
    """If conftest fixture is active, _pc_context.reference_name is None."""
    from orthonym.rules.parent_correctness import _pc_context
    ref = getattr(_pc_context, 'reference_name', None)
    assert ref is None, (
        f"Stale reference_name={ref!r}; cleanup failed "
        f"(prior test set _pc_context.reference_name without clearing)"
    )


def test_confidence_store_cleared_between_tests():
    """If conftest fixture is active, _confidence_store has no leftover
    candidate (retrieve_confidence returns the empty default dict).
    """
    from orthonym.assembly.coverage_scoring import retrieve_confidence
    info = retrieve_confidence()
    # After clear_confidence(), retrieve returns the empty default dict
    assert info.get('name') == '', (
        f"Stale confidence: {info!r}; cleanup failed"
    )
    assert info.get('handler') == 'unknown', (
        f"Stale confidence handler: {info!r}; cleanup failed"
    )
