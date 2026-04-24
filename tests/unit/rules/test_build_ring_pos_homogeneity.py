"""
Tests for _build_ring_pos tuple-locant acceptance and homogeneity coercion.

Phase 147 Plan 01 Task 2 (SC-1, SC-7 evidence).

These tests cover:
- Pure-int back-compat (fallback to sorted atom indices when no ring_info)
- Pure-int iupac_locants pass-through (D-09 back-compat)
- Mixed int + (int, str) tuple iupac_locants → homogeneity coercion at return
- Pure-tuple iupac_locants pass-through
- Partial coverage falls back to sorted-int (D-09 back-compat)
- Regression test for parent_selection.py:416 min() hazard
  (RESEARCH §3 Risk 3): mixed int + tuple in the returned dict would cause
  min(ring_pos[a], ring_pos[b]) to raise TypeError. After Phase 147 coercion,
  the dict is homogeneous and min() is safe.
- Non-int/tuple locant values (e.g. legacy string) are filtered out and
  partial coverage triggers sorted fallback.

IUPAC source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
              https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.5.2, P-14.7

Project source: Phase 147 CONTEXT D-01 (tuple encoding);
                Phase 147 CONTEXT D-07, D-09 (back-compat preservation);
                Phase 147 RESEARCH §3 Risk 3 (min() hazard at :416).
"""

from orthonym.rules.parent_selection import _build_ring_pos


# ----------------------------------------------------------------------------
# Pure-int back-compat
# ----------------------------------------------------------------------------


def test_fallback_sorted_when_no_ring_info():
    """No ring_info → sorted atom indices mapped 1-indexed (back-compat).

    Source: Phase 147 CONTEXT D-09 (sorted-fallback preserved for
            non-cascade-step-6 callers).
    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.5.2
    """
    result = _build_ring_pos({0, 1, 2, 3}, None)
    assert result == {0: 1, 1: 2, 2: 3, 3: 4}


def test_pure_int_iupac_locants_passthrough():
    """Pure-int iupac_locants pass through unchanged (back-compat).

    Source: Phase 147 CONTEXT D-09.
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
    """
    ring_info = {"iupac_locants": {0: 1, 1: 2, 2: 3}}
    result = _build_ring_pos({0, 1, 2}, ring_info)
    assert result == {0: 1, 1: 2, 2: 3}
    # Every value must be int (no tuple coercion when no tuple present).
    assert all(isinstance(v, int) for v in result.values())


# ----------------------------------------------------------------------------
# Phase 147 tuple acceptance + homogeneity coercion
# ----------------------------------------------------------------------------


def test_mixed_int_and_tuple_coerces_all_to_tuple():
    """Any tuple present → all int values coerced to (n, '') tuples.

    Input: {0:1, 1:2, 2:3, 3:(4,'a')}
    Output: {0:(1,''), 1:(2,''), 2:(3,''), 3:(4,'a')}

    This is the core Phase 147 D-01 behavior — homogeneity invariant
    enforced at function return so downstream min()/sort() is safe.

    Source: Phase 147 CONTEXT D-01 (tuple encoding).
    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.7
    """
    ring_info = {"iupac_locants": {0: 1, 1: 2, 2: 3, 3: (4, 'a')}}
    result = _build_ring_pos({0, 1, 2, 3}, ring_info)
    # All values must be tuples (homogeneity).
    assert all(isinstance(v, tuple) for v in result.values()), result
    assert result == {0: (1, ''), 1: (2, ''), 2: (3, ''), 3: (4, 'a')}


def test_pure_tuple_iupac_locants_passthrough():
    """Pure-tuple iupac_locants pass through unchanged (already homogeneous).

    Source: Phase 147 CONTEXT D-01.
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
    """
    ring_info = {"iupac_locants": {0: (1, ''), 1: (2, 'a')}}
    result = _build_ring_pos({0, 1}, ring_info)
    assert result == {0: (1, ''), 1: (2, 'a')}
    assert all(isinstance(v, tuple) for v in result.values())


# ----------------------------------------------------------------------------
# Partial coverage → sorted fallback (D-09 back-compat)
# ----------------------------------------------------------------------------


def test_partial_coverage_falls_back_to_sorted():
    """When iupac_locants covers only some ring atoms, fall back to sorted.

    Per D-09 back-compat: non-cascade-step-6 callers rely on the sorted
    proxy. Cascade step 6 gates on complete coverage via _has_iupac_locants
    at candidate_pool.py so this fallback is safe there.

    Source: Phase 147 CONTEXT D-09 (sorted-fallback preserved).
    """
    ring_info = {"iupac_locants": {0: 1}}  # only 1 of 3 atoms covered
    result = _build_ring_pos({0, 1, 2}, ring_info)
    # Fallback result — sorted 1-indexed mapping.
    assert result == {0: 1, 1: 2, 2: 3}


def test_non_int_non_tuple_values_filtered_triggering_fallback():
    """Legacy string locants are filtered out; incomplete coverage triggers
    sorted fallback.

    Source: Phase 147 CONTEXT D-09.
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
    """
    # One valid tuple, one invalid string — string is filtered, leaving
    # incomplete coverage, so sorted fallback kicks in.
    ring_info = {"iupac_locants": {0: (1, 'a'), 1: "invalid"}}
    result = _build_ring_pos({0, 1}, ring_info)
    # Fallback: sorted-int {0:1, 1:2}.
    assert result == {0: 1, 1: 2}
    assert all(isinstance(v, int) for v in result.values())


# ----------------------------------------------------------------------------
# Regression guard for parent_selection.py:416 min() hazard
# ----------------------------------------------------------------------------


def test_min_safe_after_coercion():
    """Regression guard for parent_selection.py:416 min() hazard.

    _compare_multiple_bond_locants does min(ring_pos[a], ring_pos[b]).
    Mixed int + tuple would raise TypeError on Python 3. After Phase 147
    homogeneity coercion inside _build_ring_pos, every value is the
    same type, so min() is safe.

    This is the SC-7 acceptance evidence — the hazard that made Phase 147
    a correctness prerequisite rather than an optional polish.

    Source: Phase 147 RESEARCH §3 Risk 3.
    Source: Phase 146 D-19 (locant type safety lock-in).
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
    """
    ring_pos = _build_ring_pos(
        {0, 1, 2, 3},
        {"iupac_locants": {0: 1, 1: 2, 2: 3, 3: (4, 'a')}},
    )
    # Every value must be a tuple (coercion applied).
    assert all(isinstance(v, tuple) for v in ring_pos.values())
    # min() over any pair must not raise TypeError.
    result = min(ring_pos[0], ring_pos[3])
    assert result == (1, '')
    # Also: min across all values
    overall_min = min(ring_pos.values())
    assert overall_min == (1, '')
