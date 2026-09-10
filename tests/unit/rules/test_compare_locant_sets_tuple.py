"""
Tests for tuple-aware compare_locant_sets and _assert_homogeneous_locants.

a phase Plan 01 Task 1 (, evidence).

These tests cover:
- Pure-int back-compat (the existing 3 doctests must also pass byte-identical)
- Pure-tuple first-point-of-difference on (int_base, str_suffix) fusion locants
- Mixed int + tuple coercion where int `n` becomes `(n, '')` so that empty-string
  sorts lexicographically before any letter (IUPAC: locant `4` < `4a`)
- The homogeneity guard `_assert_homogeneous_locants` rejects true heterogeneous
  lists with a ValueError whose message mentions both "mixed" and the offending types

IUPAC source: https://iupac.qmul.ac.uk/BlueBook/P1.html
  - First-point-of-difference rule
  - Locant set comparison
  - (g) Lowest locants for substituents (alphabetical tiebreaker context)

Project source: a phase (locant type safety lock-in);
                a phase internal notes, (tuple encoding + coercion strategy);
                a phase RESEARCH Risk 3 (min hazard downstream).
AUTONOM-1990 (criterion-order comparison after permutation generation).
"""

import pytest

from orthonym.rules.locants import (
    _assert_homogeneous_locants,
    compare_locant_sets,
)


# ----------------------------------------------------------------------------
# Pure-int back-compat (the existing doctests expressed as explicit test cases)
# ----------------------------------------------------------------------------


def test_pure_int_first_point_of_difference_a_wins():
    """Pure-int fast path: a=[2,3,5] vs b=[3,4,6] → a wins at position 0.

    IUPAC first-point-of-difference rule.
    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    Also the first existing doctest — MUST preserve byte-identical.
    """
    assert compare_locant_sets([2, 3, 5], [3, 4, 6]) == -1


def test_pure_int_first_point_of_difference_b_wins():
    """Pure-int fast path: a=[2,4,5] vs b=[2,3,5] → b wins at position 1.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    Second existing doctest — back-compat gate.
    """
    assert compare_locant_sets([2, 4, 5], [2, 3, 5]) == 1


def test_pure_int_identical_sets():
    """Pure-int fast path: identical sets return 0.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    Third existing doctest — back-compat gate.
    """
    assert compare_locant_sets([2, 3], [2, 3]) == 0


# ----------------------------------------------------------------------------
# Pure-tuple comparison (a phase new behavior)
# ----------------------------------------------------------------------------


def test_pure_tuple_equivalent_to_pure_int_when_all_empty_suffix():
    """Pure-tuple with empty suffix is the tuple-equivalent of pure-int comparison.

    a=[(2,''),(3,''),(5,'')] vs b=[(3,''),(4,''),(6,'')] → a wins at position 0.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    a phase (locant type safety).
    """
    a = [(2, ''), (3, ''), (5, '')]
    b = [(3, ''), (4, ''), (6, '')]
    assert compare_locant_sets(a, b) == -1


def test_fusion_atom_empty_string_sorts_before_letter():
    """Fusion-atom ordering: `4` < `4a` because '' < 'a' lexicographically.

    a=[(4,''),(5,'')] vs b=[(4,'a'),(5,'')] → a wins at position 0.

    IUPAC convention: plain locant 4 is "lower" than fusion locant 4a.
    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    a phase internal notes (tuple encoding with empty-string-prefix ordering).
    """
    a = [(4, ''), (5, '')]
    b = [(4, 'a'), (5, '')]
    assert compare_locant_sets(a, b) == -1


def test_fusion_letters_order_a_before_b():
    """Fusion-atom ordering among letters: `4a` < `4b`.

    a=[(4,'a'),(5,'')] vs b=[(4,'b'),(5,'')] → a wins at position 0.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    AUTONOM-1990 (criterion-order comparison).
    """
    a = [(4, 'a'), (5, '')]
    b = [(4, 'b'), (5, '')]
    assert compare_locant_sets(a, b) == -1


def test_fusion_base_beats_next_integer():
    """Fusion locant `4b` is still below `5` because (4,_) < (5,'').

    a=[(4,'b'),(5,'')] vs b=[(5,''),(5,'')] → a wins at position 0.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    a phase internal notes (tuple encoding).
    """
    a = [(4, 'b'), (5, '')]
    b = [(5, ''), (5, '')]
    assert compare_locant_sets(a, b) == -1


def test_shorter_tuple_set_wins_on_prefix_tie():
    """Shorter set wins when all shared positions are equal.

    a=[(2,''),(3,'')] vs b=[(2,''),(3,''),(5,'')] → a wins (fewer locants).

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    Back-compat contract for the tuple path (matches pure-int semantics).
    """
    a = [(2, ''), (3, '')]
    b = [(2, ''), (3, ''), (5, '')]
    assert compare_locant_sets(a, b) == -1


# ----------------------------------------------------------------------------
# Mixed int + tuple coercion (a phase coercion semantics)
# ----------------------------------------------------------------------------


def test_mixed_ints_with_one_tuple_coerces_via_empty_string_a_wins():
    """Mixed ints with a trailing tuple coerces ints to (n,'') for comparison.

    a=[1,2,(3,'a')] vs b=[1,2,(3,'b')] → coerce to a'=[(1,''),(2,''),(3,'a')]
    and b'=[(1,''),(2,''),(3,'b')]; a wins at position 2 because 'a' < 'b'.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    a phase internal notes (coercion path).
    """
    a = [1, 2, (3, 'a')]
    b = [1, 2, (3, 'b')]
    assert compare_locant_sets(a, b) == -1


def test_mixed_coerces_both_directions_to_identical():
    """Coerce-and-compare works both directions (int-list vs tuple-list).

    a=[1,(2,'a'),3] vs b=[(1,''),(2,'a'),(3,'')] → after coercion both become
    [(1,''),(2,'a'),(3,'')] — identical → returns 0.

    Reverse direction also holds.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    a phase internal notes (symmetric coercion).
    """
    a = [1, (2, 'a'), 3]
    b = [(1, ''), (2, 'a'), (3, '')]
    assert compare_locant_sets(a, b) == 0
    # Reverse direction
    assert compare_locant_sets(b, a) == 0


# ----------------------------------------------------------------------------
# Homogeneity guard (_assert_homogeneous_locants)
# ----------------------------------------------------------------------------


def test_assert_homogeneous_all_int_ok():
    """_assert_homogeneous_locants accepts all-int list silently (returns None).

    Source: a phase (locant type safety lock-in).
    """
    assert _assert_homogeneous_locants([1, 2, 3]) is None


def test_assert_homogeneous_all_tuple_ok():
    """_assert_homogeneous_locants accepts all-tuple list silently (returns None).

    Source: a phase (locant type safety lock-in).
    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html
    """
    assert _assert_homogeneous_locants([(1, ''), (2, ''), (3, 'a')]) is None


def test_assert_homogeneous_rejects_mixed_with_diagnostic_message():
    """Truly heterogeneous list (mixed int + tuple post-coercion-attempt) must
    raise ValueError with a message mentioning both "mixed" and the offending
    types.

    Source: a phase (raises ValueError, not TypeError — actionable for
    callers to fix their construction).
    """
    with pytest.raises(ValueError, match="mixed int and tuple"):
        _assert_homogeneous_locants([1, (2, 'a')])


def test_assert_homogeneous_rejects_mixed_tuple_first():
    """Guard works regardless of which type appears first (order-independent).

    Source: a phase.
    """
    with pytest.raises(ValueError, match="mixed int and tuple"):
        _assert_homogeneous_locants([(1, ''), 2, (3, 'a')])


def test_assert_homogeneous_empty_list_ok():
    """Empty list is trivially homogeneous (vacuously true).

    Source: a phase (no-op on empty — don't raise).
    """
    assert _assert_homogeneous_locants([]) is None
