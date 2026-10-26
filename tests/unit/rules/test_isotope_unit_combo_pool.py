"""``rules.isotopes._unit_combo_pool``: the locant specifications tried for one multiplied unit.

 (the descriptor stands inside the multiplier's marks) and (no locant first, then the lowest
locants): the pool is the first ``_UNIT_LOCANT_COMBO_CAP * 40`` tuples of the product of every nuclide's
choices, sorted by the number of cited locants and then by the locants. A nuclide with u labels per unit has
every u-combination of the ``k_max`` positions among its choices (142,506 for u=5 at k_max=30), and only the
first ``pool`` of them can reach a tuple of the pool: the lists are cut to ``pool`` items. The tests compare the
pool with the one built from the full lists.
"""
from itertools import islice, product

import pytest

from orthonym.rules import isotopes as iso

pytestmark = pytest.mark.unit


def _pool_from_full_lists(unit_keys, k_max):
    """The pool as it was built before the lists were cut: every choice of every nuclide, listed in full."""
    choices = [list(iso._unit_locant_choices(el, c, k_max)) for (_mass, el), c, _maxpos in unit_keys]
    return sorted(
        islice(product(*choices), iso._UNIT_LOCANT_COMBO_CAP * 40),
        key=lambda combo: (sum(1 for x in combo if x is not None),
                           [(x if isinstance(x, tuple) else (x,)) if x is not None else () for x in combo]))


# (mass, element), labels per unit, maximum positions: one nuclide or several, small counts and large ones
CASES = {
    "one hydrogen": [((2, "H"), 1, 1)],
    "d5, the phenyl of a terphenyl": [((2, "H"), 5, 5)],
    "d4": [((2, "H"), 4, 4)],
    "d3 and 13C": [((2, "H"), 3, 3), ((13, "C"), 1, 1)],
    "d5 and two 13C": [((2, "H"), 5, 5), ((13, "C"), 2, 2)],
    "three nuclides": [((2, "H"), 3, 3), ((13, "C"), 2, 2), ((15, "N"), 1, 1)],
    "two big nuclides": [((2, "H"), 5, 5), ((13, "C"), 4, 4)],
    "14C pair": [((14, "C"), 2, 2)],
}


@pytest.mark.parametrize("k_max", [3, 6, 12, 30])
@pytest.mark.parametrize("case", CASES)
def test_pool_equals_the_pool_of_the_full_lists(case, k_max):
    unit_keys = CASES[case]
    assert iso._unit_combo_pool(unit_keys, k_max) == _pool_from_full_lists(unit_keys, k_max)


def test_pool_is_cut_where_the_product_is_larger_than_it():
    """The pool is full (6,000 tuples) for the cases whose product is larger, and the cut changed none of them."""
    unit_keys = CASES["two big nuclides"]
    pool = iso._unit_combo_pool(unit_keys, 30)
    assert len(pool) == iso._UNIT_LOCANT_COMBO_CAP * 40
    assert pool == _pool_from_full_lists(unit_keys, 30)


def test_pool_does_not_depend_on_the_order_of_the_calls():
    """No state is kept: the same arguments give the same pool whatever was asked before."""
    first = {name: iso._unit_combo_pool(keys, 30) for name, keys in CASES.items()}
    second = {name: iso._unit_combo_pool(keys, 30) for name, keys in reversed(list(CASES.items()))}
    assert first == second


def test_no_locant_comes_first_then_the_lowest_locants():
    """: the choice with no locant leads, then the lowest locants: the shared locant 1, then the
    positions 1,2 and 1,3 of two labels before the shared locant 2."""
    pool = iso._unit_combo_pool([((2, "H"), 2, 2)], 4)
    assert pool[:5] == [(None,), (1,), ((1, 2),), ((1, 3),), ((1, 4),)]
    assert pool[5] == (2,)


def test_fewer_cited_locants_come_before_more():
    """Two nuclides: a tuple that cites one locant comes before any tuple that cites two."""
    pool = iso._unit_combo_pool([((2, "H"), 1, 1), ((13, "C"), 1, 1)], 3)
    cited = [sum(1 for x in combo if x is not None) for combo in pool]
    assert cited == sorted(cited)
    assert pool[0] == (None, None)
