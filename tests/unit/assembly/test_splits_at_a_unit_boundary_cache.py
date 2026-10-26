"""``naming_utils._splits_at_a_unit_boundary`` is kept per argument for the life of the process.

(c) / (a): a compound prefix is built by concatenating units of the producers' vocabulary; the
function asks whether a prefix body needs two or more. It reads only its arguments and vocabulary tables that
never change after import, so the answer for an argument is one answer whenever it is asked. The tests pin that
claim: the kept answers equal the unkept ones, in any order and from a cold or a warm cache.
"""
import random

import pytest

from orthonym.assembly import naming_utils as nu

pytestmark = pytest.mark.unit

# (c) / (a): two units are a substituted (compound) prefix, one unit a simple one
SPLITS = ["methoxymethyl", "pyridin-2-ylmethyl", "acetamidophenyl", "chloropropan-2-yl", "hydroxymethyl",
          "bromomethyl", "2-methylbutyl", "trimethylsilyl", "difluoromethyl", "carbamoyloxy", "ethyl-1-oxabutyl"]
SIMPLE = ["methyl", "ethyl", "propan-2-yl", "naphthalen-2-yl", "tridecyl", "oxolan-2-yl", "nitroso", "phenyl",
          "benzenesulfinyl", "1-oxabutyl", "cyclohexyl"]


def _bodies():
    """The bodies above, their halves, and shuffled pieces of them: names and non-names."""
    rng = random.Random(20261026)
    names = SPLITS + SIMPLE
    pieces = [n[:k] for n in names for k in range(1, len(n) + 1, 2)] + [n[k:] for n in names for k in range(1, len(n), 3)]
    glued = [a + b for a, b in (rng.sample(names, 2) for _ in range(150))]
    return names + pieces + glued + ["", "-", "(2-chloroethyl)", "1,2-di", "tert-butyl", "4a-methyl"]


def _uncached(body, allow_multiplier=True):
    # the function recurses through the module name, so the cached wrapper is swapped out for the whole call:
    # every depth is computed afresh, not only depth 0
    cached = nu._splits_at_a_unit_boundary
    nu._splits_at_a_unit_boundary = cached.__wrapped__
    try:
        return cached.__wrapped__(body, allow_multiplier)
    finally:
        nu._splits_at_a_unit_boundary = cached


def test_known_prefixes():
    for body in SPLITS:
        assert nu._splits_at_a_unit_boundary(body) is True, body
    for body in SIMPLE:
        assert nu._splits_at_a_unit_boundary(body) is False, body


@pytest.mark.parametrize("allow_multiplier", [True, False])
def test_kept_answers_equal_unkept_answers_in_any_order(allow_multiplier):
    bodies = _bodies()
    unkept = [_uncached(b, allow_multiplier) for b in bodies]
    nu._splits_at_a_unit_boundary.cache_clear()
    backward = [nu._splits_at_a_unit_boundary(b, allow_multiplier) for b in reversed(bodies)][::-1]
    shuffled = list(range(len(bodies)))
    random.Random(7).shuffle(shuffled)
    mixed = {i: nu._splits_at_a_unit_boundary(bodies[i], allow_multiplier) for i in shuffled}
    warm = [nu._splits_at_a_unit_boundary(b, allow_multiplier) for b in bodies]
    assert backward == unkept
    assert [mixed[i] for i in range(len(bodies))] == unkept
    assert warm == unkept


def test_the_answer_is_kept():
    nu._splits_at_a_unit_boundary.cache_clear()
    body = "methoxymethyl"
    nu._splits_at_a_unit_boundary(body)
    before = nu._splits_at_a_unit_boundary.cache_info().hits
    nu._splits_at_a_unit_boundary(body)
    assert nu._splits_at_a_unit_boundary.cache_info().hits == before + 1


def test_multiplier_flag_is_part_of_the_argument():
    """A leading multiplying syllable is peeled only when the flag allows it: the two answers are kept apart."""
    nu._splits_at_a_unit_boundary.cache_clear()
    with_flag = nu._splits_at_a_unit_boundary("dimethyl", True)
    without = nu._splits_at_a_unit_boundary("dimethyl", False)
    assert (with_flag, without) == (_uncached("dimethyl", True), _uncached("dimethyl", False))
    assert nu._splits_at_a_unit_boundary("dimethyl", True) is with_flag
    assert nu._splits_at_a_unit_boundary("dimethyl", False) is without


def test_substituted_substituent_answers_unchanged():
    """(c) / (a), the docstring examples of ``is_substituted_substituent``."""
    for name, expected in [("propan-2-yl", False), ("naphthalen-2-yl", False), ("tridecyl", False),
                           ("hydroxymethyl", True), ("2-methylbutyl", True), ("bromomethyl", True)]:
        assert nu.is_substituted_substituent(name) is expected, name
