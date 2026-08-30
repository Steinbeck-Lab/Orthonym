"""The floor's spiro locant sort key must be LETTER-aware (P-14.5.2 / P-31.1.4).

M4-L2 (`ab9c65ab`) admits a fused component's lettered ring-fusion locants
(`4a`, `8a`) into the combined locant map. The sort key that orders substituent
citation must then distinguish:
  * a lettered locant from a primed one of the same number (`8a` != `8'`), and
  * order `4 < 4a < 5`,
and must not CRASH on the `_Locant` tuple form of a lettered locant
(`('8a', "'")`). The old `(prime_count, number)` key collided `8a` with `8'` and
raised ValueError on `('8a', "'")` -- a wrong substituent order that the
full-InChIKey offer gate cannot see (FABLE review #17, the spelling-layer blind
spot). Finding: .
"""
from orthonym.assembly.universal_substituent import _locant_sort_key


def test_lettered_locant_distinct_from_primed():
    # a fusion locant 8a must NOT sort identically to the primed locant 8'
    assert _locant_sort_key("8a") != _locant_sort_key("8'")


def test_letter_orders_after_bare_number_before_next():
    # 4 < 4a < 5  (P-14.5.2: the fusion letter is a suffix of position 4)
    assert _locant_sort_key("4") < _locant_sort_key("4a") < _locant_sort_key("5")


def test_prime_axis_dominates_number():
    # all unprimed precede all single-primed (a spiro locant-set invariant)
    assert _locant_sort_key("8a") < _locant_sort_key("1'")


def test_lettered_tuple_form_does_not_crash():
    # the _Locant tuple carrying a lettered base must not raise
    key = _locant_sort_key(("8a", "'"))
    # and it must land on the single-prime axis, distinct from unprimed 8a
    assert key > _locant_sort_key("8a")


def test_plain_int_and_string_consistent():
    assert _locant_sort_key(3) < _locant_sort_key(4)
    assert _locant_sort_key("3") < _locant_sort_key("4")
