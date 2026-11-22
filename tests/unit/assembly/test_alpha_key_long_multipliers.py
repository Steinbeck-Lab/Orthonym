"""Every multiplying prefix is ignored in alphanumerical order, not only those up to 'dodeca'.

### **** ALPHANUMERICAL ORDER, (the Blue Book): "Simple prefixes... are first alphabetized; the
multiplying prefixes, if necessary, are then inserted"; the numerical terms are those of (:2790) and
 (:2826), composed past 'icosa' ('henicosa', 'pentacosa', 'triaconta'). (:25799):
'1,1,1,2,2,3,3,4,4,5,5,6,6,7,8,8,9,9,10,10,11,11,12,12,12-pentacosafluoro-7-(1,1,1,3,3,3-hexafluoro-2-methylpropan-
2-yl)dodecane (PIN)': 'pentacosafluoro' is cited at 'f', before the compound prefix that begins with 'h',
:3477). The sort key stopped at 'dodeca' and keyed 'pentacosafluoro' at 'p'.
"""
import pytest

from orthonym.assembly.naming_utils import alpha_sort_key
from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("prefix,key", [
    ("tridecafluoro", "fluoro"), ("hexadecafluoro", "fluoro"), ("icosafluoro", "fluoro"),
    ("pentacosafluoro", "fluoro"), ("hexatriacontafluoro", "fluoro"), ("tetradecakis(methyl)", "methyl"),
    # a chain stem that begins like a multiplier is part of the name
    ("tridecanoyl", "tridecanoyl"), ("hexadecan-2-yl", "hexadecan-2-yl"), ("pentacosyl", "pentacosyl"),
    ("tritriacontyl", "tritriacontyl"), ("hexatriacontyl", "hexatriacontyl"), ("docosanamido", "docosanamido"),
    #... and so is a chain stem followed by its multiplied 'en'
    ("docosahexaenoyl", "docosahexaenoyl"), ("icosatetraenoyl", "icosatetraenoyl"),
])
def test_multiplier_left_out_of_the_key(prefix, key):
    assert str(alpha_sort_key(prefix)) == key


@pytest.mark.opsin_gate
def test_pentacosafluoro_cited_before_a_compound_h_prefix():
    smiles = ("CC(C(F)(F)F)(C(F)(F)F)C(F)(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)"
              "C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F")
    name = ("1,1,1,2,2,3,3,4,4,5,5,6,6,7,8,8,9,9,10,10,11,11,12,12,12-pentacosafluoro-7-"
            "(1,1,1,3,3,3-hexafluoro-2-methylpropan-2-yl)dodecane")
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
