"""A diacyl chalcogen chain is named on its longer acyl chain; an acyl on a chalcogen prefix is compound.

IUPAC 2013 (the Blue Book) with (b): 'CH3-CH2-CO-OO-S-CO-CH3
1-[(acetylsulfanyl)peroxy]propan-1-one (PIN)' (:39499); enclosing marks for a compound prefix.
"""

import pytest

from orthonym.assembly.naming_utils import is_complex_substituent
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("prefix,compound", [
    ("acetylsulfanyl", True), ("phenylsulfanyl", True), ("methylsulfonyl", True),
    ("methanesulfonyl", False), ("benzenesulfonyl", False), ("disulfanyl", False), ("sulfanyl", False),
])
def test_compound_chalcogen_prefixes(prefix, compound):
    assert is_complex_substituent(prefix) is compound


@pytest.mark.opsin_gate
def test_the_longer_acyl_chain_is_the_parent():
    assert name_best_effort("CCC(=O)OOSC(C)=O")["name"] == "1-[(acetylsulfanyl)peroxy]propan-1-one"
    assert name_best_effort("CC(=O)OOSC(C)=O")["name"] == "1-[(acetylperoxy)sulfanyl]ethan-1-one"
