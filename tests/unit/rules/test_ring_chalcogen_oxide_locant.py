"""A ring sulfur oxide takes the lowest locant its ring system allows.

The additive name cites the chalcogen's locant in the parent's fixed numbering. An automorphism of
the ring system renumbers it as validly as the map itself, so the lowest locant the chalcogen can
take is cited, the Blue Book: "Low locants are allocated first in accordance with
the fixed numbering of the ring system"): 'thianthrene 5-oxide', not '10-oxide' (both sulfur atoms
are equivalent). With the corrected phenoxathiine map (b),:12541: O before S, so O5
and S10) the oxide is 'phenoxathiine 10-oxide'.
Every name reads back to the input's full InChIKey with OPSIN 2.9.0.

The label of the additive names is not decided here: (:29436) expresses a ring -SO-/-SO2-
with the '-one' suffix on a lambda-6 heterocycle ('5H-λ6-thianthrene-5,5-dione (PIN)
thianthrene 5,5-dioxide',:29444), which is PIN class program Task 21 (lambda ring oxides).
"""
import pytest

from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("C1=CC=CC=2S(C3=CC=CC=C3SC12)=O", "thianthrene 5-oxide"),
    ("C1=CC=CC=2S(C3=CC=CC=C3SC12)(=O)=O", "thianthrene 5,5-dioxide"),
    ("C1=CC=CC=2OC3=CC=CC=C3S(C12)=O", "phenoxathiine 10-oxide"),
    ("C1=CC=CC=2OC3=CC=CC=C3S(C12)(=O)=O", "phenoxathiine 10,10-dioxide"),
    ("O=S1c2ccccc2-c2ccccc21", "dibenzo[b,d]thiophene 5-oxide"),
    ("O=S1CCCC1", "thiolane 1-oxide"),
])
def test_ring_sulfur_oxide_locant(smiles, name):
    b = name_breadth(smiles)
    assert b.get("name") == name, b
    assert name_is_rt_exact(name, smiles)
    # the default tier ships the same string or declines; never another name
    d = name_default(smiles)
    assert d.get("name") in (name, "unknown organic compound", None), d


def test_sultone_keeps_its_pin():
    from tests.support.pin_tiers import assert_pin_at_both_tiers
    assert_pin_at_both_tiers("O=S1(=O)CCCO1", "1,2λ6-oxathiolane-2,2-dione")
