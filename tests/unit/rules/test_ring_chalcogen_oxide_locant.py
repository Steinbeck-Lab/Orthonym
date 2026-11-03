"""A ring sulfur oxide takes the lowest locant its ring system allows.

The additive name cites the chalcogen's locant in the parent's fixed numbering. An automorphism of
the ring system renumbers it as validly as the map itself, so the lowest locant the chalcogen can
take is cited, the Blue Book: "Low locants are allocated first in accordance with
the fixed numbering of the ring system"): 'thianthrene 5-oxide', not '10-oxide' (both sulfur atoms
are equivalent). With the corrected phenoxathiine map (b),:12541: O before S, so O5
and S10) the oxide is 'phenoxathiine 10-oxide'.
Every name reads back to the input's full InChIKey with OPSIN 2.9.0.

 (:29436) expresses a ring -SO-/-SO2- with the '-one' suffix on a lambda-6 heterocycle
('5H-λ6-thianthrene-5,5-dione (PIN) thianthrene 5,5-dioxide',:29444): the rows here take that heterone name
on the catalogue parent (``rules/catalogue_heterones``), with the same lowest locant; the saturated
monocycle, 'thiolane 1-oxide', has its heterone name ('1λ4-thiolan-1-one', the same test module).
"""
import pytest

from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("C1=CC=CC=2S(C3=CC=CC=C3SC12)=O", "5H-5λ4-thianthren-5-one"),
    ("C1=CC=CC=2S(C3=CC=CC=C3SC12)(=O)=O", "5H-5λ6-thianthrene-5,5-dione"),
    ("C1=CC=CC=2OC3=CC=CC=C3S(C12)=O", "10H-10λ4-phenoxathiin-10-one"),
    ("C1=CC=CC=2OC3=CC=CC=C3S(C12)(=O)=O", "10H-10λ6-phenoxathiine-10,10-dione"),
    ("O=S1c2ccccc2-c2ccccc21", "5H-5λ4-dibenzo[b,d]thiophen-5-one"),
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
