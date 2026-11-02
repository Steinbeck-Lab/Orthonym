"""The ring system with more principal characteristic groups is the parent, before ring seniority.

IUPAC 2013 (maximum number of principal characteristic groups) precedes (seniority of rings).
"""

import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("Oc1ccc(CCc2ccc3ccccc3c2O)cc1O", "4-[2-(1-hydroxynaphthalen-2-yl)ethyl]benzene-1,2-diol"),
    ("OC(=O)c1ccc(CCc2ccc3ccccc3c2C(=O)O)cc1C(=O)O",
     "4-[2-(1-carboxynaphthalen-2-yl)ethyl]benzene-1,2-dicarboxylic acid"),
    # a tie: the senior ring (naphthalene) is the parent
    ("Oc1ccc(CCc2ccc3ccccc3c2O)cc1", "2-[2-(4-hydroxyphenyl)ethyl]naphthalen-1-ol"),
    # a tie with a heterocycle: (:19422) compares the rings, a heterocycle is senior:19445, "2-[(naphthalen-2-yl)methyl]pyridine (PIN)")
    ("Oc1ccc(Cc2ccc3ccccc3c2O)cn1", "5-[(1-hydroxynaphthalen-2-yl)methyl]pyridin-2-ol"),
])
def test_parent(smiles, name):
    assert_pin_at_both_tiers(smiles, name)
