"""Identical components joined by a space are a correct systematic name, not a PIN.

The Blue Book has no construction that repeats a name for identical components:
(the Blue Book) defines adducts of SEPARATE molecular entities ('coronene—1,3,5-
trinitrobenzene (1/1) (PIN)'). The paper's label semantics (Methods, 'Tiers'): pin_verified
only for a name the strict PIN path built and verified; "systematic_verified means a correct
systematic name that is not the PIN". 'ethanol ethanol' shipped pin_verified at both tiers; the
default tier now declines it and best-effort keeps it (OPSIN 2.9.0 reads it to the multi-copy
structure, full InChIKey exact). A two-component adduct of different entities is unchanged.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("CCO.CCO", "ethanol ethanol"),
    ("CC(N)=O.CC(N)=O", "acetamide acetamide"),
    ("COC(C)=O.COC(C)=O", "methyl acetate methyl acetate"),
])
def test_identical_components_are_not_labelled_a_pin(smiles, name):
    d = name_default(smiles)
    assert d.get("tier") == "abstain" and d.get("limit_code") == "NO_VERIFIED_PIN", d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier"), b.get("is_pin")) == (name, "systematic_verified", False), b
    assert name_is_rt_exact(name, smiles)


def test_an_adduct_of_different_entities_keeps_its_pin():
    assert_pin_at_both_tiers("CCO.CO", "ethanol—methanol (1/1)")
