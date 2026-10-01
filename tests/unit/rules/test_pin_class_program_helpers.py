import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers, assert_not_pin_labelled

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [("c1ccc2ncccc2c1", "quinoline"), ("CCO", "ethanol"),
                                         ("OC(=O)CSCC(=O)O", "2,2'-sulfanediyldiacetic acid")])
def test_helper_accepts_a_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_helper_rejects_a_planted_wrong_pin():
    with pytest.raises(AssertionError):
        assert_pin_at_both_tiers("c1ccc2ncccc2c1", "propan-2-ol")


def test_helper_flags_a_non_pin_label():
    with pytest.raises(AssertionError):
        assert_not_pin_labelled("c1ccc2ncccc2c1", "quinoline")
