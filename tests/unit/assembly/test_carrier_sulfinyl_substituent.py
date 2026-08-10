"""v31 composition lever: a sulfoxide/sulfone substituent that attaches through a
CARBON CARRIER (e.g. benzyl methyl sulfoxide -CH2-S(=O)-CH3) must NOT be named as a
bare (R)sulfinyl/(R)sulfonyl prefix -- that drops the carrier carbon and denotes a
different molecule. The prefix form is valid only when the fragment attaches through
the sulfur; otherwise the carrier carbon carries an (alkylsulfinyl) decoration.

Same bug shape as the pre-existing ether (`-CH2-O-CH3` != `methoxy`) and isocyanate
carrier guards in substituent_prefix_forms._check_substituent_prefix_form.
"""
import pytest
from tests.support.rt_assert import assert_rt_exact


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", [
    "O=S(C)Cc1ccccc1",      # (methanesulfinylmethyl)benzene (benzyl methyl sulfoxide)
    "O=S(=O)(C)Cc1ccccc1",  # (methanesulfonylmethyl)benzene (benzyl methyl sulfone)
])
def test_carrier_carbon_sulfinyl_sulfonyl(opsin_gate, smiles):
    assert_rt_exact(smiles)
