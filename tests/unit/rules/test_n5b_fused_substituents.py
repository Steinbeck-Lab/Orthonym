"""An all-carbon fused ring system as a '-yl' prefix of a chain or ester parent (lane N5b step 1).

 (the Blue Book) gives the system a fusion name, and (:15814) numbers the
free valence "as low as is consistent with any established numbering of the parent hydride"; the
hydro prefixes follow (:17326): "low locants go first to the fixed numbering of the
system, then indicated hydrogen, followed by free valence suffix, and finally 'hydro' prefixes"."""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

CASES = [
    ("OC(=O)CC1CCC2CCCC2C1", "(octahydro-1H-inden-5-yl)acetic acid"),     # (bicyclo[4.3.0]nonan-4-yl)acetic acid
    ("OC(=O)CC1CCCC2CCCC12", "(octahydro-1H-inden-4-yl)acetic acid"),
    ("CC(=O)OC1CCC2CCCC2C1", "octahydro-1H-inden-5-yl acetate"),          # bicyclo[4.3.0]nonan-4-yl acetate
]


@pytest.mark.parametrize("smiles,expected", CASES)
def test_the_ring_prefix_is_a_fusion_name_at_both_tiers(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


def test_a_prefix_the_engine_already_names_is_unchanged():
    smiles = "OC(=O)CC1CCC2CCCCC2C1"
    assert default_tier_row(smiles)["name"] == "(decahydronaphthalen-2-yl)acetic acid"
    assert name_best_effort(smiles)["name"] == "(decahydronaphthalen-2-yl)acetic acid"
