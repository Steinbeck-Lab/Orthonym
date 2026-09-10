""": multi-ring + competing-FG partition. A ring parent bearing the principal
group (acid) plus an aryl substituent bearing a SECOND characteristic group (amide)
must NOT double-count that group: `fallback_chain_ring` emitted a stray unlocanted
`carbamoyl-` parent prefix AND the substituent name `(4-carbamoylphenyl)` (which
already includes it) -> unparseable `carbamoyl-4-(4-carbamoylphenyl)cyclohexane...`
-> abstain. The FG-on-a-ring-substituent must be filtered from the parent
prefix set (the branch name covers it), regardless of the substituent's carbon count
(the old 1<=c<=3 bound missed the 6+-carbon aryl ring). Halogens already filtered.
"""
import pytest
from tests.support.rt_assert import assert_rt_exact


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", [
    "OC(=O)C1CCC(c2ccc(C(=O)N)cc2)CC1",   # 4-(4-carbamoylphenyl)cyclohexane-1-carboxylic acid
    "OC(=O)C1CCC(c2ccc(C(=O)O)cc2)CC1",    # 4-(4-carboxyphenyl)... (acid on the sub-ring)
    "OC(=O)C1CCC(c2ccc(C(C)=O)cc2)CC1",    # 4-(4-acetylphenyl)... (ketone on the sub-ring)
])
def test_multiring_competing_fg_no_double_count(smiles):
    assert_rt_exact(smiles)
