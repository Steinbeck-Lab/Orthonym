"""An acyclic branch on a fused ring core is named, not dropped.

``fused_rings._identify_fused_substituent`` knew sulfanyl and alkylsulfanyl branches on a sulfur root and sent ring-
bearing branches to the ring delegate; any other branch returned None and its caller skipped it, so the name lost the
branch and only the read-back stopped it. The branch is named by the shared substituent namer
(``rules.fused_ring_branch``): 'methanesulfonyl' as in '2-(methanesulfonyl)ethan-1-ol (PIN)', the
prefixes in alphanumerical order, the Blue Book). A name made through a promoted prefix ships below
the PIN (``pin_vocabulary.promote_at_pin_tier``).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CNC(=O)c1ccc2c(c1)ccn2S(C)(=O)=O", "1-(methanesulfonyl)-N-methyl-1H-indole-5-carboxamide"),
    ("CNC(=O)c1ccc2c(c1)ccn2S(=O)(=O)c1ccccc1", "1-(benzenesulfonyl)-N-methyl-1H-indole-5-carboxamide"),
])
def test_ring_n_sulfonyl_branch_is_named_in_order(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] != "abstain", row
    assert_full_rt(row["name"], smiles)
