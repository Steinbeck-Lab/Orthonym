"""v31 composition-completeness lever — best-effort rt_exact tests.

Root cause (measured 2026-08-10): the PIN ring-substituent path
(`classify_and_name_fragment` -> `name_ring_system_substituent` ->
`name_substituent`) was called with `allow_mancude=False` even under the
best-effort tier, so a decorated (hetero)aryl branch that `name_substituent`
CAN build under `allow_mancude=True` was refused, and the whole molecule
abstained. The fix threads the best-effort context (`best_effort_ctx`) into that
call so `allow_mancude` is True under best-effort — PIN default stays
`allow_mancude=False` -> byte-identical.

Acceptance = full isomeric round-trip (rt_exact); see tests/support/rt_assert.py.
"""
import pytest
from tests.support.rt_assert import assert_rt_exact


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles", [
    # decorated-(hetero)aryl-methyl branches on a ring parent — the class the
    # best-effort flag-threading fix unblocks. Each round-trips exactly.
    "OC(=O)C1CCC(Cc2ccccc2OC)CC1",        # 4-[(2-methoxyphenyl)methyl]cyclohexane-1-carboxylic acid
    "OC(=O)C1CCC(Cc2ccc(Cl)cc2)CC1",      # 4-[(4-chlorophenyl)methyl]cyclohexane-1-carboxylic acid
    "OC(=O)C1CCC(Cc2cccc(F)c2)CC1",       # 4-[(3-fluorophenyl)methyl]cyclohexane-1-carboxylic acid
    "OC(=O)C1CCC(Cc2ccc(OC)cc2)CC1",      # 4-[(4-methoxyphenyl)methyl]cyclohexane-1-carboxylic acid
    "OC(=O)C1CCC(Cc2ccncc2)CC1",          # 4-[(pyridin-4-yl)methyl]cyclohexane-1-carboxylic acid
    "OCC1CCC(Cc2ccc(Cl)cc2)CC1",          # ({4-[(4-chlorophenyl)methyl]cyclohexyl})methanol
])
def test_decorated_aryl_methyl_branch_on_ring_parent(smiles):
    assert_rt_exact(smiles)


@pytest.mark.roundtrip
@pytest.mark.xfail(reason="separate pre-existing partition bug: the sulfonyl S is "
                          "double-counted (branch names [4-(methanesulfonyl)phenyl]methyl "
                          "AND a spurious ring 'sulfonyl' prefix). rt_exact gate abstains "
                          "(0-wrong); follow-up = ring-substituent partition dedup.",
                   strict=True)
def test_r1_sulfonylbenzyl_partition_followup():
    assert_rt_exact("OC(=O)C1CCC(Cc2ccc(S(C)(=O)=O)cc2)CC1")
