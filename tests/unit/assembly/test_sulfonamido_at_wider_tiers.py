"""The wider tiers cite an -NH-SO2-R branch as the method (1) '{R}sulfonamido' prefix, as the PIN tier does.

 "Substituents of the types -NH-CO-R and -NH-SO2-R" (the Blue Book): "Method (1) generates
preferred IUPAC names": '3-(methanesulfonamido)propanoic acid (PIN)' (:33019). The functional-group namer of the
wider tiers (``n_branch_prefixes.hetero_root_prefix``) used to write method (2), '{[...sulfonyl]amino}'.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CC(C(=O)NC1=CC(=CC=C1)Cl)OC(=O)CNS(=O)(=O)/C=C/C2=CC=CC=C2",
     "1-(3-chloroanilino)-1-oxopropan-2-yl [(1E)-2-phenylethene-1-sulfonamido]acetate"),
    ("CS(=O)(=O)NCCC(=O)O", "3-(methanesulfonamido)propanoic acid"),
], ids=lambda v: v)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_both_tiers_give_the_sulfonamido_prefix(smiles, name, tier):
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
