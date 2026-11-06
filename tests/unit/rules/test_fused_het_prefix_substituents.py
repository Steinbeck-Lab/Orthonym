"""A fused heterocycle cited as a substituent prefix keeps every substituent of its core.

``assembly.composer._generate_ring_substituent_prefixes`` cited a matched fused heterocyclic core by its bare prefix
whenever ``_detect_fused_het_inner_subs`` found no substituent it could name, so a phenyl (or any group outside its
branches) on the core was dropped: '3-(1-benzofuran-5-yl)propan-1-ol' for the 2-phenylbenzofuran, withdrawn only by the
round trip. The detector now answers None for a substituent it does not name and the general ring namer names the
decorated core; the bare prefix is for a core without substituents only. A substituent prefix names the whole group
,: no atom of the structure may be left out of the name.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("OCCCc1ccc2oc(-c3ccccc3)cc2c1", "3-(2-phenyl-1-benzofuran-5-yl)propan-1-ol"),
    ("OCCc1ccc2[nH]c(-c3ccccc3)cc2c1", "2-(2-phenyl-1H-indol-5-yl)ethan-1-ol"),
    ("COc1cc(CCCO)cc2cc(-c3ccc4c(c3)OCO4)oc12",
     "3-[2-(2H-1,3-benzodioxol-5-yl)-7-methoxy-1-benzofuran-5-yl]propan-1-ol"),
    ("OCCCc1ccc2oc(C(C)(C)C)cc2c1", "3-(2-tert-butyl-1-benzofuran-5-yl)propan-1-ol"),
    # a substituent on the attachment atom of the core is a substituent too
    ("OCCCC1(C)CCc2ccccc2O1", "3-(2-methyl-3,4-dihydro-2H-1-benzopyran-2-yl)propan-1-ol"),
    ("Cc1c(C)c2c(c(C)c1O)CCC(C)(CCCC(C)C(=O)O)O2",
     "5-(6-hydroxy-2,5,7,8-tetramethyl-3,4-dihydro-2H-1-benzopyran-2-yl)-2-methylpentanoic acid"),
    # the cores the detector names keep their names
    ("OCCCc1ccc2oc(C)cc2c1", "3-(2-methyl-1-benzofuran-5-yl)propan-1-ol"),
    ("OCCCc1ccc2oc(OC)cc2c1", "3-(2-methoxy-1-benzofuran-5-yl)propan-1-ol"),
    ("OCCCc1ccc2occc2c1", "3-(1-benzofuran-5-yl)propan-1-ol"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)
