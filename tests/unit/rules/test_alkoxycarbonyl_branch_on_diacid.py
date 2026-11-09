"""An ester on a branch carbon of a chain diacid is one 'alkoxycarbonyl' prefix (program 15 lead, 2026-11-09).

 "Esters cited as prefixes" (the Blue Book): beside a senior acid the ester is the prefix
'alkoxycarbonyl'. The polyfunctional producer credited the ester match only (it stops one atom into the alkoxy group),
so the substituent walk named the same branch again as a phantom '(hydroxymethyl)' and built another molecule
('3-(ethoxycarbonyl)-3-(hydroxymethyl)pentanedioic acid', rejected by the read-back): the PIN tier declined.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CCOC(=O)C(CC(=O)O)CC(=O)O", "3-(ethoxycarbonyl)pentanedioic acid"),
    ("CCCCCOC(=O)C(CC(=O)O)CC(=O)O", "3-[(pentyloxy)carbonyl]pentanedioic acid"),
    # unchanged: an acyloxy ester (the alkoxy side is the chain), an ester on the chain end
    ("CC(=O)OC(CC(=O)O)CC(=O)O", "3-(acetyloxy)pentanedioic acid"),
    ("COC(=O)C(C)C(=O)O", "3-methoxy-2-methyl-3-oxopropanoic acid"),
])
def test_the_alkoxycarbonyl_branch_is_named_once(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
