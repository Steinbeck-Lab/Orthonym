"""An ester of a junior acid on the acid side of a senior chain ester is its acyloxy prefix (2026-11-09).

 (the Blue Book): of the esters of a polyester the senior one is the functional class name and the
others are prefixes ('phenyl 3-(benzoyloxy)benzoate (PIN) [not 3-(phenoxycarbonyl)phenyl benzoate...]',:31873). The
strict path classified these as independent esters and built '(acetyloxy) methyl propanoate', which the read-back
rejected; only a decomposition re-run named them, below the PIN.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)OC(C)C(=O)OC", "methyl 2-(acetyloxy)propanoate"),
    ("CC(=O)OCCC(=O)OC", "methyl 3-(acetyloxy)propanoate"),
    ("CCOC(=O)[C@@H](C)OC(=O)C", "ethyl (2R)-2-(acetyloxy)propanoate"),
    # unchanged: a ring acid ester with an acyloxy prefix
    ("COC(=O)c1ccccc1OC(C)=O", "methyl 2-(acetyloxy)benzoate"),
])
def test_ester_in_ester_pin(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles, name", [
    # three esters of ONE polyacid are named on that acid, not on one of its esters,:31781/:31783)
    ("COC(=O)C(C)CC(C(=O)OC)C(=O)OC", "trimethyl butane-1,1,3-tricarboxylate"),
    ("CCOC(=O)C(CC(C)C(=O)OC)C(=O)OCC", "1,1-diethyl 3-methyl butane-1,1,3-tricarboxylate"),
])
def test_esters_of_one_polyacid_stay_on_the_acid(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
