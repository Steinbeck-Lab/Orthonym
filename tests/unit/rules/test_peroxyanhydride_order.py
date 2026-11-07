"""The acid words of a mixed peroxyanhydride in alphanumerical order, the Blue Book, with
"Mixed anhydrides":32253: "citing in alphabetical order the names of the two acids"; the order reads no locants or marks:
'chloroacetic 4-nitrobenzene-1-sulfonic anhydride (PIN)':32270). The raw string order put '3-aminopropanoic' before
'acetic' on a pin_verified name."""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)OOC(=O)CCN", "acetic 3-aminopropanoic peroxyanhydride"),
    ("O=C(OOC(=O)c1ccccc1)c1ccccc1Cl", "benzoic 2-chlorobenzoic peroxyanhydride"),
    ("CC(=O)OOC(=O)CC", "acetic propanoic peroxyanhydride"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_an_acyl_disulfide_beside_a_senior_acid_keeps_its_name():
    # the acyl-S-S-acyl group stays a prefix of the senior acid (a fold into the anhydride class lost it)
    row = tier_row("CC(=O)SSC(=O)CCC(=O)O", "pin")
    assert (row["name"], row["tier"]) == ("4-(acetyldisulfanyl)-4-oxobutanoic acid", "pin_verified"), row
