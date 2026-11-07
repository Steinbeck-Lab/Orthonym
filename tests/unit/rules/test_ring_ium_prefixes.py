"""A protonated ring nitrogen whose neutral molecule takes a suffix is the ring-'ium' parent (``rules.ring_ium_prefixes``).

 (the Blue Book-18170): '6 Cations' before '7 Acids'; the ring hydride takes the '-ium' suffix
 :41366) and the acid is the 'carboxy' prefix ('4-carboxy-1-methylpyridin-1-ium chloride (PIN)',
 :29914); an acyl group is the acyl prefix:30438). The ring-acyl rows were the
pseudoketone-based general-engine names below the PIN; the ring-acid rows were labelled below the PIN.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CCOCC(=O)N1CC[NH+](CC)CC1", "4-(ethoxyacetyl)-1-ethylpiperazin-1-ium"),
    ("C[NH+]1CCN(C(=O)c2ccccc2)CC1", "4-benzoyl-1-methylpiperazin-1-ium"),
    ("CC(=O)N1CC[NH+](C)CC1", "4-acetyl-1-methylpiperazin-1-ium"),
    ("C[NH+]1CCC(C(=O)O)CC1", "4-carboxy-1-methylpiperidin-1-ium"),
    ("OC(=O)c1cc[nH+]cc1", "4-carboxypyridin-1-ium"),
    ("C[NH+]1CCC(C(=O)OC)CC1", "4-(methoxycarbonyl)-1-methylpiperidin-1-ium"),
    # a quaternary ring nitrogen (no hydrogen) takes the same acyl prefix
    ("CCC(=O)c1cc[n+](C)cc1", "1-methyl-4-propanoylpyridin-1-ium"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_a_bare_ring_cation_keeps_the_neutral_ring_route():
    assert tier_row("CC[NH+]1CCCCC1", "pin")["name"] == "1-ethylpiperidin-1-ium"


def test_a_protonated_enamine_ring_nitrogen_claims_no_pin():
    # the resonance form is an iminium; (d) (the Blue Book) does not decide it (LO5 decision)
    assert tier_row("CC1(CC(=CC(=N)C1)[NH+]2CCCCC2)C", "pin")["tier"] != "pin_verified"


def test_an_azonia_name_is_never_labelled_the_pin():
    # 'SKELETAL REPLACEMENT ('a') NOMENCLATURE FOR CATIONS' (the Blue Book): "Method (1) gives preferred
    # IUPAC names" (:41769; '1-methyl-1-azabicyclo[2.2.1]heptan-1-ium chloride (PIN)', not '...azoniabicyclo...')
    row = tier_row("OC(=O)C1C[NH+]2CCC1CC2", "best-effort")
    assert "azonia" in row["name"] and row["tier"] != "pin_verified", row
    assert tier_row("C1C[NH+]2CCC1CC2", "pin")["name"] == "1-azabicyclo[2.2.2]octan-1-ium"
