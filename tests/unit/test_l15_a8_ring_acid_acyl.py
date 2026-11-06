"""An acyl branch on a ring acid keeps the acid suffix at every wider tier (lane/l15, item 11 lead 5).

 "Carboxylic acids... having a carboxy group attached to a ring" (the Blue Book): "Carboxy groups
attached to cyclic parent hydrides... are always named by using the suffix 'carboxylic acid'". The acid outranks the
ketone, Table 4.1,:18169) and the acyl group of a ring-bearing acid is a prefix of the acid:
Definitions and name formation (:30424) names '1-oxo...' prefixes as general nomenclature only. A wider tier named the
molecule in one pass, the strict producer declined the ring-bearing acyl branch with the 'substituent' sentinel, and the
acid fell to a 'carboxy' prefix on a benzene or cyclohexane parent ('[(2S)-1-(4-carboxycyclohexyl)-1-oxopropan-2-yl]
benzene'). The substituent namer now asks the strict ring producer (``strict_ring_prefix``) at the wider tiers, so the
branch is spelled as the PIN tier spells it ('4-[(2S)-2-phenylpropanoyl]cyclohexane-1-carboxylic acid').
"""
import pytest

from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_WIDER = ["valid", "complete", "best-effort"]


_RING_ACID_ACYL = [
    # the three targets of the lead: an acyl branch with a stereocentre, an arylacetyl, an aroyl
    ("OC(=O)C1CCC(CC1)C(=O)[C@@H](C)c1ccccc1", "4-[(2S)-2-phenylpropanoyl]cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)Cc1ccccc1", "4-(phenylacetyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCCCC1C(=O)c1ccccc1", "2-benzoylcyclohexane-1-carboxylic acid"),
    # the same class on other ring acids and with other acyl kinds (all in the strict acyl writer's reach)
    ("OC(=O)c1ccc(cc1)C(=O)[C@@H](C)c1ccccc1", "4-[(2S)-2-phenylpropanoyl]benzoic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)C=Cc1ccccc1", "4-(3-phenylprop-2-enoyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)c1ccc2ccccc2c1C(=O)Cc1ccccc1", "1-(phenylacetyl)naphthalene-2-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)c1ccncc1", "4-(pyridine-4-carbonyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)NCc1ccccc1", "4-(benzylcarbamoyl)cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC1)C(=O)Oc1ccccc1", "4-(phenoxycarbonyl)cyclohexane-1-carboxylic acid"),
]


@pytest.mark.parametrize("tier", _WIDER)
@pytest.mark.parametrize("smiles,name", _RING_ACID_ACYL)
def test_ring_acid_acyl_branch_at_valid_and_complete(smiles, name, tier):
    """ (:29886): the ring acid keeps 'carboxylic acid'; the branch is an acyl prefix, not 'carboxy' on a
    benzene parent."""
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert "carboxy" not in row["name"].replace("carboxylic", ""), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", _RING_ACID_ACYL)
def test_the_pin_tier_rows_are_unchanged(smiles, name):
    """The PIN tier already named these: its row is the one the wider tiers now give."""
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("tier", ["pin"] + _WIDER)
def test_a_chain_acid_with_a_ketone_keeps_its_name(tier):
    """Control: (:18169) acids above ketones, the chain acid '4-oxo-6-phenylhexanoic acid' (no ring acid, no
    ring-bearing acyl branch) is named as before."""
    row = tier_row("OC(=O)CCC(=O)CCc1ccccc1", tier)
    assert (row["name"], row["tier"]) == ("4-oxo-6-phenylhexanoic acid", "pin_verified"), row


@pytest.mark.parametrize("tier", _WIDER)
def test_a_branch_the_strict_producer_declines_keeps_the_wider_tier_name(tier):
    """Control: a spiro ring acid with an acyl branch has no PIN-tier name (the strict producer declines the branch);
    the wider tiers keep the systematic name they gave, read back whole."""
    smiles = "OC(=O)C1CCC2(CC1)CC(C(=O)Cc1ccccc1)C2"
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (
        "2-(phenylacetyl)spiro[3.5]nonane-7-carboxylic acid", "systematic_verified"), row
    assert_full_rt(row["name"], smiles)
