"""The general engine's fused parent cites a simple suffix group (``assembly.fused_ring_suffix``).

 (the Blue Book): fusion names for systems of two or more rings of five or more members; the suffix locants
come first after indicated hydrogen (c)). These best-effort names were von Baeyer polyenes
('bicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene-3-carboxylic acid').
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1cc2ccccc2cc1CC(=O)O", "3-(carboxymethyl)naphthalene-2-carboxylic acid"),
    ("OC(=O)c1cc2ccccc2cc1C(=O)NN", "3-(aminocarbamoyl)naphthalene-2-carboxylic acid"),
])
def test_best_effort_spells_the_fusion_parent(smiles, expected):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == expected and "bicyclo" not in row["name"], row
    assert row["tier"] != "pin_verified", row   # the general engine claims no PIN


def test_a_branch_with_more_acid_groups_than_the_ring_stays_the_parent():
    # the most principal groups: the propanedioic acid chain, not the ring
    assert tier_row("OC(=O)c1cc2ccccc2cc1C(C(=O)O)C(=O)O", "best-effort")["name"].endswith("propanedioic acid")


@pytest.mark.parametrize("smiles,expected", [
    # the parent resolver reads linear 5-5-5 as a bicyclic annulene (an inner bond lost): the von Baeyer name stays
    ("SC1CC2CC3CCCC3C2C1", "tricyclo[6.3.0.0^2,6]undecane-4-thiol"),
    # a hyphen before an italic prefix: (d) (the Blue Book), '*as*-indacene' (:6966)
    ("SC1CC2CCC3CCCC3C2C1", "dodecahydro-as-indacene-2-thiol"),
])
def test_review_rows(smiles, expected):
    assert tier_row(smiles, "best-effort")["name"] == expected
