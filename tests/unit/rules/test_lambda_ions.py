"""Ions of a hydride with a nonstandard bonding number (``rules/lambda_convention.py``: ``ionic_lambda``,
``lambda_ion_name``).

 "Nonstandard bonding numbers" (the Blue Book) cites the λ; "General rule for systematically
naming cationic centers in parent hydrides" keeps it on the cation: 'tetrafluoro(methyl)-λ4-sulfanium (PIN)'
(:41384). "Anions formed by addition of hydride ions" names an anion that holds one bond more than its
standard hydride by 'uide', without a λ: 'tetramethylphosphanuide (PIN) tetramethyl-λ5-phosphanide' (:41106). With
a double bond the book's cation of a characteristic group takes another hydride ('(1,4-phenylene)bis(dioxo-
λ6-sulfanylium) (PIN)',,:42174), so a λ ion with a multiple bond is named but not as the PIN.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.lambda_convention import ionic_lambda
from tests.support.default_tier import default_tier_rule_applies

_BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


def _charged(smiles):
    return next(a for a in Chem.MolFromSmiles(smiles).GetAtoms() if a.GetFormalCharge())


@pytest.mark.parametrize("smiles,expected", [
    ("C[S+](C)(C)(C)C", 4),     # five bonds: the 'ium' of λ4-sulfane
    ("C[S+](C)(C)C", 4),        # four bonds and a radical electron
    ("C[S+](C)C", None),        # three bonds: the 'ium' of sulfane
    ("C[P-](C)C", None),        # three bonds and a radical electron: the 'uide' of phosphane
    ("C[N+](C)(C)C", None),     # four bonds: the 'ium' of azane
    ("C[CH+]C", None),          # carbon has no λ
])
def test_ionic_lambda(smiles, expected):
    assert ionic_lambda(_charged(smiles)) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[S+](C)(C)(C)C", "pentamethyl-λ4-sulfanium"),
    ("C[Se+](C)(C)(C)C", "pentamethyl-λ4-selanium"),
    ("C[S+](C)(C)(C)c1ccccc1", "tetramethyl(phenyl)-λ4-sulfanium"),
    ("C[S+](C)(C)(C)CC(=O)[O-]", "(tetramethyl-λ4-sulfaniumyl)acetate"),
])
def test_a_five_bond_chalcogen_cation_is_a_lambda4_ion(smiles, expected):
    with jvm_slots(1, purpose="test-lambda-ions"):
        row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C=[S+](C)(C)C", "trimethyl(methylidene)-λ4-sulfanium"),
    ("C=[S+](C)C", "dimethyl(methylidene)-λ4-sulfaniumyl"),
])
def test_a_lambda_ion_with_a_double_bond_is_not_the_pin(smiles, expected):
    with jvm_slots(1, purpose="test-lambda-ions"):
        row = Orthonym(**_BEST_EFFORT).name_tiered(smiles)
        pin = Orthonym().name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "systematic_verified" and not row["is_pin"], row
    if default_tier_rule_applies():
        assert pin["tier"] == "abstain" and pin["limit_code"] == "NO_VERIFIED_PIN", pin


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[S+](C)C", "trimethylsulfanium"),
    ("C[S+](C)CC(=O)[O-]", "(dimethylsulfaniumyl)acetate"),
    ("C[O+](C)C", "trimethyloxidanium"),
])
def test_a_standard_ion_has_no_lambda(smiles, expected):
    with jvm_slots(1, purpose="test-lambda-ions"):
        row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
